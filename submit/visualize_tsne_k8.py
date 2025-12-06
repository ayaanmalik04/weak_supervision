import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from types import SimpleNamespace
from collections import defaultdict

from transformers import AutoProcessor
from baseline1_fewshot import (
    UCF101Dataset, collate_fn, extract_features,
    get_features_cache_path, train_pytorch_classifier
)
from train_pose_mil import (
    TransformerMIL, PoseMILDataset,
    collate_fn as mil_collate_fn, get_ucf101_classes
)


def extract_mil_embeddings(model, dataloader, device):
    model.eval()
    all_emb, all_lab, all_ids = [], [], []

    with torch.no_grad():
        for features, mask, labels, clip_ids in tqdm(dataloader):
            features, mask = features.to(device), mask.to(device)
            m = model.module if isinstance(model, nn.DataParallel) else model

            key_padding_mask = (mask == 0)
            x = m.embedding(features)
            x = m.pos_encoder(x)
            h = m.transformer(x, src_key_padding_mask=key_padding_mask)

            att_v = torch.tanh(m.att_V(h))
            att_u = torch.sigmoid(m.att_U(h))
            att_logits = m.att_weights(att_v * att_u).squeeze(-1)
            att_logits = att_logits.masked_fill(mask == 0, -1e9)
            att_weights = F.softmax(att_logits, dim=1)
            r = torch.bmm(att_weights.unsqueeze(1), h).squeeze(1)

            all_emb.append(r.cpu().numpy())
            all_lab.extend(labels.numpy())
            all_ids.extend(clip_ids)

    return np.concatenate(all_emb), np.array(all_lab), all_ids


def run_tsne(features, perplexity=30):
    scaler = StandardScaler()
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=1000,
        random_state=42,
        init='pca',
        learning_rate='auto'
    )
    return tsne.fit_transform(scaler.fit_transform(features))


def plot_embeddings(emb, labels, class_names, title, ax, selected):
    class_to_idx = {n: i for i, n in enumerate(class_names)}
    sel_idx = [class_to_idx.get(c, -1) for c in selected if c in class_to_idx]
    cmap = plt.cm.get_cmap('tab10')

    mask_other = ~np.isin(labels, sel_idx)
    ax.scatter(emb[mask_other, 0], emb[mask_other, 1],
               c='lightgray', s=8, alpha=0.3, label='Other')

    for i, (cidx, cname) in enumerate(zip(sel_idx, selected)):
        if cidx == -1:
            continue
        m = labels == cidx
        if m.sum() > 0:
            ax.scatter(emb[m, 0], emb[m, 1],
                       c=[cmap(i % 10)], s=30, alpha=0.8,
                       label=cname, edgecolors='white', linewidths=0.5)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    args = SimpleNamespace(
        cache_dir="/mnt/amlfs-03/shared/ayaanm/yanav/cache",
        model_name="microsoft/xclip-base-patch16",
        mil_npz_path="vitpose_sub8_train_joint_chunk_vels.npz",
        mil_checkpoint="weak_motion_labels_train_model.pth",
        k_shot=8,
        n_vis_samples=2000,
        output="tsne_k8_comparison.png",
        batch_size=16,
        num_workers=8,
        use_multi_gpu=True,
        num_frames=32,
        input_size=224,
        scale_resize=256,
        use_train_augmentation=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(args.model_name)

    train_dataset = UCF101Dataset(
        split="train", processor=processor, num_frames=32, input_size=224,
        scale_resize=256, use_train_augmentation=False,
        cache_dir=args.cache_dir, train_subset=1.0, seed=42
    )

    class_names, class_to_idx = get_ucf101_classes()
    pose_dataset = PoseMILDataset(args.mil_npz_path, class_to_idx, is_training=False)

    train_clip_ids = train_dataset.clip_ids
    pose_clip_set = set(pose_dataset.valid_clips)

    matched_indices, matched_clip_ids, matched_labels = [], [], []
    for i, clip_id in enumerate(train_clip_ids):
        if clip_id in pose_clip_set:
            matched_indices.append(i)
            matched_clip_ids.append(clip_id)
            matched_labels.append(train_dataset.labels[i])

    np.random.seed(42)
    if len(matched_indices) > args.n_vis_samples:
        sample_idx = np.random.choice(len(matched_indices), args.n_vis_samples, replace=False)
        matched_indices = [matched_indices[i] for i in sample_idx]
        matched_clip_ids = [matched_clip_ids[i] for i in sample_idx]
        matched_labels = [matched_labels[i] for i in sample_idx]

    matched_labels = np.array(matched_labels)

    class_to_indices = defaultdict(list)
    for i, label in enumerate(train_dataset.labels):
        class_to_indices[label].append(i)

    np.random.seed(42)
    labeled_indices, val_indices = [], []
    for class_idx in sorted(class_to_indices.keys()):
        indices = class_to_indices[class_idx]
        np.random.shuffle(indices)
        labeled_indices.extend(indices[:args.k_shot])
        val_indices.extend(indices[args.k_shot:args.k_shot + max(2, len(indices)//10)])

    cache_dir = Path(args.cache_dir)

    labeled_loader = DataLoader(
        Subset(train_dataset, labeled_indices),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )
    X_labeled, y_labeled = extract_features(
        None, processor, labeled_loader, device,
        use_multi_gpu=args.use_multi_gpu, args=args,
        dataset=train_dataset, indices=labeled_indices,
        cache_path=get_features_cache_path(cache_dir, "labeled", args.k_shot, 42, args.model_name)
    )

    val_loader = DataLoader(
        Subset(train_dataset, val_indices),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )
    X_val, y_val = extract_features(
        None, processor, val_loader, device,
        use_multi_gpu=args.use_multi_gpu, args=args,
        dataset=train_dataset, indices=val_indices,
        cache_path=get_features_cache_path(cache_dir, "val", args.k_shot, 42, args.model_name)
    )

    classifier = train_pytorch_classifier(
        X_labeled, y_labeled, X_val, y_val,
        X_weak=None, y_weak_soft=None,
        num_classes=len(class_names), lr=0.01, epochs=500,
        batch_size=256, patience=20, device=device
    )

    vis_loader = DataLoader(
        Subset(train_dataset, matched_indices),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )
    X_vis, _ = extract_features(
        None, processor, vis_loader, device,
        use_multi_gpu=args.use_multi_gpu, args=args,
        dataset=train_dataset, indices=matched_indices,
        cache_path=get_features_cache_path(cache_dir, f"vis_{args.n_vis_samples}", args.k_shot, 42, args.model_name)
    )

    mil_model = TransformerMIL(
        input_dim=68, hidden_dim=256, num_classes=len(class_names),
        num_heads=4, num_layers=2, dropout=0.4
    )
    if Path(args.mil_checkpoint).exists():
        mil_model.load_state_dict(torch.load(args.mil_checkpoint, map_location="cpu"))
    mil_model = mil_model.to(device)

    pose_subset = PoseMILDataset(args.mil_npz_path, class_to_idx, is_training=False)
    pose_subset.valid_clips = matched_clip_ids
    pose_subset.labels = matched_labels.tolist()

    pose_loader = DataLoader(
        pose_subset, batch_size=64, shuffle=False,
        collate_fn=mil_collate_fn, num_workers=8
    )
    X_mil, _, _ = extract_mil_embeddings(mil_model, pose_loader, device)

    xclip_2d = run_tsne(X_vis)
    mil_2d = run_tsne(X_mil)

    highlight = [
        'Basketball', 'PushUps', 'PlayingGuitar', 'Typing', 'Diving',
        'JumpRope', 'WalkingWithDog', 'Haircut', 'Bowling', 'HorseRiding'
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    plot_embeddings(xclip_2d, matched_labels, class_names, 'X-CLIP Features (512-D)', axes[0], highlight)
    plot_embeddings(mil_2d, matched_labels, class_names, 'MIL Pose Features (256-D)', axes[1], highlight)

    handles, labels_ = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels_, loc='lower center', ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.02))
    plt.suptitle(f't-SNE: X-CLIP vs MIL (n={len(matched_labels)})', y=1.02)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')


if __name__ == "__main__":
    main()
