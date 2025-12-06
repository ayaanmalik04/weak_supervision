import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from types import SimpleNamespace

from transformers import AutoProcessor
from baseline1_fewshot import UCF101Dataset, collate_fn, extract_features, get_features_cache_path
from train_pose_mil import TransformerMIL, PoseMILDataset, collate_fn as mil_collate_fn, get_ucf101_classes


def get_mil_predictions(model, dataloader, device):
    model.eval()
    preds, labels, probs, clip_ids = [], [], [], []

    with torch.no_grad():
        for features, mask, labs, cids in tqdm(dataloader):
            features, mask = features.to(device), mask.to(device)
            output = model(features, mask)
            logits = output[0] if isinstance(output, tuple) else output
            p = torch.softmax(logits, dim=1)
            preds.extend(p.argmax(dim=1).cpu().numpy())
            labels.extend(labs.numpy())
            probs.append(p.cpu().numpy())
            clip_ids.extend(list(cids))

    return np.array(preds), np.array(labels), np.concatenate(probs) if probs else np.zeros((0, 0)), np.array(clip_ids)


def plot_confusion_matrix(cm, class_names, title, ax, top_n=15):
    cm_off = cm.copy().astype(float)
    np.fill_diagonal(cm_off, 0)

    pairs = [
        (cm_off[i, j], i, j)
        for i in range(len(class_names))
        for j in range(len(class_names))
        if i != j and cm_off[i, j] > 0
    ]
    pairs.sort(reverse=True)

    sel = set()
    for _, i, j in pairs:
        sel.add(i)
        sel.add(j)
        if len(sel) >= top_n:
            break

    sel_idx = sorted(list(sel))[:top_n]
    cm_sub = cm[np.ix_(sel_idx, sel_idx)].astype(float)
    cm_norm = cm_sub / (cm_sub.sum(axis=1, keepdims=True) + 1e-10)
    names_sub = [class_names[i] for i in sel_idx]

    sns.heatmap(
        cm_norm, annot=False, cmap='Blues',
        xticklabels=names_sub, yticklabels=names_sub,
        ax=ax, cbar_kws={'shrink': 0.8}, vmin=0, vmax=1
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)


def plot_full_confusion_matrix(cm, title, ax):
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
    im = ax.imshow(cm_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.8)


def main():
    args = SimpleNamespace(
        cache_dir="/mnt/amlfs-03/shared/ayaanm/yanav/cache",
        model_name="microsoft/xclip-base-patch16",
        mil_npz_path="vitpose_sub8_train_joint_chunk_vels.npz",
        mil_checkpoint="weak_motion_labels_train_model.pth",
        n_samples=2000,
        output="confusion_matrices.png",
        use_multi_gpu=True,
        num_frames=32,
        input_size=224,
        scale_resize=256,
        use_train_augmentation=False,
        batch_size=16,
        num_workers=8
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
    if len(matched_indices) > args.n_samples:
        sample_idx = np.random.choice(len(matched_indices), args.n_samples, replace=False)
        matched_indices = [matched_indices[i] for i in sample_idx]
        matched_clip_ids = [matched_clip_ids[i] for i in sample_idx]
        matched_labels = [matched_labels[i] for i in sample_idx]

    sorted_order = np.argsort(matched_indices)
    matched_indices = [matched_indices[i] for i in sorted_order]
    matched_clip_ids = [matched_clip_ids[i] for i in sorted_order]
    matched_labels = [matched_labels[i] for i in sorted_order]
    matched_labels = np.array(matched_labels)

    vis_loader = DataLoader(
        Subset(train_dataset, matched_indices),
        batch_size=16, shuffle=False, num_workers=8,
        collate_fn=collate_fn, pin_memory=True
    )

    X_features, _ = extract_features(
        None, processor, vis_loader, device,
        use_multi_gpu=args.use_multi_gpu, args=args,
        dataset=train_dataset, indices=matched_indices,
        cache_path=get_features_cache_path(Path(args.cache_dir), f"confmat_{args.n_samples}", 0, 42, args.model_name)
    )

    clf = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)
    clf.fit(X_features, matched_labels)
    xclip_preds = clf.predict(X_features)
    xclip_acc = accuracy_score(matched_labels, xclip_preds)

    mil_model = TransformerMIL(
        input_dim=68, hidden_dim=256, num_classes=len(class_names),
        num_heads=4, num_layers=2, dropout=0.4
    )
    mil_model.load_state_dict(torch.load(args.mil_checkpoint, map_location="cpu"))
    mil_model = mil_model.to(device)

    pose_subset = PoseMILDataset(args.mil_npz_path, class_to_idx, is_training=False)
    pose_subset.valid_clips = matched_clip_ids
    pose_subset.labels = matched_labels.tolist()

    pose_loader = DataLoader(
        pose_subset, batch_size=64, shuffle=False,
        collate_fn=mil_collate_fn, num_workers=8
    )

    mil_preds, _, mil_probs, _ = get_mil_predictions(mil_model, pose_loader, device)
    mil_acc = accuracy_score(matched_labels, mil_preds)

    cm_xclip = confusion_matrix(matched_labels, xclip_preds, labels=range(len(class_names)))
    cm_mil = confusion_matrix(matched_labels, mil_preds, labels=range(len(class_names)))

    xclip_proba = np.zeros((X_features.shape[0], len(class_names)), dtype=np.float32)
    for i, c in enumerate(clf.classes_):
        if 0 <= c < len(class_names):
            xclip_proba[:, c] = clf.predict_proba(X_features)[:, i]

    np.savez_compressed(
        "xclip_preds_confmat.npz",
        y_true=matched_labels, probs=xclip_proba,
        class_names=np.array(class_names), clip_ids=np.array(matched_clip_ids),
        features=X_features
    )
    np.savez_compressed(
        "mil_preds_confmat.npz",
        y_true=matched_labels, probs=mil_probs,
        class_names=np.array(class_names), clip_ids=np.array(matched_clip_ids)
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    plot_full_confusion_matrix(cm_xclip, f'X-CLIP Full Confusion Matrix (Acc: {xclip_acc:.1%})', axes[0, 0])
    plot_full_confusion_matrix(cm_mil, f'MIL Pose Full Confusion Matrix (Acc: {mil_acc:.1%})', axes[0, 1])
    plot_confusion_matrix(cm_xclip, class_names, 'X-CLIP: Top-15 Most Confused Classes', axes[1, 0])
    plot_confusion_matrix(cm_mil, class_names, 'MIL: Top-15 Most Confused Classes', axes[1, 1])

    plt.suptitle(f'Confusion Matrices (n={len(matched_labels)})', y=1.02)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')


if __name__ == "__main__":
    main()
