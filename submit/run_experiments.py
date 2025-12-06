import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from types import SimpleNamespace
from sklearn.metrics import accuracy_score

from transformers import AutoProcessor, AutoModel
from baseline1_fewshot import (
    UCF101Dataset, collate_fn, create_fewshot_split,
    extract_features, get_features_cache_path, train_pytorch_classifier
)


def run_experiment_1_baseline(args, train_dataset, test_dataset, model, processor, device):
    splits = create_fewshot_split(train_dataset, test_dataset, k_shot=2, val_ratio=0.15, seed=42)
    cache_dir = Path(args.cache_dir)

    labeled_loader = DataLoader(
        Subset(train_dataset, splits['labeled']),
        batch_size=16, shuffle=False, num_workers=16, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        Subset(train_dataset, splits['val']),
        batch_size=16, shuffle=False, num_workers=16, collate_fn=collate_fn, pin_memory=True
    )
    test_loader = DataLoader(
        Subset(test_dataset, splits['test']),
        batch_size=16, shuffle=False, num_workers=16, collate_fn=collate_fn, pin_memory=True
    )

    X_labeled, y_labeled = extract_features(
        model, processor, labeled_loader, device,
        use_multi_gpu=args.use_multi_gpu, args=args,
        dataset=train_dataset, indices=splits['labeled'],
        cache_path=get_features_cache_path(cache_dir, "labeled", 2, 42, args.model_name)
    )
    X_val, y_val = extract_features(
        model, processor, val_loader, device,
        use_multi_gpu=args.use_multi_gpu, args=args,
        dataset=train_dataset, indices=splits['val'],
        cache_path=get_features_cache_path(cache_dir, "val", 2, 42, args.model_name)
    )
    X_test, y_test = extract_features(
        model, processor, test_loader, device,
        use_multi_gpu=args.use_multi_gpu, args=args,
        dataset=test_dataset, indices=splits['test'],
        cache_path=get_features_cache_path(cache_dir, "test", 2, 42, args.model_name)
    )

    classifier = train_pytorch_classifier(
        X_labeled, y_labeled, X_val, y_val,
        X_weak=None, y_weak_soft=None,
        num_classes=101, lr=0.01, epochs=500, batch_size=256, patience=20, device=device
    )

    classifier.eval()
    with torch.no_grad():
        test_preds = classifier.predict(torch.FloatTensor(X_test).to(device)).cpu().numpy()

    return accuracy_score(y_test, test_preds), splits, X_labeled, y_labeled, X_val, y_val, X_test, y_test


def run_experiment_2_mil_weak(args, train_dataset, test_dataset, splits, X_labeled, y_labeled, X_val, y_val, X_test, y_test, model, processor, device):
    mil_labels_path = Path("weak_motion_labels_train.npz")
    if not mil_labels_path.exists():
        os.system("python3 train_pose_mil.py")
    if not mil_labels_path.exists():
        return None

    mil_data = np.load(mil_labels_path, allow_pickle=True)
    clip_id_to_mil = {str(cid): mil_data['Y_weak'][i] for i, cid in enumerate(mil_data['clip_ids'])}

    unlabeled_with_mil, mil_labels_for_unlabeled = [], []
    for idx in splits['unlabeled']:
        clip_id = train_dataset.clip_ids[idx]
        if clip_id in clip_id_to_mil:
            unlabeled_with_mil.append(idx)
            mil_labels_for_unlabeled.append(clip_id_to_mil[clip_id])

    if not unlabeled_with_mil:
        return None

    y_weak_soft = np.array(mil_labels_for_unlabeled)
    cache_dir = Path(args.cache_dir)

    unlabeled_loader = DataLoader(
        Subset(train_dataset, unlabeled_with_mil),
        batch_size=16, shuffle=False, num_workers=16, collate_fn=collate_fn, pin_memory=True
    )
    X_weak, _ = extract_features(
        model, processor, unlabeled_loader, device,
        use_multi_gpu=args.use_multi_gpu, args=args,
        dataset=train_dataset, indices=unlabeled_with_mil,
        cache_path=get_features_cache_path(cache_dir, "unlabeled_mil", 2, 42, args.model_name)
    )

    classifier = train_pytorch_classifier(
        X_labeled, y_labeled, X_val, y_val,
        X_weak=X_weak, y_weak_soft=y_weak_soft,
        num_classes=101, lr=0.01, epochs=500, batch_size=256, patience=20, device=device, labeled_weight=5.0
    )

    classifier.eval()
    with torch.no_grad():
        test_preds = classifier.predict(torch.FloatTensor(X_test).to(device)).cpu().numpy()

    return accuracy_score(y_test, test_preds)


def run_experiment_2_mae_weak(args, train_dataset, test_dataset, splits, X_labeled, y_labeled, X_val, y_val, X_test, y_test, model, processor, device):
    weak_labels_path = Path("weak_labels_mae.npz")
    pretrained_mae_path = Path("pose_mae_best.pth")

    if not weak_labels_path.exists():
        if pretrained_mae_path.exists():
            os.system(f"python3 finetune_pose_mae.py --epochs 50 --batch_size 128 --k_shot 2 --seed 42 --pretrained_path {pretrained_mae_path}")
        else:
            return None

    if not weak_labels_path.exists():
        return None

    mae_data = np.load(weak_labels_path, allow_pickle=True)
    clip_id_to_mae = {str(cid): mae_data['Y_weak'][i] for i, cid in enumerate(mae_data['clip_ids'])}

    unlabeled_with_mae, mae_labels_for_unlabeled = [], []
    for idx in splits['unlabeled']:
        clip_id = train_dataset.clip_ids[idx]
        if clip_id in clip_id_to_mae:
            unlabeled_with_mae.append(idx)
            mae_labels_for_unlabeled.append(clip_id_to_mae[clip_id])

    if not unlabeled_with_mae:
        return None

    y_weak_soft = np.array(mae_labels_for_unlabeled)
    cache_dir = Path(args.cache_dir)

    unlabeled_loader = DataLoader(
        Subset(train_dataset, unlabeled_with_mae),
        batch_size=16, shuffle=False, num_workers=16, collate_fn=collate_fn, pin_memory=True
    )
    X_weak, _ = extract_features(
        model, processor, unlabeled_loader, device,
        use_multi_gpu=args.use_multi_gpu, args=args,
        dataset=train_dataset, indices=unlabeled_with_mae,
        cache_path=get_features_cache_path(cache_dir, "unlabeled_mae", 2, 42, args.model_name)
    )

    classifier = train_pytorch_classifier(
        X_labeled, y_labeled, X_val, y_val,
        X_weak=X_weak, y_weak_soft=y_weak_soft,
        num_classes=101, lr=0.01, epochs=500, batch_size=256, patience=20, device=device, labeled_weight=5.0
    )

    classifier.eval()
    with torch.no_grad():
        test_preds = classifier.predict(torch.FloatTensor(X_test).to(device)).cpu().numpy()

    return accuracy_score(y_test, test_preds)


def run_experiment_3_xclip_self_training(args, train_dataset, test_dataset, splits, X_labeled, y_labeled, X_val, y_val, X_test, y_test, model, processor, device):
    classifier = train_pytorch_classifier(
        X_labeled, y_labeled, X_val, y_val,
        X_weak=None, y_weak_soft=None,
        num_classes=101, lr=0.01, epochs=500, batch_size=256, patience=20, device=device
    )

    cache_dir = Path(args.cache_dir)
    unlabeled_loader = DataLoader(
        Subset(train_dataset, splits['unlabeled']),
        batch_size=16, shuffle=False, num_workers=16, collate_fn=collate_fn, pin_memory=True
    )
    X_unlabeled, _ = extract_features(
        model, processor, unlabeled_loader, device,
        use_multi_gpu=args.use_multi_gpu, args=args,
        dataset=train_dataset, indices=splits['unlabeled'],
        cache_path=get_features_cache_path(cache_dir, "unlabeled_xclip", 2, 42, args.model_name)
    )

    classifier.eval()
    with torch.no_grad():
        logits = classifier.linear(torch.FloatTensor(X_unlabeled).to(device))
        pseudo_soft_labels = F.softmax(logits, dim=1).cpu().numpy()

    confidence = np.max(pseudo_soft_labels, axis=1)
    high_conf_mask = confidence >= 0.5
    X_weak = X_unlabeled[high_conf_mask]
    y_weak_soft = pseudo_soft_labels[high_conf_mask]

    classifier2 = train_pytorch_classifier(
        X_labeled, y_labeled, X_val, y_val,
        X_weak=X_weak, y_weak_soft=y_weak_soft,
        num_classes=101, lr=0.01, epochs=500, batch_size=256, patience=20, device=device, labeled_weight=5.0
    )

    classifier2.eval()
    with torch.no_grad():
        test_preds = classifier2.predict(torch.FloatTensor(X_test).to(device)).cpu().numpy()

    return accuracy_score(y_test, test_preds)


def run_experiment_4_oracle_distillation(args, train_dataset, test_dataset, splits, X_labeled, y_labeled, X_val, y_val, X_test, y_test, model, processor, device):
    cache_dir = Path(args.cache_dir)
    full_train_indices = list(range(len(train_dataset)))

    full_loader = DataLoader(
        Subset(train_dataset, full_train_indices),
        batch_size=16, shuffle=False, num_workers=16, collate_fn=collate_fn, pin_memory=True
    )
    X_full, y_full = extract_features(
        model, processor, full_loader, device,
        use_multi_gpu=args.use_multi_gpu, args=args,
        dataset=train_dataset, indices=full_train_indices,
        cache_path=get_features_cache_path(cache_dir, "full_train", "all", 42, args.model_name)
    )

    teacher = train_pytorch_classifier(
        X_full, y_full, X_val, y_val,
        X_weak=None, y_weak_soft=None,
        num_classes=101, lr=0.01, epochs=500, batch_size=256, patience=20, device=device
    )

    unlabeled_loader = DataLoader(
        Subset(train_dataset, splits['unlabeled']),
        batch_size=16, shuffle=False, num_workers=16, collate_fn=collate_fn, pin_memory=True
    )
    X_unlabeled, _ = extract_features(
        model, processor, unlabeled_loader, device,
        use_multi_gpu=args.use_multi_gpu, args=args,
        dataset=train_dataset, indices=splits['unlabeled'],
        cache_path=get_features_cache_path(cache_dir, "unlabeled_xclip", 2, 42, args.model_name)
    )

    teacher.eval()
    with torch.no_grad():
        logits = teacher.linear(torch.FloatTensor(X_unlabeled).to(device))
        y_weak_soft = F.softmax(logits, dim=1).cpu().numpy()

    student = train_pytorch_classifier(
        X_labeled, y_labeled, X_val, y_val,
        X_weak=X_unlabeled, y_weak_soft=y_weak_soft,
        num_classes=101, lr=0.01, epochs=500, batch_size=256, patience=20, device=device, labeled_weight=5.0
    )

    student.eval()
    with torch.no_grad():
        test_preds = student.predict(torch.FloatTensor(X_test).to(device)).cpu().numpy()

    return accuracy_score(y_test, test_preds)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    train_dataset = UCF101Dataset(split='train', processor=processor, num_frames=32, cache_dir=args.cache_dir)
    test_dataset = UCF101Dataset(split='test', processor=processor, num_frames=32, cache_dir=args.cache_dir)

    results = {}

    exp1_acc, splits, X_labeled, y_labeled, X_val, y_val, X_test, y_test = run_experiment_1_baseline(
        args, train_dataset, test_dataset, model, processor, device
    )
    results['exp1_baseline'] = exp1_acc

    results['exp2_mil_weak'] = run_experiment_2_mil_weak(
        args, train_dataset, test_dataset, splits,
        X_labeled, y_labeled, X_val, y_val, X_test, y_test,
        model, processor, device
    )

    results['exp3_xclip_self'] = run_experiment_3_xclip_self_training(
        args, train_dataset, test_dataset, splits,
        X_labeled, y_labeled, X_val, y_val, X_test, y_test,
        model, processor, device
    )

    results['exp4_oracle_distillation'] = run_experiment_4_oracle_distillation(
        args, train_dataset, test_dataset, splits,
        X_labeled, y_labeled, X_val, y_val, X_test, y_test,
        model, processor, device
    )

    with open("experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    args = SimpleNamespace(
        model_name="microsoft/xclip-base-patch16",
        cache_dir="/mnt/amlfs-03/shared/ayaanm/yanav/cache",
        use_multi_gpu=True,
        num_frames=32,
        input_size=224,
        scale_resize=256,
        use_train_augmentation=False,
        batch_size=16,
        num_workers=16
    )
    main(args)
