#!/usr/bin/env python3
"""
Baseline 3: Full Supervised Linear Probe
UCF-101 → ALL training labels → X-CLIP features → Logistic Regression → Evaluate

Uses all available training data to establish upper bound performance.
"""

import os
import json
import pickle
import argparse
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import copy

# Import from baseline1
import sys
sys.path.append(str(Path(__file__).parent))
from baseline1_fewshot import UCF101Dataset, collate_fn, extract_features
from torch.utils.data import DataLoader


def train_and_evaluate_full(X_train, y_train, X_test, y_test, C=1.0, 
                            max_iter=1000, patience=10, min_delta=0.001, seed=42):
    """Train logistic regression on full training set with early stopping."""
    
    num_classes = len(np.unique(y_train))
    print(f"\nTraining on {len(y_train)} labeled examples (FULL DATASET)...")
    print(f"Number of classes: {num_classes}")
    print(f"Feature dimension: {X_train.shape[1]}")
    print(f"Early stopping: patience={patience}, min_delta={min_delta}")
    
    # Split train into train/val for early stopping
    # Use 15% of training data for validation
    n_val = int(0.15 * len(y_train))
    indices = np.random.RandomState(seed).permutation(len(y_train))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    X_train_subset = X_train[train_indices]
    y_train_subset = y_train[train_indices]
    X_val = X_train[val_indices]
    y_val = y_train[val_indices]
    
    print(f"Split: {len(train_indices)} train, {len(val_indices)} val for early stopping")
    
    # Train iteratively with early stopping
    best_val_acc = 0.0
    best_clf = None
    patience_counter = 0
    
    clf = LogisticRegression(
        C=C,
        max_iter=1,
        random_state=seed,
        multi_class='multinomial',
        solver='lbfgs',
        n_jobs=-1,
        verbose=0,
        warm_start=True
    )
    
    print("\nTraining with validation-based early stopping:")
    for iteration in range(1, max_iter + 1):
        clf.fit(X_train_subset, y_train_subset)
        
        val_pred = clf.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        
        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            best_clf = copy.deepcopy(clf)
            patience_counter = 0
            improvement = "✓"
        else:
            patience_counter += 1
            improvement = " "
        
        if iteration % 10 == 0 or improvement == "✓":
            print(f"  Iter {iteration:4d}: Val Acc = {val_acc:.4f} (Best: {best_val_acc:.4f}) {improvement}")
        
        if patience_counter >= patience:
            print(f"\n✓ Early stopping triggered after {iteration} iterations")
            print(f"  No improvement for {patience} consecutive iterations")
            break
    else:
        print(f"\n✓ Training completed: reached max_iter={max_iter}")
    
    clf = best_clf if best_clf is not None else clf
    
    print(f"\nLogistic regression weight matrix shape: {clf.coef_.shape}")
    
    # Final evaluation on test set
    test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average='macro')
    
    # Also compute train accuracy to check for overfitting
    train_pred = clf.predict(X_train_subset)
    train_acc = accuracy_score(y_train_subset, train_pred)
    
    results = {
        'train_accuracy': float(train_acc),
        'val_accuracy': float(best_val_acc),
        'test_accuracy': float(test_acc),
        'test_macro_f1': float(test_f1),
        'num_train': len(y_train_subset),
        'num_val': len(y_val),
        'num_test': len(y_test),
        'stopped_at_iter': iteration
    }
    
    print(f"\nFinal Results:")
    print(f"Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Val   Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"Test  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%), Macro-F1: {test_f1:.4f}")
    
    return clf, results


def main(args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    
    if torch.cuda.is_available():
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
        print(f"Available GPUs: {num_gpus}")
        if num_gpus > 1:
            print(f"Using DataParallel across {num_gpus} GPUs")
    else:
        print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("BASELINE 3: Full Supervised Linear Probe")
    print("=" * 80)
    
    # Load X-CLIP model
    print("\n[1/4] Loading X-CLIP model...")
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    
    model = model.to(device)
    print(f"Loaded: {args.model_name}")
    
    # Load datasets
    print("\n[2/4] Loading UCF-101 dataset...")
    print(f"Using {args.num_frames} frames per video, {args.input_size}x{args.input_size} resolution")
    
    if args.use_train_augmentation:
        print("Training set: Using training augmentation (random crop, flip, color jitter)")
    else:
        print("Training set: Using validation preprocessing (center crop)")
    
    train_dataset = UCF101Dataset(
        split='train',
        processor=processor,
        num_frames=args.num_frames,
        input_size=args.input_size,
        scale_resize=args.scale_resize,
        use_train_augmentation=args.use_train_augmentation,
        cache_dir=args.cache_dir,
        train_subset=args.train_subset,
        seed=42
    )
    
    test_dataset = UCF101Dataset(
        split='test',
        processor=processor,
        num_frames=args.num_frames,
        input_size=args.input_size,
        scale_resize=args.scale_resize,
        use_train_augmentation=False,
        cache_dir=args.cache_dir,
        train_subset=1.0,
        seed=42
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} videos")
    print(f"  Test:  {len(test_dataset)} videos")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2
    )
    
    # Extract features
    print("\n[3/4] Extracting X-CLIP features...")
    print(f"Extracting train features ({len(train_dataset)} videos)...")
    train_indices = list(range(len(train_dataset)))
    X_train, y_train = extract_features(
        model, processor, train_loader, device,
        use_multi_gpu=args.use_multi_gpu_extraction,
        args=args,
        dataset=train_dataset,
        indices=train_indices
    )
    
    print(f"Extracting test features ({len(test_dataset)} videos)...")
    test_indices = list(range(len(test_dataset)))
    X_test, y_test = extract_features(
        model, processor, test_loader, device,
        use_multi_gpu=args.use_multi_gpu_extraction,
        args=args,
        dataset=test_dataset,
        indices=test_indices
    )
    
    # Train and evaluate
    print("\n[4/4] Training logistic regression on full dataset...")
    clf, results = train_and_evaluate_full(
        X_train, y_train,
        X_test, y_test,
        C=args.C,
        max_iter=args.max_iter,
        patience=args.patience,
        min_delta=args.min_delta,
        seed=42
    )
    
    # Save model
    model_path = output_dir / "model_fullsupervised.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'classifier': clf,
            'num_classes': train_dataset.num_classes
        }, f)
    print(f"\n✓ Model saved to: {model_path}")
    
    # Save results
    results['config'] = {
        'model_name': args.model_name,
        'num_frames': args.num_frames,
        'train_subset': args.train_subset,
        'use_train_augmentation': args.use_train_augmentation,
        'C': args.C,
        'max_iter': args.max_iter,
        'patience': args.patience
    }
    
    results_path = output_dir / "baseline3_fullsupervised_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {results_path}")
    
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline 3: Full Supervised Linear Probe")
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/amlfs-03/shared/ayaanm/yanav/results/baseline3",
        help="Output directory for results"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/mnt/amlfs-03/shared/ayaanm/yanav/cache",
        help="Cache directory for grouped dataset"
    )
    parser.add_argument(
        "--train_subset",
        type=float,
        default=1.0,
        help="Fraction of training data to use (default: 1.0 = 100%%)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/xclip-base-patch16",
        help="X-CLIP model name"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=32,
        help="Number of frames per video"
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=224,
        help="Input resolution"
    )
    parser.add_argument(
        "--scale_resize",
        type=int,
        default=256,
        help="Resize shorter edge size"
    )
    parser.add_argument(
        "--use_train_augmentation",
        action='store_true',
        help="Use training augmentation"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers"
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Regularization parameter"
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="Max iterations"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=0.001,
        help="Minimum improvement for early stopping"
    )
    parser.add_argument(
        "--use_multi_gpu_extraction",
        action='store_true',
        help="Use parallel extraction across all GPUs (much faster!)"
    )
    
    args = parser.parse_args()
    main(args)

