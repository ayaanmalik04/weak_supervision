#!/usr/bin/env python3
"""
Baseline 1: Few-Shot Linear Probe
UCF-101 → Few-shot splits → X-CLIP features → Logistic Regression → Evaluate
"""

import os
import sys
import json
import pickle
import argparse
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoProcessor, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset
from PIL import Image, ImageEnhance
from tqdm import tqdm
import random


def preprocess_frames_train(frames, input_size=224, scale_resize=256, 
                            flip_ratio=0.5, apply_augmentation=True):
    """
    Preprocess frames following X-CLIP training pipeline:
    1. Resize shorter edge to scale_resize (256)
    2. Random/Multi-scale crop (simplified to random crop from resized image)
    3. Random horizontal flip
    4. Color jitter and grayscale augmentation
    5. Resize to input_size (224x224)
    
    Args:
        frames: List of PIL images
        input_size: Target size (224)
        scale_resize: Size to resize shorter edge to (256)
        flip_ratio: Probability of horizontal flip (0.5)
        apply_augmentation: Whether to apply augmentations
    
    Returns:
        List of preprocessed PIL images
    """
    processed_frames = []
    
    # Determine augmentation parameters (same for all frames in video)
    do_flip = random.random() < flip_ratio if apply_augmentation else False
    
    for frame in frames:
        # Resize shorter edge to scale_resize
        w, h = frame.size
        if h < w:
            new_h, new_w = scale_resize, int(scale_resize * w / h)
        else:
            new_h, new_w = int(scale_resize * h / w), scale_resize
        frame = frame.resize((new_w, new_h), Image.BILINEAR)
        
        # Random crop (simplified version of MultiScaleCrop)
        # In practice, we do a random crop from the resized image
        w, h = frame.size
        if apply_augmentation and w > input_size and h > input_size:
            left = random.randint(0, w - input_size)
            top = random.randint(0, h - input_size)
            frame = frame.crop((left, top, left + input_size, top + input_size))
        else:
            # Center crop if not applying augmentation or image too small
            left = (w - input_size) // 2
            top = (h - input_size) // 2
            frame = frame.crop((left, top, left + input_size, top + input_size))
        
        # Resize to exact input_size (in case crop size doesn't match)
        if frame.size != (input_size, input_size):
            frame = frame.resize((input_size, input_size), Image.BILINEAR)
        
        # Random horizontal flip
        if do_flip:
            frame = frame.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Color jitter (brightness, contrast, saturation) - simplified
        if apply_augmentation and random.random() < 0.8:  # 80% chance
            # Brightness
            if random.random() < 0.5:
                enhancer = ImageEnhance.Brightness(frame)
                frame = enhancer.enhance(random.uniform(0.6, 1.4))
            # Contrast
            if random.random() < 0.5:
                enhancer = ImageEnhance.Contrast(frame)
                frame = enhancer.enhance(random.uniform(0.6, 1.4))
            # Saturation
            if random.random() < 0.5:
                enhancer = ImageEnhance.Color(frame)
                frame = enhancer.enhance(random.uniform(0.6, 1.4))
        
        # Random grayscale - simplified
        if apply_augmentation and random.random() < 0.2:  # 20% chance
            frame = frame.convert('L').convert('RGB')
        
        processed_frames.append(frame)
    
    return processed_frames


def preprocess_frames_val(frames, input_size=224, scale_resize=256):
    """
    Preprocess frames following X-CLIP validation pipeline:
    1. Resize shorter edge to scale_resize (256)
    2. Center crop to input_size (224x224)
    3. Return PIL images (normalization handled by processor)
    
    Args:
        frames: List of PIL images
        input_size: Target size for center crop (224)
        scale_resize: Size to resize shorter edge to (256)
    
    Returns:
        List of preprocessed PIL images
    """
    processed_frames = []
    
    for frame in frames:
        # Resize shorter edge to scale_resize
        w, h = frame.size
        if h < w:
            new_h, new_w = scale_resize, int(scale_resize * w / h)
        else:
            new_h, new_w = int(scale_resize * h / w), scale_resize
        frame = frame.resize((new_w, new_h), Image.BILINEAR)
        
        # Center crop to input_size x input_size
        w, h = frame.size
        left = (w - input_size) // 2
        top = (h - input_size) // 2
        right = left + input_size
        bottom = top + input_size
        frame = frame.crop((left, top, right, bottom))
        
        processed_frames.append(frame)
    
    return processed_frames


class UCF101Dataset(Dataset):
    """Dataset class for UCF-101 from HuggingFace (frame-by-frame format) with caching."""
    
    def __init__(self, split='train', processor=None, num_frames=32, input_size=224, 
                 scale_resize=256, use_train_augmentation=False, cache_dir=None, 
                 train_subset=1.0, seed=42):
        """
        Args:
            split: 'train' or 'test'
            processor: X-CLIP processor
            num_frames: Number of frames to sample from each video (default: 32)
            input_size: Target size for center crop (default: 224)
            scale_resize: Size to resize shorter edge to (default: 256)
            use_train_augmentation: Whether to use training augmentation (default: False)
            cache_dir: Directory to cache grouped dataset (default: None, uses current dir)
            train_subset: Fraction of training data to use (default: 1.0 = 100%)
            seed: Random seed for subset sampling
        """
        self.processor = processor
        self.num_frames = num_frames
        self.split = split
        self.input_size = input_size
        self.scale_resize = scale_resize
        self.use_train_augmentation = use_train_augmentation
        self.train_subset = train_subset
        
        # Setup cache
        if cache_dir is None:
            cache_dir = "."
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"ucf101_grouped_{split}.pkl"
        
        print(f"Loading UCF-101 dataset (split: {split})...")
        
        # Load dataset from HuggingFace (always - it's already cached by HF)
        print(f"Loading from HuggingFace (HF caches automatically)...")
        dataset = load_dataset("flwrlabs/ucf101", split=split)
        total_frames = len(dataset)
        
        # Try to load metadata cache (clip structure, not images)
        metadata_cache = cache_dir / f"ucf101_metadata_{split}.pkl"
        
        if metadata_cache.exists():
            print(f"Loading clip metadata from cache: {metadata_cache}")
            with open(metadata_cache, 'rb') as f:
                cache_data = pickle.load(f)
                clip_to_indices = cache_data['clip_to_indices']
                clip_labels = cache_data['clip_labels']
            print(f"✓ Loaded metadata cache!")
        else:
            print(f"Grouping frames by video (fast metadata-only scan)...")
            # Build mapping: clip_id -> list of dataset indices
            # Use select() to load ONLY metadata columns (not images!)
            clip_to_indices = defaultdict(list)
            clip_labels = {}
            clip_frame_nums = defaultdict(list)
            
            # Select only metadata columns (skip 'image' for huge speedup!)
            print("Loading metadata columns only (clip_id, label, frame)...")
            metadata_dataset = dataset.select_columns(['clip_id', 'label', 'frame'])
            
            batch_size = 50000  # Much larger batch now that we're not loading images
            num_batches = (total_frames + batch_size - 1) // batch_size
            
            start_idx = 0
            for batch_idx in tqdm(range(num_batches), desc=f"Grouping {split} frames"):
                end_idx = min(start_idx + batch_size, total_frames)
                
                # Get batch of metadata only (no images = super fast!)
                batch = metadata_dataset[start_idx:end_idx]
                
                # Process batch
                for i, (clip_id, label, frame) in enumerate(zip(
                    batch['clip_id'], 
                    batch['label'], 
                    batch['frame']
                )):
                    idx = start_idx + i
                    clip_to_indices[clip_id].append(idx)
                    clip_frame_nums[clip_id].append(frame)
                    if clip_id not in clip_labels:
                        clip_labels[clip_id] = label
                
                start_idx = end_idx
            
            # Sort indices by frame number for each clip
            print("Sorting frames within clips...")
            for clip_id in clip_to_indices:
                # Sort indices by frame number
                sorted_pairs = sorted(zip(clip_frame_nums[clip_id], clip_to_indices[clip_id]))
                clip_to_indices[clip_id] = [idx for _, idx in sorted_pairs]
            
            # Save metadata cache (small - just indices and labels)
            print(f"Saving metadata cache: {metadata_cache}")
            cache_data = {
                'clip_to_indices': dict(clip_to_indices),
                'clip_labels': clip_labels
            }
            with open(metadata_cache, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"✓ Metadata cache saved! (size: {metadata_cache.stat().st_size / 1024 / 1024:.1f} MB)")
        
        # Store dataset and clip structure
        self.dataset = dataset
        self.clip_to_indices = clip_to_indices
        self.clip_labels = clip_labels
        
        # Create index mapping
        self.clip_ids = sorted(list(self.clip_to_indices.keys()))
        self.labels = [self.clip_labels[clip_id] for clip_id in self.clip_ids]
        
        # Apply subset sampling for training data
        if split == 'train' and train_subset < 1.0:
            np.random.seed(seed)
            n_videos = len(self.clip_ids)
            n_subset = max(1, int(n_videos * train_subset))
            
            print(f"\n📊 Sampling {train_subset*100:.0f}% subset of training data...")
            print(f"   Full dataset: {n_videos} videos")
            print(f"   Subset: {n_subset} videos")
            
            # Random sample
            subset_indices = np.random.choice(n_videos, n_subset, replace=False)
            subset_indices = sorted(subset_indices)
            
            self.clip_ids = [self.clip_ids[i] for i in subset_indices]
            self.labels = [self.labels[i] for i in subset_indices]
            
            print(f"✓ Using {len(self.clip_ids)} videos for training")
        
        # Get label names (UCF-101 has 101 classes)
        self.num_classes = len(set(self.labels))
        
        print(f"\n✓ Loaded {len(self.clip_ids)} videos from {self.num_classes} classes")
        print(f"  Total frames in {split}: {total_frames}")
    
    def __len__(self):
        return len(self.clip_ids)
    
    def __getitem__(self, idx):
        clip_id = self.clip_ids[idx]
        frame_indices = self.clip_to_indices[clip_id]
        label = self.labels[idx]
        
        # Sample frames uniformly
        frames = self.sample_frames(frame_indices, self.num_frames)
        
        if frames is None or len(frames) == 0:
            # Return dummy data if sampling fails
            return None, label, clip_id
        
        return frames, label, clip_id
    
    def sample_frames(self, frame_indices, num_frames):
        """Sample num_frames uniformly from frame_indices and preprocess.
        
        Args:
            frame_indices: List of dataset indices for this clip's frames (already sorted)
            num_frames: Number of frames to sample
            
        Returns:
            List of preprocessed PIL images
        """
        try:
            total_frames = len(frame_indices)
            
            if total_frames == 0:
                return None
            
            # Sample frame indices uniformly (same as X-CLIP)
            if total_frames >= num_frames:
                sample_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            else:
                # If not enough frames, repeat the last frame
                sample_indices = list(range(total_frames))
                while len(sample_indices) < num_frames:
                    sample_indices.append(total_frames - 1)
            
            # Load sampled frames from dataset (PIL images)
            sampled_frames = [self.dataset[frame_indices[i]]['image'] for i in sample_indices]
            
            # Apply preprocessing based on mode
            if self.use_train_augmentation:
                preprocessed_frames = preprocess_frames_train(
                    sampled_frames, 
                    input_size=self.input_size, 
                    scale_resize=self.scale_resize,
                    apply_augmentation=True
                )
            else:
                preprocessed_frames = preprocess_frames_val(
                    sampled_frames, 
                    input_size=self.input_size, 
                    scale_resize=self.scale_resize
                )
            
            return preprocessed_frames
        
        except Exception as e:
            print(f"Error sampling frames: {e}")
            return None


def collate_fn(batch):
    """Custom collate function."""
    frames_list, labels, paths = zip(*batch)
    labels = torch.tensor(labels)
    return frames_list, labels, paths


def create_fewshot_split(train_dataset, test_dataset, k_shot, val_ratio=0.15, seed=42):
    """
    Create few-shot splits from train and test datasets:
    - labeled: k_shot examples per class from train
    - unlabeled: remaining training videos (unused in baseline 1)
    - val: validation set from train
    - test: use the official test set
    
    Returns dict with indices for each split and dataset references
    """
    np.random.seed(seed)
    
    labels = np.array(train_dataset.labels)
    
    # Group indices by class
    class_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_to_indices[label].append(idx)
    
    # Shuffle each class
    for label in class_to_indices:
        np.random.shuffle(class_to_indices[label])
    
    splits = {
        'labeled': [],
        'unlabeled': [],
        'val': [],
        'test': list(range(len(test_dataset)))  # Use all test data
    }
    
    # For each class in train, split into: val | labeled(k) | unlabeled
    for label, indices in class_to_indices.items():
        n = len(indices)
        
        # Calculate split points
        n_val = max(1, int(n * val_ratio))
        n_labeled = min(k_shot, max(1, n - n_val - 1))
        
        # Split indices
        val_idx = indices[:n_val]
        labeled_idx = indices[n_val:n_val + n_labeled]
        unlabeled_idx = indices[n_val + n_labeled:]
        
        splits['val'].extend(val_idx)
        splits['labeled'].extend(labeled_idx)
        splits['unlabeled'].extend(unlabeled_idx)
    
    return splits


def extract_features(model, processor, dataloader, device, use_multi_gpu=False, args=None, dataset=None, indices=None):
    """
    Extract X-CLIP features from videos (PIL image frames).
    
    Args:
        use_multi_gpu: If True, use parallel extraction across all GPUs (much faster!)
        args: Arguments (required if use_multi_gpu=True)
        dataset: Dataset object (required if use_multi_gpu=True)
        indices: Dataset indices (required if use_multi_gpu=True)
    """
    if use_multi_gpu and args is not None and indices is not None:
        # Use multi-GPU parallel extraction
        import torch.multiprocessing as mp
        from extract_features_multigpu import extract_features_parallel
        
        print("\n🚀 Using MULTI-GPU parallel extraction (8 GPUs)")
        features, labels = extract_features_parallel(
            dataset_split=dataset.split,
            all_indices=indices,
            args=args,
            num_gpus=torch.cuda.device_count()
        )
        return features, labels
    
    # Original single-process extraction (with DataParallel)
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for frames_list, labels, clip_ids in tqdm(dataloader, desc="Extracting features"):
            batch_features = []
            
            for frames in frames_list:
                if frames is None:
                    # Use zero features for failed videos
                    batch_features.append(torch.zeros(512))
                    continue
                
                try:
                    # Process frames with X-CLIP processor
                    # Frames are PIL images, processor handles them directly
                    inputs = processor(images=frames, return_tensors="pt")
                    
                    # Debug: check what keys are available
                    if 'pixel_values' not in inputs:
                        print(f"Available keys: {inputs.keys()}")
                        batch_features.append(torch.zeros(512))
                        continue
                    
                    # Extract video features
                    # The processor returns 'pixel_values' with shape (num_frames, C, H, W)
                    pixel_values = inputs['pixel_values'].to(device)
                    
                    # Add batch dimension if needed: (1, num_frames, C, H, W)
                    if pixel_values.dim() == 4:
                        # Shape is (num_frames, C, H, W), need to add batch dimension
                        pixel_values = pixel_values.unsqueeze(0)
                    
                    # Now reshape for vision model: (batch_size * num_frames, C, H, W)
                    batch_size, num_frames, num_channels, height, width = pixel_values.shape
                    pixel_values = pixel_values.reshape(-1, num_channels, height, width)
                    
                    # Get vision features using projection (for zero-shot compatibility)
                    # Handle DataParallel - access vision_model through module if wrapped
                    if isinstance(model, torch.nn.DataParallel):
                        # Get projected image features
                        vision_outputs = model.module.vision_model(pixel_values)
                        image_embeds = vision_outputs.last_hidden_state
                        image_embeds = model.module.visual_projection(image_embeds)
                    else:
                        vision_outputs = model.vision_model(pixel_values)
                        image_embeds = vision_outputs.last_hidden_state  
                        image_embeds = model.visual_projection(image_embeds)
                    
                    # Pool the features across frames to get single video representation
                    # Use the CLS token (first token) pooling
                    features = image_embeds[:, 0, :].mean(dim=0).cpu()
                    
                    # Normalize features
                    features = F.normalize(features, dim=-1)
                    batch_features.append(features)
                
                except Exception as e:
                    print(f"Error processing video: {e}")
                    import traceback
                    traceback.print_exc()
                    batch_features.append(torch.zeros(512))
            
            # Stack features
            batch_features = torch.stack(batch_features)
            all_features.append(batch_features)
            all_labels.append(labels)
    
    # Concatenate all
    all_features = torch.cat(all_features, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    return all_features, all_labels


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, 
                       class_names, C=1.0, max_iter=1000, seed=42, patience=10, min_delta=0.001):
    """Train logistic regression with early stopping based on validation plateau.
    
    Args:
        patience: Number of iterations to wait for improvement before stopping (default: 10)
        min_delta: Minimum change in validation accuracy to be considered improvement (default: 0.001)
    """
    
    num_classes = len(np.unique(y_train))
    print(f"\nTraining on {len(y_train)} labeled examples...")
    print(f"Number of classes: {num_classes}")
    print(f"Feature dimension: {X_train.shape[1]}")
    print(f"Early stopping: patience={patience}, min_delta={min_delta}")
    
    # Train iteratively with early stopping
    best_val_acc = 0.0
    best_clf = None
    patience_counter = 0
    
    # Use warm_start to train iteratively
    clf = LogisticRegression(
        C=C,
        max_iter=1,  # Train one iteration at a time
        random_state=seed,
        multi_class='multinomial',
        solver='lbfgs',
        n_jobs=-1,
        verbose=0,
        warm_start=True  # Keep weights from previous iteration
    )
    
    print("\nTraining with validation-based early stopping:")
    for iteration in range(1, max_iter + 1):
        # Train one more iteration
        clf.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_pred = clf.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        
        # Check for improvement
        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            # Deep copy the best model
            import copy
            best_clf = copy.deepcopy(clf)
            patience_counter = 0
            improvement = "✓"
        else:
            patience_counter += 1
            improvement = " "
        
        # Print progress every 10 iterations or on improvement
        if iteration % 10 == 0 or improvement == "✓":
            print(f"  Iter {iteration:4d}: Val Acc = {val_acc:.4f} (Best: {best_val_acc:.4f}) {improvement}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n✓ Early stopping triggered after {iteration} iterations")
            print(f"  No improvement for {patience} consecutive iterations")
            break
    else:
        print(f"\n✓ Training completed: reached max_iter={max_iter}")
    
    # Use best model
    clf = best_clf if best_clf is not None else clf
    
    # Confirm output shape
    print(f"\nLogistic regression weight matrix shape: {clf.coef_.shape}")  # Should be (101, 512)
    
    # Final evaluation on validation and test sets
    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)
    
    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred, average='macro')
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average='macro')
    
    results = {
        'val_accuracy': float(val_acc),
        'val_macro_f1': float(val_f1),
        'test_accuracy': float(test_acc),
        'test_macro_f1': float(test_f1),
        'num_train': len(y_train),
        'num_val': len(y_val),
        'num_test': len(y_test),
        'stopped_at_iter': iteration
    }
    
    print(f"\nFinal Results:")
    print(f"Val  Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%), Macro-F1: {val_f1:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%), Macro-F1: {test_f1:.4f}")
    
    return clf, results


def main(args):
    # Set device and enable multi-GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    
    if torch.cuda.is_available():
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
        print(f"Available GPUs: {num_gpus}")
        if num_gpus > 1:
            print(f"Using DataParallel across {num_gpus} GPUs for faster feature extraction")
    else:
        print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("BASELINE 1: Few-Shot Linear Probe")
    print("=" * 80)
    
    # Load X-CLIP model
    print("\n[1/5] Loading X-CLIP model...")
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    
    # Wrap model with DataParallel if multiple GPUs available
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    
    model = model.to(device)
    print(f"Loaded: {args.model_name}")
    
    # Load UCF-101 datasets from HuggingFace
    print("\n[2/5] Loading UCF-101 dataset from HuggingFace...")
    print(f"Using {args.num_frames} frames per video, {args.input_size}x{args.input_size} resolution")
    print(f"Preprocessing: Resize shorter edge to {args.scale_resize}, then center crop to {args.input_size}")
    
    if args.use_train_augmentation:
        print("Training set: Using training augmentation (random crop, flip, color jitter)")
        print("Val/Test sets: Using validation preprocessing (center crop, no augmentation)")
    else:
        print("All sets: Using validation preprocessing (center crop, no augmentation)")
    
    train_dataset = UCF101Dataset(
        split='train',
        processor=processor,
        num_frames=args.num_frames,
        input_size=args.input_size,
        scale_resize=args.scale_resize,
        use_train_augmentation=args.use_train_augmentation,
        cache_dir=args.cache_dir,
        train_subset=args.train_subset,
        seed=42  # Use fixed seed for reproducible subsets
    )
    
    test_dataset = UCF101Dataset(
        split='test',
        processor=processor,
        num_frames=args.num_frames,
        input_size=args.input_size,
        scale_resize=args.scale_resize,
        use_train_augmentation=False,  # Always use val preprocessing for test
        cache_dir=args.cache_dir,
        train_subset=1.0,  # Always use full test set
        seed=42
    )
    
    # Results across all k-shots and seeds
    all_results = {
        'k_shots': args.k_shots,
        'seeds': args.seeds,
        'experiments': []
    }
    
    # Total experiments
    total_experiments = len(args.k_shots) * len(args.seeds)
    experiment_counter = 0
    
    # Run experiments for each k-shot value
    for k_shot in tqdm(args.k_shots, desc="K-shot values", position=0):
        k_results = {
            'k': k_shot,
            'seeds': []
        }
        
        print(f"\n{'=' * 80}")
        print(f"K-SHOT = {k_shot}")
        print(f"{'=' * 80}")
        
        for seed in tqdm(args.seeds, desc=f"  Seeds (K={k_shot})", position=1, leave=False):
            experiment_counter += 1
            exp_start_time = time.time()
            print(f"\n--- Experiment {experiment_counter}/{total_experiments}: K={k_shot}, Seed={seed} ---")
            
            # Create few-shot split
            print("[3/5] Creating few-shot splits...")
            splits = create_fewshot_split(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                k_shot=k_shot,
                val_ratio=args.val_ratio,
                seed=seed
            )
            
            print(f"Labeled:   {len(splits['labeled'])} videos ({k_shot} per class)")
            print(f"Unlabeled: {len(splits['unlabeled'])} videos (not used in baseline 1)")
            print(f"Val:       {len(splits['val'])} videos")
            print(f"Test:      {len(splits['test'])} videos")
            
            # Save split info
            split_info = {
                'labeled': [train_dataset.clip_ids[i] for i in splits['labeled']],
                'unlabeled': [train_dataset.clip_ids[i] for i in splits['unlabeled']],
                'val': [train_dataset.clip_ids[i] for i in splits['val']],
                'test': [test_dataset.clip_ids[i] for i in splits['test']]
            }
            split_path = output_dir / f"split_k{k_shot}_seed{seed}.json"
            with open(split_path, 'w') as f:
                json.dump(split_info, f, indent=2)
            
            # Create dataloaders for each split
            print("\n[4/5] Extracting X-CLIP features...")
            
            labeled_loader = DataLoader(
                Subset(train_dataset, splits['labeled']),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_fn,
                pin_memory=True,  # Faster GPU transfer
                prefetch_factor=2  # Prefetch batches
            )
            
            val_loader = DataLoader(
                Subset(train_dataset, splits['val']),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                prefetch_factor=2
            )
            
            test_loader = DataLoader(
                Subset(test_dataset, splits['test']),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                prefetch_factor=2
            )
            
            # Extract features
            print(f"Extracting labeled set features ({len(splits['labeled'])} videos)...")
            X_labeled, y_labeled = extract_features(
                model, processor, labeled_loader, device,
                use_multi_gpu=args.use_multi_gpu_extraction,
                args=args,
                dataset=train_dataset,
                indices=splits['labeled']
            )
            
            print(f"Extracting val set features ({len(splits['val'])} videos)...")
            X_val, y_val = extract_features(
                model, processor, val_loader, device,
                use_multi_gpu=args.use_multi_gpu_extraction,
                args=args,
                dataset=train_dataset,
                indices=splits['val']
            )
            
            print(f"Extracting test set features ({len(splits['test'])} videos)...")
            X_test, y_test = extract_features(
                model, processor, test_loader, device,
                use_multi_gpu=args.use_multi_gpu_extraction,
                args=args,
                dataset=test_dataset,
                indices=splits['test']
            )
            
            # Train and evaluate
            print("\n[5/5] Training logistic regression...")
            # Get class names (just use label indices as UCF-101 has 101 classes)
            num_classes = train_dataset.num_classes
            class_names = [f"class_{i}" for i in range(num_classes)]
            
            clf, results = train_and_evaluate(
                X_labeled, y_labeled,
                X_val, y_val,
                X_test, y_test,
                class_names,
                C=args.C,
                max_iter=args.max_iter,
                seed=seed,
                patience=args.patience,
                min_delta=args.min_delta
            )
            
            # Save model
            model_path = output_dir / f"model_k{k_shot}_seed{seed}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'classifier': clf,
                    'num_classes': num_classes
                }, f)
            
            results['seed'] = seed
            k_results['seeds'].append(results)
            
            exp_duration = time.time() - exp_start_time
            print(f"✓ Experiment completed in {exp_duration/60:.1f} minutes")
        
        # Compute statistics across seeds
        val_accs = [s['val_accuracy'] for s in k_results['seeds']]
        test_accs = [s['test_accuracy'] for s in k_results['seeds']]
        val_f1s = [s['val_macro_f1'] for s in k_results['seeds']]
        test_f1s = [s['test_macro_f1'] for s in k_results['seeds']]
        
        k_results['summary'] = {
            'val_accuracy_mean': float(np.mean(val_accs)),
            'val_accuracy_std': float(np.std(val_accs)),
            'test_accuracy_mean': float(np.mean(test_accs)),
            'test_accuracy_std': float(np.std(test_accs)),
            'val_f1_mean': float(np.mean(val_f1s)),
            'val_f1_std': float(np.std(val_f1s)),
            'test_f1_mean': float(np.mean(test_f1s)),
            'test_f1_std': float(np.std(test_f1s))
        }
        
        print(f"\n{'=' * 80}")
        print(f"SUMMARY for K={k_shot} (across {len(args.seeds)} seeds)")
        print(f"{'=' * 80}")
        print(f"Val  Accuracy: {k_results['summary']['val_accuracy_mean']:.4f} ± {k_results['summary']['val_accuracy_std']:.4f}")
        print(f"Test Accuracy: {k_results['summary']['test_accuracy_mean']:.4f} ± {k_results['summary']['test_accuracy_std']:.4f}")
        print(f"Val  Macro-F1: {k_results['summary']['val_f1_mean']:.4f} ± {k_results['summary']['val_f1_std']:.4f}")
        print(f"Test Macro-F1: {k_results['summary']['test_f1_mean']:.4f} ± {k_results['summary']['test_f1_std']:.4f}")
        
        all_results['experiments'].append(k_results)
    
    # Save all results
    results_path = output_dir / "baseline1_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'=' * 80}")
    print("DONE! Results saved to:")
    print(f"  {results_path}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline 1: Few-Shot Linear Probe")
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/amlfs-03/shared/ayaanm/yanav/results/baseline1",
        help="Output directory for results"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/mnt/amlfs-03/shared/ayaanm/yanav/cache",
        help="Directory to cache grouped UCF-101 dataset"
    )
    parser.add_argument(
        "--train_subset",
        type=float,
        default=1.0,
        help="Fraction of training data to use (0.0-1.0). E.g., 0.2 = 20%% (default: 1.0 = 100%%)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/xclip-base-patch16",
        help="X-CLIP model name from HuggingFace"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=32,
        help="Number of frames to sample from each video (default: 32)"
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=224,
        help="Input resolution (width and height) after center crop (default: 224)"
    )
    parser.add_argument(
        "--scale_resize",
        type=int,
        default=256,
        help="Size to resize shorter edge before center crop (default: 256)"
    )
    parser.add_argument(
        "--use_train_augmentation",
        action='store_true',
        help="Use training augmentation for labeled training set (random crop, flip, color jitter)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for feature extraction"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers"
    )
    parser.add_argument(
        "--k_shots",
        type=int,
        nargs='+',
        default=[5, 10, 20],
        help="Number of labeled examples per class"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs='+',
        default=[42, 43, 44],
        help="Random seeds for multiple runs"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Ratio of validation set (from training data)"
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Regularization parameter for logistic regression"
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="Max iterations for logistic regression"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience: number of iterations without improvement (default: 10)"
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=0.001,
        help="Minimum validation accuracy improvement to be considered progress (default: 0.001)"
    )
    parser.add_argument(
        "--use_multi_gpu_extraction",
        action='store_true',
        help="Use parallel extraction across all GPUs (much faster!)"
    )
    
    args = parser.parse_args()
    main(args)
