#!/usr/bin/env python3
"""
Multi-GPU Feature Extraction (8 GPUs in parallel)
Each GPU processes its own chunk independently = maximum efficiency!
"""

import os
import sys
import argparse
import pickle
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from transformers import AutoProcessor, AutoModel
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.multiprocessing as mp

sys.path.append(str(Path(__file__).parent))
from baseline1_fewshot import UCF101Dataset, collate_fn


def extract_features_single_gpu(gpu_id, dataset_split, indices, args, output_queue):
    """
    Extract features on a single GPU.
    
    Args:
        gpu_id: GPU device ID (0-7)
        dataset_split: 'train' or 'test'
        indices: List of dataset indices to process on this GPU
        args: Arguments
        output_queue: Queue to return results
    """
    try:
        # Set device for this process
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)
        
        print(f"[GPU {gpu_id}] Processing {len(indices)} videos...")
        
        # Load model on this GPU (no DataParallel!)
        processor = AutoProcessor.from_pretrained(args.model_name)
        model = AutoModel.from_pretrained(args.model_name)
        model = model.to(device)
        model.eval()
        
        # Load dataset
        if dataset_split == 'train':
            dataset = UCF101Dataset(
                split='train',
                processor=processor,
                num_frames=args.num_frames,
                input_size=args.input_size,
                scale_resize=args.scale_resize,
                use_train_augmentation=args.use_train_augmentation,
                cache_dir=args.cache_dir,
                train_subset=1.0,
                seed=42
            )
        else:
            dataset = UCF101Dataset(
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
        
        # Create dataloader for this GPU's subset
        subset = Subset(dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers // 8,  # Split workers across GPUs
            collate_fn=collate_fn,
            pin_memory=True,
            prefetch_factor=2
        )
        
        # Extract features
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for frames_list, labels, clip_ids in tqdm(
                loader, 
                desc=f"GPU {gpu_id}", 
                position=gpu_id,
                leave=True
            ):
                batch_features = []
                
                for frames in frames_list:
                    if frames is None:
                        batch_features.append(torch.zeros(512).to(device))
                        continue
                    
                    try:
                        inputs = processor(images=frames, return_tensors="pt")
                        pixel_values = inputs['pixel_values'].to(device)
                        
                        if pixel_values.dim() == 4:
                            pixel_values = pixel_values.unsqueeze(0)
                        
                        batch_size, num_frames, num_channels, height, width = pixel_values.shape
                        pixel_values = pixel_values.reshape(-1, num_channels, height, width)
                        
                        # Direct model call (no DataParallel wrapper) with projection
                        vision_outputs = model.vision_model(pixel_values)
                        image_embeds = vision_outputs.last_hidden_state
                        image_embeds = model.visual_projection(image_embeds)
                        
                        # Use CLS token pooling across frames
                        features = image_embeds[:, 0, :].mean(dim=0)
                        features = F.normalize(features, dim=-1)
                        batch_features.append(features)
                    
                    except Exception as e:
                        print(f"[GPU {gpu_id}] Error: {e}")
                        batch_features.append(torch.zeros(512).to(device))
                
                batch_features = torch.stack(batch_features)
                all_features.append(batch_features.cpu())
                all_labels.append(labels.cpu())
        
        # Concatenate results
        features = torch.cat(all_features, dim=0).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()
        
        print(f"[GPU {gpu_id}] Done! Extracted {len(features)} features")
        
        # Return results via queue
        output_queue.put((gpu_id, features, labels, indices))
    
    except Exception as e:
        print(f"[GPU {gpu_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        output_queue.put((gpu_id, None, None, None))


def extract_features_parallel(dataset_split, all_indices, args, num_gpus=8):
    """
    Extract features in parallel across multiple GPUs.
    
    Args:
        dataset_split: 'train' or 'test'
        all_indices: All dataset indices to process
        args: Arguments
        num_gpus: Number of GPUs to use
    
    Returns:
        features, labels (sorted by original indices)
    """
    # Split indices across GPUs
    indices_per_gpu = np.array_split(all_indices, num_gpus)
    
    print(f"\n{'='*80}")
    print(f"Parallel Feature Extraction on {num_gpus} GPUs")
    print(f"{'='*80}")
    print(f"Total videos: {len(all_indices)}")
    for i, chunk in enumerate(indices_per_gpu):
        print(f"  GPU {i}: {len(chunk)} videos")
    print(f"{'='*80}\n")
    
    # Create output queue
    ctx = mp.get_context('spawn')
    output_queue = ctx.Queue()
    
    # Launch processes
    processes = []
    for gpu_id in range(num_gpus):
        if len(indices_per_gpu[gpu_id]) == 0:
            continue
        
        p = ctx.Process(
            target=extract_features_single_gpu,
            args=(gpu_id, dataset_split, indices_per_gpu[gpu_id].tolist(), args, output_queue)
        )
        p.start()
        processes.append(p)
    
    # Collect results
    results = {}
    for _ in range(len(processes)):
        gpu_id, features, labels, indices = output_queue.get()
        if features is not None:
            results[gpu_id] = (features, labels, indices)
    
    # Wait for all processes
    for p in processes:
        p.join()
    
    # Combine results in original order
    print("\nCombining results...")
    all_features = []
    all_labels = []
    all_gpu_indices = []
    
    for gpu_id in sorted(results.keys()):
        features, labels, indices = results[gpu_id]
        all_features.append(features)
        all_labels.append(labels)
        all_gpu_indices.extend(indices)
    
    combined_features = np.concatenate(all_features, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)
    
    # Reorder to match original indices
    reorder_indices = np.argsort(all_gpu_indices)
    combined_features = combined_features[reorder_indices]
    combined_labels = combined_labels[reorder_indices]
    
    print(f"✓ Extracted {len(combined_features)} features across {num_gpus} GPUs")
    
    return combined_features, combined_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True, choices=['train', 'test'])
    parser.add_argument("--indices_file", type=str, required=True, help="Pickle file with indices")
    parser.add_argument("--output_file", type=str, required=True, help="Output pickle file")
    parser.add_argument("--cache_dir", type=str, default="/mnt/amlfs-03/shared/ayaanm/yanav/cache")
    parser.add_argument("--model_name", type=str, default="microsoft/xclip-base-patch16")
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--scale_resize", type=int, default=256)
    parser.add_argument("--use_train_augmentation", action='store_true')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--num_gpus", type=int, default=8)
    
    args = parser.parse_args()
    
    # Load indices
    with open(args.indices_file, 'rb') as f:
        indices = pickle.load(f)
    
    print(f"Loaded {len(indices)} indices from {args.indices_file}")
    
    # Extract features in parallel
    features, labels = extract_features_parallel(
        args.split, 
        indices, 
        args, 
        num_gpus=args.num_gpus
    )
    
    # Save results
    output_data = {
        'features': features,
        'labels': labels,
        'indices': indices
    }
    
    with open(args.output_file, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"\n✓ Features saved to: {args.output_file}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()

