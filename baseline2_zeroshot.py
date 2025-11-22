#!/usr/bin/env python3
"""
Baseline 2: Zero-Shot X-CLIP Evaluation
UCF-101 → X-CLIP text+vision encoders → Zero-shot classification → Evaluate

No training required - uses contrastive similarity between video features and text prompts.
"""

import os
import json
import argparse
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm

# Import from baseline1
import sys
sys.path.append(str(Path(__file__).parent))
from baseline1_fewshot import UCF101Dataset, collate_fn, extract_features
from torch.utils.data import DataLoader, Subset


# UCF-101 class names (in alphabetical order as indexed)
UCF101_CLASSES = [
    "ApplyEyeMakeup", "ApplyLipstick", "Archery", "BabyCrawling", "BalanceBeam",
    "BandMarching", "BaseballPitch", "Basketball", "BasketballDunk", "BenchPress",
    "Biking", "Billiards", "BlowDryHair", "BlowingCandles", "BodyWeightSquats",
    "Bowling", "BoxingPunchingBag", "BoxingSpeedBag", "BreastStroke", "BrushingTeeth",
    "CleanAndJerk", "CliffDiving", "CricketBowling", "CricketShot", "CuttingInKitchen",
    "Diving", "Drumming", "Fencing", "FieldHockeyPenalty", "FloorGymnastics",
    "FrisbeeCatch", "FrontCrawl", "GolfSwing", "Haircut", "Hammering",
    "HammerThrow", "HandstandPushups", "HandstandWalking", "HeadMassage", "HighJump",
    "HorseRace", "HorseRiding", "HulaHoop", "IceDancing", "JavelinThrow",
    "JugglingBalls", "JumpingJack", "JumpRope", "Kayaking", "Knitting",
    "LongJump", "Lunges", "MilitaryParade", "Mixing", "MoppingFloor",
    "Nunchucks", "ParallelBars", "PizzaTossing", "PlayingCello", "PlayingDaf",
    "PlayingDhol", "PlayingFlute", "PlayingGuitar", "PlayingPiano", "PlayingSitar",
    "PlayingTabla", "PlayingViolin", "PoleVault", "PommelHorse", "PullUps",
    "Punch", "PushUps", "Rafting", "RockClimbingIndoor", "RopeClimbing",
    "Rowing", "SalsaSpin", "ShavingBeard", "Shotput", "SkateBoarding",
    "Skiing", "Skijet", "SkyDiving", "SoccerJuggling", "SoccerPenalty",
    "StillRings", "SumoWrestling", "Surfing", "Swing", "TableTennisShot",
    "TaiChi", "TennisSwing", "ThrowDiscus", "TrampolineJumping", "Typing",
    "UnevenBars", "VolleyballSpiking", "WalkingWithDog", "WallPushups", "WritingOnBoard",
    "YoYo"
]


def create_text_prompts(class_names, prompt_templates=None):
    """
    Create text prompts for each class using multiple templates.
    
    Args:
        class_names: List of class names
        prompt_templates: List of prompt templates (default: standard action templates)
    
    Returns:
        List of prompts for each class
    """
    if prompt_templates is None:
        # Default templates inspired by CLIP and action recognition literature
        prompt_templates = [
            "a video of a person {}",
            "a person {}",
            "someone {}",
            "a video showing {}",
            "{}"
        ]
    
    all_prompts = []
    for class_name in class_names:
        # Convert CamelCase to readable format
        readable_name = ''.join([' ' + c.lower() if c.isupper() else c for c in class_name]).strip()
        
        class_prompts = []
        for template in prompt_templates:
            prompt = template.format(readable_name)
            class_prompts.append(prompt)
        
        all_prompts.append(class_prompts)
    
    return all_prompts


def extract_video_features(model, processor, dataloader, device, use_multi_gpu=False, args=None, dataset=None, indices=None):
    """
    Extract video features using X-CLIP vision encoder.
    Wrapper around extract_features from baseline1 for compatibility.
    """
    # Use the shared extract_features function which has multi-GPU support
    features, labels = extract_features(
        model, processor, dataloader, device,
        use_multi_gpu=use_multi_gpu,
        args=args,
        dataset=dataset,
        indices=indices
    )
    
    # Convert to torch tensors (extract_features returns numpy arrays)
    return torch.from_numpy(features), torch.from_numpy(labels)


def extract_text_features(model, processor, prompts, device):
    """
    Extract text features for all class prompts.
    
    Args:
        model: X-CLIP model
        processor: X-CLIP processor
        prompts: List of prompt lists (one list per class)
        device: torch device
    
    Returns:
        Tensor of text features [num_classes, feature_dim]
    """
    model.eval()
    class_features = []
    
    print("Extracting text features for prompts...")
    with torch.no_grad():
        for class_prompts in tqdm(prompts, desc="Processing text prompts"):
            prompt_features = []
            
            for prompt in class_prompts:
                # Tokenize text
                inputs = processor(text=[prompt], return_tensors="pt", padding=True)
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                
                # Get text features using the full model (includes projection)
                if isinstance(model, torch.nn.DataParallel):
                    text_outputs = model.module.get_text_features(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                else:
                    text_outputs = model.get_text_features(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                
                # Normalize
                text_features = F.normalize(text_outputs, dim=-1)
                prompt_features.append(text_features)
            
            # Average across all prompts for this class
            class_feature = torch.stack(prompt_features).mean(dim=0)
            class_feature = F.normalize(class_feature, dim=-1)
            class_features.append(class_feature)
    
    # Stack all class features
    text_features = torch.cat(class_features, dim=0)  # [num_classes, feature_dim]
    
    return text_features


def zero_shot_classify(video_features, text_features, temperature=1.0):
    """
    Perform zero-shot classification using cosine similarity.
    
    Args:
        video_features: [N, feature_dim]
        text_features: [num_classes, feature_dim]
        temperature: Temperature for softmax (default: 1.0)
    
    Returns:
        predictions: [N] class indices
        probabilities: [N, num_classes] class probabilities
    """
    # Compute cosine similarity: [N, num_classes]
    similarity = video_features @ text_features.T
    
    # Apply temperature and softmax
    logits = similarity / temperature
    probabilities = F.softmax(logits, dim=-1)
    
    # Get predictions
    predictions = logits.argmax(dim=-1)
    
    return predictions, probabilities


def evaluate_zero_shot(video_features, text_features, labels, class_names):
    """Evaluate zero-shot classification."""
    # Ensure both are on the same device
    device = text_features.device
    video_features = video_features.to(device)
    labels = labels.to(device)
    
    predictions, probabilities = zero_shot_classify(video_features, text_features)
    
    # Move to CPU for sklearn
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    probabilities = probabilities.cpu().numpy()
    
    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average='macro')
    
    # Per-class accuracy
    conf_matrix = confusion_matrix(labels, predictions)
    per_class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    
    # Top-5 accuracy
    top5_preds = probabilities.argsort(axis=-1)[:, -5:]
    top5_correct = np.array([label in top5 for label, top5 in zip(labels, top5_preds)])
    top5_accuracy = top5_correct.mean()
    
    results = {
        'accuracy': float(accuracy),
        'top5_accuracy': float(top5_accuracy),
        'macro_f1': float(macro_f1),
        'per_class_accuracy': {
            class_names[i]: float(acc) 
            for i, acc in enumerate(per_class_acc) 
            if not np.isnan(acc)
        }
    }
    
    return results, predictions, probabilities


def main(args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    
    if torch.cuda.is_available():
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
        print(f"Available GPUs: {num_gpus}")
    else:
        print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("BASELINE 2: Zero-Shot X-CLIP Evaluation")
    print("=" * 80)
    
    # Load X-CLIP model
    print("\n[1/4] Loading X-CLIP model...")
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    
    model = model.to(device)
    print(f"Loaded: {args.model_name}")
    
    # Load test dataset
    print("\n[2/4] Loading UCF-101 test dataset...")
    test_dataset = UCF101Dataset(
        split='test',
        processor=processor,
        num_frames=args.num_frames,
        input_size=args.input_size,
        scale_resize=args.scale_resize,
        use_train_augmentation=False,
        cache_dir=args.cache_dir,
        train_subset=1.0
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
    
    # Create text prompts
    print("\n[3/4] Creating text prompts...")
    prompts = create_text_prompts(UCF101_CLASSES)
    print(f"Created {len(prompts)} class prompts with {len(prompts[0])} templates each")
    print(f"Example prompts for '{UCF101_CLASSES[0]}':")
    for prompt in prompts[0]:
        print(f"  - '{prompt}'")
    
    # Extract text features
    text_features = extract_text_features(model, processor, prompts, device)
    print(f"Text features shape: {text_features.shape}")
    
    # Extract video features
    print("\n[4/4] Extracting video features and evaluating...")
    test_indices = list(range(len(test_dataset)))
    video_features, labels = extract_video_features(
        model, processor, test_loader, device,
        use_multi_gpu=args.use_multi_gpu_extraction,
        args=args,
        dataset=test_dataset,
        indices=test_indices
    )
    print(f"Video features shape: {video_features.shape}")
    
    # Evaluate zero-shot
    print("\nPerforming zero-shot classification...")
    results, predictions, probabilities = evaluate_zero_shot(
        video_features, text_features, labels, UCF101_CLASSES
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("ZERO-SHOT RESULTS")
    print("=" * 80)
    print(f"Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Top-5 Accuracy: {results['top5_accuracy']:.4f} ({results['top5_accuracy']*100:.2f}%)")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    
    # Save results
    results_path = output_dir / "baseline2_zeroshot_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline 2: Zero-Shot X-CLIP")
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/amlfs-03/shared/ayaanm/yanav/results/baseline2",
        help="Output directory for results"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/mnt/amlfs-03/shared/ayaanm/yanav/cache",
        help="Cache directory for grouped dataset"
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
        "--batch_size",
        type=int,
        default=8,
        help="Batch size"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers"
    )
    parser.add_argument(
        "--use_multi_gpu_extraction",
        action='store_true',
        help="Use parallel extraction across all GPUs (much faster!)"
    )
    
    args = parser.parse_args()
    main(args)

