import os
import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoProcessor, AutoModel
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
from PIL import Image, ImageEnhance
from tqdm import tqdm
import random


class SoftLabelClassifier(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

    def predict(self, x):
        with torch.no_grad():
            return torch.argmax(self.forward(x), dim=1)


def soft_cross_entropy(pred_logits, soft_targets):
    log_probs = F.log_softmax(pred_logits, dim=1)
    return (-torch.sum(soft_targets * log_probs, dim=1)).mean()


def train_pytorch_classifier(X_labeled, y_labeled, X_val, y_val,
                              X_weak=None, y_weak_soft=None,
                              num_classes=101, lr=0.01, epochs=100,
                              batch_size=256, patience=10, device='cuda',
                              labeled_weight=5.0):
    input_dim = X_labeled.shape[1]
    X_labeled_t = torch.FloatTensor(X_labeled).to(device)
    y_labeled_t = torch.LongTensor(y_labeled).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)

    has_weak = X_weak is not None and y_weak_soft is not None
    if has_weak:
        X_weak_t = torch.FloatTensor(X_weak).to(device)
        y_weak_soft_t = torch.FloatTensor(y_weak_soft).to(device)
        n_weak = len(X_weak)

    classifier = SoftLabelClassifier(input_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    n_labeled = len(X_labeled)

    for epoch in range(epochs):
        classifier.train()
        labeled_idx = torch.randperm(n_labeled)
        if has_weak:
            weak_idx = torch.randperm(n_weak)

        for i in range(0, n_labeled, batch_size):
            batch_idx = labeled_idx[i:i+batch_size]
            optimizer.zero_grad()
            loss = F.cross_entropy(classifier(X_labeled_t[batch_idx]), y_labeled_t[batch_idx]) * labeled_weight
            loss.backward()
            optimizer.step()

        if has_weak:
            for i in range(0, n_weak, batch_size):
                batch_idx = weak_idx[i:i+batch_size]
                optimizer.zero_grad()
                loss = soft_cross_entropy(classifier(X_weak_t[batch_idx]), y_weak_soft_t[batch_idx])
                loss.backward()
                optimizer.step()

        classifier.eval()
        with torch.no_grad():
            val_preds = torch.argmax(classifier(X_val_t), dim=1)
            val_acc = (val_preds == y_val_t).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = classifier.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        classifier.load_state_dict(best_state)
    return classifier


def preprocess_frames_train(frames, input_size=224, scale_resize=256, flip_ratio=0.5, apply_augmentation=True):
    processed = []
    do_flip = random.random() < flip_ratio if apply_augmentation else False

    for frame in frames:
        w, h = frame.size
        if h < w:
            new_h, new_w = scale_resize, int(scale_resize * w / h)
        else:
            new_h, new_w = int(scale_resize * h / w), scale_resize
        frame = frame.resize((new_w, new_h), Image.BILINEAR)

        w, h = frame.size
        if apply_augmentation and w > input_size and h > input_size:
            left = random.randint(0, w - input_size)
            top = random.randint(0, h - input_size)
        else:
            left = (w - input_size) // 2
            top = (h - input_size) // 2
        frame = frame.crop((left, top, left + input_size, top + input_size))

        if frame.size != (input_size, input_size):
            frame = frame.resize((input_size, input_size), Image.BILINEAR)

        if do_flip:
            frame = frame.transpose(Image.FLIP_LEFT_RIGHT)

        if apply_augmentation and random.random() < 0.8:
            if random.random() < 0.5:
                frame = ImageEnhance.Brightness(frame).enhance(random.uniform(0.6, 1.4))
            if random.random() < 0.5:
                frame = ImageEnhance.Contrast(frame).enhance(random.uniform(0.6, 1.4))
            if random.random() < 0.5:
                frame = ImageEnhance.Color(frame).enhance(random.uniform(0.6, 1.4))

        if apply_augmentation and random.random() < 0.2:
            frame = frame.convert('L').convert('RGB')

        processed.append(frame)
    return processed


def preprocess_frames_val(frames, input_size=224, scale_resize=256):
    processed = []
    for frame in frames:
        w, h = frame.size
        if h < w:
            new_h, new_w = scale_resize, int(scale_resize * w / h)
        else:
            new_h, new_w = int(scale_resize * h / w), scale_resize
        frame = frame.resize((new_w, new_h), Image.BILINEAR)

        w, h = frame.size
        left, top = (w - input_size) // 2, (h - input_size) // 2
        frame = frame.crop((left, top, left + input_size, top + input_size))
        processed.append(frame)
    return processed


class UCF101Dataset(Dataset):
    def __init__(self, split='train', processor=None, num_frames=32, input_size=224,
                 scale_resize=256, use_train_augmentation=False, cache_dir=None,
                 train_subset=1.0, seed=42):
        self.processor = processor
        self.num_frames = num_frames
        self.split = split
        self.input_size = input_size
        self.scale_resize = scale_resize
        self.use_train_augmentation = use_train_augmentation

        if cache_dir is None:
            cache_dir = "."
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        dataset = load_dataset("flwrlabs/ucf101", split=split)
        metadata_cache = cache_dir / f"ucf101_metadata_{split}.pkl"

        if metadata_cache.exists():
            with open(metadata_cache, 'rb') as f:
                cache_data = pickle.load(f)
                clip_to_indices = cache_data['clip_to_indices']
                clip_labels = cache_data['clip_labels']
        else:
            clip_to_indices = defaultdict(list)
            clip_labels = {}
            clip_frame_nums = defaultdict(list)
            metadata_dataset = dataset.select_columns(['clip_id', 'label', 'frame'])

            for idx in tqdm(range(len(metadata_dataset))):
                row = metadata_dataset[idx]
                clip_id, label, frame = row['clip_id'], row['label'], row['frame']
                clip_to_indices[clip_id].append(idx)
                clip_frame_nums[clip_id].append(frame)
                if clip_id not in clip_labels:
                    clip_labels[clip_id] = label

            for clip_id in clip_to_indices:
                sorted_pairs = sorted(zip(clip_frame_nums[clip_id], clip_to_indices[clip_id]))
                clip_to_indices[clip_id] = [idx for _, idx in sorted_pairs]

            with open(metadata_cache, 'wb') as f:
                pickle.dump({'clip_to_indices': dict(clip_to_indices), 'clip_labels': clip_labels}, f)

        self.dataset = dataset
        self.clip_to_indices = clip_to_indices
        self.clip_labels = clip_labels
        self.clip_ids = sorted(list(self.clip_to_indices.keys()))
        self.labels = [self.clip_labels[clip_id] for clip_id in self.clip_ids]

        if split == 'train' and train_subset < 1.0:
            np.random.seed(seed)
            n_subset = max(1, int(len(self.clip_ids) * train_subset))
            subset_indices = sorted(np.random.choice(len(self.clip_ids), n_subset, replace=False))
            self.clip_ids = [self.clip_ids[i] for i in subset_indices]
            self.labels = [self.labels[i] for i in subset_indices]

        self.num_classes = len(set(self.labels))

    def __len__(self):
        return len(self.clip_ids)

    def __getitem__(self, idx):
        clip_id = self.clip_ids[idx]
        frame_indices = self.clip_to_indices[clip_id]
        label = self.labels[idx]
        frames = self.sample_frames(frame_indices, self.num_frames)
        return frames, label, clip_id

    def sample_frames(self, frame_indices, num_frames):
        total = len(frame_indices)
        if total == 0:
            return None

        if total >= num_frames:
            sample_indices = np.linspace(0, total - 1, num_frames, dtype=int)
        else:
            sample_indices = list(range(total))
            while len(sample_indices) < num_frames:
                sample_indices.append(total - 1)

        sampled = [self.dataset[frame_indices[i]]['image'] for i in sample_indices]

        if self.use_train_augmentation:
            return preprocess_frames_train(sampled, self.input_size, self.scale_resize)
        return preprocess_frames_val(sampled, self.input_size, self.scale_resize)


def collate_fn(batch):
    frames_list, labels, paths = zip(*batch)
    return frames_list, torch.tensor(labels), paths


def create_fewshot_split(train_dataset, test_dataset, k_shot, val_ratio=0.15, seed=42):
    np.random.seed(seed)
    labels = np.array(train_dataset.labels)

    class_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_to_indices[label].append(idx)

    for label in class_to_indices:
        np.random.shuffle(class_to_indices[label])

    splits = {'labeled': [], 'unlabeled': [], 'val': [], 'test': list(range(len(test_dataset)))}

    for label, indices in class_to_indices.items():
        n = len(indices)
        n_val = max(1, int(n * val_ratio))
        n_labeled = min(k_shot, max(1, n - n_val - 1))
        splits['val'].extend(indices[:n_val])
        splits['labeled'].extend(indices[n_val:n_val + n_labeled])
        splits['unlabeled'].extend(indices[n_val + n_labeled:])

    return splits


def get_features_cache_path(cache_dir, split_name, k_shot, seed, model_name):
    import hashlib
    model_hash = hashlib.md5(model_name.encode()).hexdigest()[:8]
    return Path(cache_dir) / f"features_{split_name}_k{k_shot}_s{seed}_{model_hash}.npz"


def extract_features(model, processor, dataloader, device, use_multi_gpu=False,
                     args=None, dataset=None, indices=None, cache_path=None):
    if cache_path is not None and os.path.exists(cache_path):
        data = np.load(cache_path)
        return data['features'], data['labels']

    if use_multi_gpu and args is not None and indices is not None:
        from extract_features_multigpu import extract_features_parallel
        features, labels = extract_features_parallel(
            dataset_split=dataset.split, all_indices=indices,
            args=args, num_gpus=torch.cuda.device_count()
        )
        if cache_path is not None:
            np.savez_compressed(cache_path, features=features, labels=labels)
        return features, labels

    model.eval()
    all_features, all_labels = [], []

    with torch.no_grad():
        for frames_list, labels, _ in tqdm(dataloader):
            batch_features = []
            for frames in frames_list:
                if frames is None:
                    batch_features.append(torch.zeros(512))
                    continue
                try:
                    inputs = processor(text=["dummy"], images=frames, return_tensors="pt", padding=True)
                    pixel_values = inputs['pixel_values'].to(device)
                    if pixel_values.dim() == 4:
                        pixel_values = pixel_values.unsqueeze(0)
                    B, T, C, H, W = pixel_values.shape
                    pixel_values = pixel_values.reshape(-1, C, H, W)

                    if isinstance(model, torch.nn.DataParallel):
                        vision_outputs = model.module.vision_model(pixel_values)
                        image_embeds = model.module.visual_projection(vision_outputs.last_hidden_state)
                    else:
                        vision_outputs = model.vision_model(pixel_values)
                        image_embeds = model.visual_projection(vision_outputs.last_hidden_state)

                    features = F.normalize(image_embeds[:, 0, :].mean(dim=0), dim=-1).cpu()
                    batch_features.append(features)
                except:
                    batch_features.append(torch.zeros(512))

            all_features.append(torch.stack(batch_features))
            all_labels.append(labels)

    all_features = torch.cat(all_features).numpy()
    all_labels = torch.cat(all_labels).numpy()

    if cache_path is not None:
        np.savez_compressed(cache_path, features=all_features, labels=all_labels)
    return all_features, all_labels

