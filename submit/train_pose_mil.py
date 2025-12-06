import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from tqdm import tqdm
import random
import math
from collections import defaultdict
from sklearn.model_selection import train_test_split
from types import SimpleNamespace


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=101, num_heads=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=2000)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.att_V = nn.Linear(hidden_dim, hidden_dim)
        self.att_U = nn.Linear(hidden_dim, hidden_dim)
        self.att_weights = nn.Linear(hidden_dim, 1)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, mask=None):
        key_padding_mask = (mask == 0) if mask is not None else None

        x = self.embedding(x)
        x = self.pos_encoder(x)
        h = self.transformer(x, src_key_padding_mask=key_padding_mask)

        att_v = torch.tanh(self.att_V(h))
        att_u = torch.sigmoid(self.att_U(h))
        att_logits = self.att_weights(att_v * att_u).squeeze(-1)

        if mask is not None:
            att_logits = att_logits.masked_fill(mask == 0, -1e9)

        att_weights = F.softmax(att_logits, dim=1)
        r = torch.bmm(att_weights.unsqueeze(1), h).squeeze(1)

        return self.classifier(r), att_weights


class PoseMILDataset(Dataset):
    def __init__(self, npz_path, class_to_idx, is_training=False):
        data = np.load(npz_path, allow_pickle=True)
        self.clip_ids_raw = data['clip_ids']
        self.vel = data['vel_17d']
        self.speed = data['speed_17d']
        self.is_training = is_training

        self.clip_to_indices = defaultdict(list)
        for idx, clip_id in enumerate(self.clip_ids_raw):
            self.clip_to_indices[str(clip_id)].append(idx)

        self.unique_clips = sorted(list(self.clip_to_indices.keys()))
        self.class_to_idx = class_to_idx

        self.valid_clips = []
        self.labels = []
        for clip_id in self.unique_clips:
            parts = clip_id.split('_')
            if len(parts) >= 2:
                class_name = parts[1]
                if class_name in self.class_to_idx:
                    self.valid_clips.append(clip_id)
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.valid_clips)

    def __getitem__(self, idx):
        clip_id = self.valid_clips[idx]
        label = self.labels[idx]
        indices = self.clip_to_indices[clip_id]

        vel_seq = np.nan_to_num(self.vel[indices], nan=0.0)
        raw_speed = self.speed[indices]
        visibility = (~np.isnan(raw_speed)).astype(np.float32)
        speed_seq = np.nan_to_num(raw_speed, nan=0.0)

        features = np.concatenate([vel_seq.reshape(len(indices), -1), speed_seq, visibility], axis=1)

        if self.is_training:
            if len(features) > 10 and random.random() < 0.5:
                keep_mask = np.random.rand(len(features)) > 0.2
                if keep_mask.sum() > 5:
                    features = features[keep_mask]
            if random.random() < 0.5:
                features = features + np.random.normal(0, 0.01, features.shape)

        return torch.FloatTensor(features), label, clip_id


def collate_fn(batch):
    features, labels, clip_ids = zip(*batch)
    padded = pad_sequence(features, batch_first=True, padding_value=0.0)

    mask = torch.zeros(padded.shape[0], padded.shape[1])
    for i, f in enumerate(features):
        mask[i, :f.shape[0]] = 1.0

    return padded, mask, torch.LongTensor(labels), clip_ids


def get_ucf101_classes():
    dataset = load_dataset("flwrlabs/ucf101", split="train", trust_remote_code=True)
    class_label = dataset.features['label']
    return class_label.names, class_label._str2int


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    class_names, class_to_idx = get_ucf101_classes()
    num_classes = len(class_names)

    full_dataset = PoseMILDataset(args.npz_path, class_to_idx, is_training=False)
    indices = list(range(len(full_dataset)))
    labels = np.array(full_dataset.labels)

    k_shot = getattr(args, "k_shot", 0)
    val_ratio = getattr(args, "val_ratio", 0.1)
    seed = getattr(args, "seed", 42)

    if k_shot and k_shot > 0:
        rng = np.random.RandomState(seed)
        class_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            class_to_indices[label].append(idx)

        train_idx, val_idx = [], []
        for label, cidx in class_to_indices.items():
            cidx = np.array(cidx)
            rng.shuffle(cidx)
            n = len(cidx)
            n_val = max(1, int(n * val_ratio))
            if n_val >= n:
                n_val = max(1, n - 1)
            remaining = n - n_val
            if remaining <= 0:
                continue
            n_train = min(k_shot, remaining)
            val_idx.extend(cidx[:n_val].tolist())
            train_idx.extend(cidx[n_val:n_val + n_train].tolist())
    else:
        train_idx, val_idx = train_test_split(indices, test_size=val_ratio, stratify=labels, random_state=seed)

    train_dataset = PoseMILDataset(args.npz_path, class_to_idx, is_training=True)
    train_dataset.valid_clips = [full_dataset.valid_clips[i] for i in train_idx]
    train_dataset.labels = [full_dataset.labels[i] for i in train_idx]

    val_dataset = PoseMILDataset(args.npz_path, class_to_idx, is_training=False)
    val_dataset.valid_clips = [full_dataset.valid_clips[i] for i in val_idx]
    val_dataset.labels = [full_dataset.labels[i] for i in val_idx]

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size * num_gpus, shuffle=True,
        collate_fn=collate_fn, num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * num_gpus, shuffle=False,
        collate_fn=collate_fn, num_workers=8, pin_memory=True
    )

    model = TransformerMIL(
        input_dim=68, hidden_dim=256, num_classes=num_classes,
        num_heads=4, num_layers=2, dropout=0.4
    )
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    best_model_state = None
    patience = 15
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        for features, mask, labs, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            features, mask, labs = features.to(device), mask.to(device), labs.to(device)
            optimizer.zero_grad()
            logits, _ = model(features, mask)
            loss = criterion(logits, labs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for features, mask, labs, _ in val_loader:
                features, mask, labs = features.to(device), mask.to(device), labs.to(device)
                logits, _ = model(features, mask)
                val_correct += (torch.argmax(logits, dim=1) == labs).sum().item()
                val_total += labs.size(0)

        val_acc = val_correct / val_total
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = (model.module if isinstance(model, nn.DataParallel) else model).state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(best_model_state)
    else:
        model.load_state_dict(best_model_state)
    model.eval()

    full_loader = DataLoader(
        full_dataset, batch_size=args.batch_size * num_gpus,
        shuffle=False, collate_fn=collate_fn, num_workers=8
    )

    all_clip_ids, all_soft_labels = [], []
    with torch.no_grad():
        for features, mask, labs, clip_ids in tqdm(full_loader):
            features, mask = features.to(device), mask.to(device)
            logits, _ = model(features, mask)
            all_clip_ids.extend(clip_ids)
            all_soft_labels.append(F.softmax(logits, dim=1).cpu().numpy())

    np.savez(args.output_path, clip_ids=np.array(all_clip_ids), Y_weak=np.concatenate(all_soft_labels))

    model_save_path = args.output_path.replace('.npz', '_model.pth')
    torch.save((model.module if isinstance(model, nn.DataParallel) else model).state_dict(), model_save_path)


if __name__ == "__main__":
    args = SimpleNamespace(
        npz_path="vitpose_sub8_train_joint_chunk_vels.npz",
        output_path="weak_motion_labels_train.npz",
        epochs=100,
        batch_size=32,
        lr=0.0005,
        k_shot=0,
        val_ratio=0.1,
        seed=42
    )
    train(args)
