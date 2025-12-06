import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from types import SimpleNamespace
import copy

from baseline1_fewshot import UCF101Dataset, collate_fn, extract_features


def train_and_evaluate_full(X_train, y_train, X_test, y_test, C=1.0,
                            max_iter=1000, patience=10, min_delta=0.001, seed=42):
    n_val = int(0.15 * len(y_train))
    indices = np.random.RandomState(seed).permutation(len(y_train))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    X_train_sub = X_train[train_indices]
    y_train_sub = y_train[train_indices]
    X_val = X_train[val_indices]
    y_val = y_train[val_indices]

    best_val_acc = 0.0
    best_clf = None
    patience_counter = 0

    clf = LogisticRegression(
        C=C, max_iter=1, random_state=seed,
        multi_class='multinomial', solver='lbfgs',
        n_jobs=-1, warm_start=True
    )

    for iteration in range(1, max_iter + 1):
        clf.fit(X_train_sub, y_train_sub)
        val_acc = accuracy_score(y_val, clf.predict(X_val))

        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            best_clf = copy.deepcopy(clf)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    clf = best_clf if best_clf else clf

    test_acc = accuracy_score(y_test, clf.predict(X_test))
    test_f1 = f1_score(y_test, clf.predict(X_test), average='macro')

    return clf, {
        'val_accuracy': float(best_val_acc),
        'test_accuracy': float(test_acc),
        'test_macro_f1': float(test_f1)
    }


def main():
    args = SimpleNamespace(
        model_name="microsoft/xclip-base-patch16",
        cache_dir="/mnt/amlfs-03/shared/ayaanm/yanav/cache",
        num_frames=32,
        input_size=224,
        scale_resize=256,
        batch_size=4,
        num_workers=4,
        use_multi_gpu=True,
        use_train_augmentation=False,
        train_subset=1.0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    train_dataset = UCF101Dataset(
        split='train', processor=processor, num_frames=args.num_frames,
        input_size=args.input_size, scale_resize=args.scale_resize,
        use_train_augmentation=args.use_train_augmentation,
        cache_dir=args.cache_dir, train_subset=args.train_subset
    )

    test_dataset = UCF101Dataset(
        split='test', processor=processor, num_frames=args.num_frames,
        input_size=args.input_size, scale_resize=args.scale_resize,
        use_train_augmentation=False, cache_dir=args.cache_dir
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )

    X_train, y_train = extract_features(
        model, processor, train_loader, device,
        use_multi_gpu=args.use_multi_gpu, args=args,
        dataset=train_dataset, indices=list(range(len(train_dataset)))
    )

    X_test, y_test = extract_features(
        model, processor, test_loader, device,
        use_multi_gpu=args.use_multi_gpu, args=args,
        dataset=test_dataset, indices=list(range(len(test_dataset)))
    )

    _, results = train_and_evaluate_full(X_train, y_train, X_test, y_test)
    return results


if __name__ == "__main__":
    main()

