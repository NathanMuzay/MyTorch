#!/usr/bin/env python3
"""
Train a neural network on labeled chessboards dataset and save the trained model.
Supports both 3-class and 5-class architectures with balanced sampling and early stopping.
"""

import os
import argparse
import hashlib
from collections import defaultdict
import numpy as np

from src.neural_network.model import Sequential
from src.neural_network.layers import Dense, Dropout, BatchNorm1d
from src.neural_network.activations import ReLU, Softmax
from src.neural_network.optimizers import Adam
from src.neural_network.losses import CategoricalCrossEntropy
from src.neural_network.schedulers import CosineAnnealingLR
from src.chess_engine.encoders import encode_boards_parallel, input_dimension, INPUT_DIM

from src.data.labels import CLASSES_3, CLASSES_5, MAPPING_5_TO_3
from src.training.evaluate import classification_metrics, print_classification_metrics
from src.data.augmentation import augment_color_symmetry


def load_dataset_balanced(filepath: str, target_classes: list, max_per_class: int = 5000, seed: int = 42,
                          encoding="basic", strict=False):
    """Reservoir sampling over the whole file, independently for each class."""
    if max_per_class is not None and max_per_class <= 0:
        raise ValueError("max_per_class must be positive")
    class_to_index = {c: i for i, c in enumerate(target_classes)}
    samples_per_class = defaultdict(list)
    seen_per_class = defaultdict(int)
    generator = np.random.default_rng(seed)

    limit = f"max {max_per_class}/class" if max_per_class is not None else "all positions"
    print(f"Reading dataset from {filepath} (target {len(target_classes)} classes, {limit})...")
    with open(filepath, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            parts = line_str.split()
            if len(parts) < 7:
                if strict:
                    raise ValueError(f"Validation line {line_number} missing expected label")
                continue
            fen = " ".join(parts[:6])
            label = " ".join(parts[6:])

            if len(target_classes) == 3 and label not in class_to_index:
                label = MAPPING_5_TO_3.get(label, label)

            if label not in class_to_index:
                if strict:
                    raise ValueError(f"Validation line {line_number} has unknown label: {label}")
                continue

            c_idx = class_to_index[label]
            seen_per_class[c_idx] += 1
            if max_per_class is None or len(samples_per_class[c_idx]) < max_per_class:
                samples_per_class[c_idx].append(fen)
            else:
                replacement = int(generator.integers(seen_per_class[c_idx]))
                if replacement < max_per_class:
                    samples_per_class[c_idx][replacement] = fen

    all_fens = []
    all_labels = []
    for c_idx in sorted(samples_per_class.keys()):
        fens = samples_per_class[c_idx]
        print(f"  Class {c_idx} ({target_classes[c_idx]}): {len(fens)} samples")
        all_fens.extend(fens)
        all_labels.extend([c_idx] * len(fens))

    print(f"Encoding {len(all_fens)} positions in parallel...")
    X = (encode_boards_parallel(all_fens) if encoding == "basic"
         else encode_boards_parallel(all_fens, encoding=encoding))
    y = np.array(all_labels, dtype=np.int32)
    return X, y


def board_keys(inputs):
    """Conservative identity shared by encodings; ignores move counters and EP."""
    return np.packbits(inputs[:, :INPUT_DIM].astype(np.uint8), axis=1)


def partition_fingerprint(inputs, labels):
    """Order-sensitive identity, independent of extra tactical features."""
    digest = hashlib.sha256(board_keys(inputs).tobytes())
    digest.update(np.asarray(labels, dtype="<i4").tobytes())
    return digest.hexdigest()


def stratified_split(labels, seed=42, validation_fraction=0.15):
    """Preserve each class in both partitions when it has at least two samples."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    generator = np.random.default_rng(seed)
    train_indices, validation_indices = [], []
    for label in np.unique(labels):
        indices = generator.permutation(np.flatnonzero(labels == label))
        count = min(len(indices) - 1, max(1, int(round(len(indices) * validation_fraction))))
        validation_indices.extend(indices[:count])
        train_indices.extend(indices[count:])
    if not train_indices or not validation_indices:
        raise ValueError("Not enough positions to create training and validation partitions")
    return generator.permutation(train_indices), generator.permutation(validation_indices)


def build_enhanced_model(num_classes: int, use_batchnorm: bool = True, wide: bool = True,
                         dropout: float = 0.25, input_dim: int = INPUT_DIM) -> Sequential:
    """Build high-capacity deep neural network with BatchNorm1d and Dropout."""
    if wide:
        d1 = dropout if dropout is not None else 0.15
        d2 = 0.1 if dropout is None else min(dropout, 0.1)
        layers = [
            Dense(input_dim, 512, weight_initializer="he"),
            BatchNorm1d(512) if use_batchnorm else None,
            ReLU(),
            Dropout(p=d1),
            Dense(512, 256, weight_initializer="he"),
            BatchNorm1d(256) if use_batchnorm else None,
            ReLU(),
            Dropout(p=d2),
            Dense(256, 128, weight_initializer="he"),
            BatchNorm1d(128) if use_batchnorm else None,
            ReLU(),
            Dense(128, num_classes, weight_initializer="xavier"),
            Softmax(),
        ]
        layers = [l for l in layers if l is not None]
    elif use_batchnorm:
        layers = [
            Dense(input_dim, 256, weight_initializer="he"),
            BatchNorm1d(256),
            ReLU(),
            Dropout(p=dropout),
            Dense(256, 128, weight_initializer="he"),
            BatchNorm1d(128),
            ReLU(),
            Dropout(p=dropout),
            Dense(128, 64, weight_initializer="he"),
            BatchNorm1d(64),
            ReLU(),
            Dropout(p=dropout),
            Dense(64, num_classes, weight_initializer="xavier"),
            Softmax(),
        ]
    else:
        layers = [
            Dense(input_dim, 128, weight_initializer="he"),
            ReLU(),
            Dropout(p=dropout),
            Dense(128, 64, weight_initializer="he"),
            ReLU(),
            Dropout(p=dropout),
            Dense(64, num_classes, weight_initializer="xavier"),
            Softmax(),
        ]
    return Sequential(layers)


def add_training_arguments(parser):
    """Share training options with the project's main entry point."""
    parser.add_argument("--dataset", type=str, default="dataset/cleaned/train.txt", help="Path to labeled dataset")
    parser.add_argument("--validation-dataset", type=str,
                        help="Fixed labeled validation file; use every row and exclude matching training boards")
    parser.add_argument("--encoding", choices=["basic", "tactical"], default="basic",
                        help="Input features: basic (773, default) or tactical (1333)")
    parser.add_argument("--output", type=str, default="my_torch_network_full.nn", help="Output path (.nn file for training)")
    parser.add_argument("--classes", type=int, choices=[3, 5], default=5, help="Number of target classes (3 or 5)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--max-per-class", type=int, default=12000, help="Max samples per class")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling, splitting and training")
    parser.add_argument("--init-seed", type=int, help="Separate initialization/training seed; defaults to --seed")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout probability in hidden layers")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="L2 regularization strength")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Training targets smoothing for CCE (default: disabled)")
    augmentation = parser.add_mutually_exclusive_group()
    augmentation.add_argument("--augment-colors", dest="augment_colors", action="store_true",
                              help="Add color-mirrored training boards, excluding validation matches (default)")
    augmentation.add_argument("--no-augment-colors", dest="augment_colors", action="store_false",
                              help="Disable color symmetry augmentation")
    parser.set_defaults(augment_colors=True)
    parser.add_argument("--patience", type=int, default=6, help="Epochs without validation improvement before stopping")
    parser.add_argument("--monitor", choices=["val_loss", "val_accuracy"], default="val_accuracy",
                        help="Metric used to restore the best model and stop training")
    parser.add_argument("--use-bn", action="store_true", default=True, help="Enable BatchNorm1d layers")
    parser.add_argument("--no-bn", action="store_false", dest="use_bn", help="Disable BatchNorm1d layers")
    parser.add_argument("--boost-mate", action="store_true", default=False, help="Apply higher loss weight on Checkmate classes")
    parser.add_argument("--no-boost-mate", action="store_false", dest="boost_mate", help="Use uniform class weights")
    parser.add_argument("--loss", type=str, choices=["focal", "cce"], default="cce", help="Loss function (focal or cce)")
    return parser


def train_model(args):
    """Train, validate and save the project's Sequential network."""
    for name in ("epochs", "batch_size", "max_per_class", "patience"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be strictly positive")
    if not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("Learning rate must be finite and strictly positive")
    if not np.isfinite(args.dropout) or not 0 <= args.dropout < 1:
        raise ValueError("dropout must be between 0 (included) and 1 (excluded)")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative")
    label_smoothing = args.label_smoothing
    if not np.isfinite(label_smoothing) or not 0 <= label_smoothing < 1:
        raise ValueError("label_smoothing must be finite and between 0 (included) and 1 (excluded)")
    if args.loss != "cce" and label_smoothing:
        raise ValueError("label_smoothing is supported with --loss cce only; use 0 with focal")
    init_seed = args.seed if args.init_seed is None else args.init_seed
    for value in (args.seed, init_seed):
        if not 0 <= value < 2 ** 32:
            raise ValueError("seed and init_seed must be between 0 and 2**32 - 1")
    target_classes = CLASSES_5 if args.classes == 5 else CLASSES_3
    num_classes = len(target_classes)

    dataset_path = args.dataset
    if dataset_path == "dataset/cleaned/train.txt" and not os.path.exists(dataset_path):
        fallback = "dataset/examples/train_chessboards.txt"
        if os.path.exists(fallback):
            print(f"Warning: {dataset_path} not found, falling back to {fallback}")
            dataset_path = fallback
            if args.classes == 5:
                print("Fallback only has 3 classes, setting target classes to 3.")
                target_classes = CLASSES_3
                num_classes = 3

    X, y = load_dataset_balanced(dataset_path, target_classes, max_per_class=args.max_per_class,
                                 seed=args.seed, encoding=args.encoding)
    if len(X) < 2:
        raise ValueError("At least two valid labeled positions are required for training and validation")
    print(f"Total dataset loaded: {len(X)} samples, feature dimension: {X.shape[1]}.")

    if args.validation_dataset:
        X_train, y_train = X, y
        X_val, y_val = load_dataset_balanced(args.validation_dataset, target_classes, max_per_class=None,
                                             seed=args.seed, encoding=args.encoding, strict=True)
        if not len(X_val):
            raise ValueError("Validation dataset is empty or has no recognized labels")
        validation_keys = {row.tobytes() for row in board_keys(X_val)}
        keep = np.array([row.tobytes() not in validation_keys for row in board_keys(X_train)], dtype=bool)
        excluded_overlap = int((~keep).sum())
        X_train, y_train = X_train[keep], y_train[keep]
        if not len(X_train):
            raise ValueError("No training positions remain after excluding validation overlaps")
        split_description = f"Fixed validation: {args.validation_dataset}; {excluded_overlap} training overlaps excluded"
    else:
        # Preserve the existing stratified 85/15 reference partition.
        train_idx, val_idx = stratified_split(y, seed=args.seed)
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        excluded_overlap = 0
        split_description = f"Stratified split (seed={args.seed})"
    # Fancy indexing created independent partitions; release the full encoded corpus.
    del X, y
    print(f"{split_description}: {len(X_train)} training / {len(X_val)} validation positions")
    train_samples_before_augmentation = len(X_train)
    partition = {"method": "external" if args.validation_dataset else "stratified",
                 "validation_dataset": str(args.validation_dataset) if args.validation_dataset else None,
                 "excluded_training_overlaps": excluded_overlap,
                 "training_fingerprint": partition_fingerprint(X_train, y_train),
                 "validation_fingerprint": partition_fingerprint(X_val, y_val)}
    augmentation_stats = {"enabled": bool(args.augment_colors)}
    if args.augment_colors:
        X_train, y_train, stats = augment_color_symmetry(X_train, y_train, X_val, num_classes)
        augmentation_stats.update(stats)
        print(f"Color symmetry: {stats['original_samples']} original + {stats['mirrored_samples']} mirrored "
              f"= {len(X_train)} training positions; {stats['excluded_validation_matches']} validation matches excluded")
        print(f"Validation unchanged: {len(X_val)} positions")

    # Convert y to one-hot
    y_train_onehot = np.zeros((len(y_train), num_classes), dtype=np.float32)
    y_train_onehot[np.arange(len(y_train)), y_train] = 1.0
    # Smooth training labels only: validation remains hard-label CCE for comparison.
    y_train_onehot *= 1.0 - label_smoothing
    y_train_onehot += label_smoothing / num_classes

    y_val_onehot = np.zeros((len(y_val), num_classes), dtype=np.float32)
    y_val_onehot[np.arange(len(y_val)), y_val] = 1.0

    print(f"Building Deep-Wide neural network architecture for {num_classes} classes (BatchNorm={args.use_bn})...")
    np.random.seed(init_seed)
    model = build_enhanced_model(num_classes, use_batchnorm=args.use_bn, wide=True,
                                 dropout=args.dropout, input_dim=input_dimension(args.encoding))
    model.metadata["encoding"] = args.encoding
    model.metadata["encoding_version"] = 1

    # Class weighting to penalize checkmate errors more heavily
    if args.boost_mate:
        if num_classes == 5:
            class_weights = np.array([1.0, 1.2, 1.2, 2.0, 2.0], dtype=np.float32)
        else:
            class_weights = np.array([1.0, 1.2, 2.0], dtype=np.float32)
    else:
        class_weights = None

    if args.loss == "focal":
        from src.neural_network.losses import FocalLoss
        loss = FocalLoss(gamma=2.0, alpha=class_weights)
        print(f"Using Multi-Class Focal Loss (gamma=2.0) with tactical alpha weights")
    else:
        loss = CategoricalCrossEntropy(class_weights=class_weights)
        print("Using Categorical Cross Entropy Loss")
    print(f"Class weights: {class_weights.tolist() if class_weights is not None else 'uniform (1 per class)'}")
    tie_breaker = "val_loss" if args.monitor == "val_accuracy" else "none"
    print(f"Selection metric: {args.monitor}; tie-breaker: {tie_breaker}; "
          f"patience: {args.patience}; restore best weights: True")
    print(f"Initial learning rate: {args.lr}; dropout: {args.dropout}; L2 regularization: {args.weight_decay}")
    print(f"Training label smoothing: {label_smoothing}; validation labels: unchanged")

    optimizer = Adam(learning_rate=args.lr, weight_decay=args.weight_decay)
    model.compile(optimizer=optimizer, loss=loss)
    model.classes = target_classes

    # Cosine Annealing learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    print(f"Training for up to {args.epochs} epochs with CosineAnnealingLR...")
    history = model.fit(
        X_train,
        y_train_onehot,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, y_val_onehot),
        early_stopping_patience=args.patience,
        restore_best_weights=True,
        monitor=args.monitor,
        verbose=True,
        scheduler=scheduler,
    )

    final_loss, final_acc, predictions = model.evaluate(X_val, y_val_onehot, return_predictions=True)
    best_accuracy_epoch = max(range(len(history["val_accuracy"])),
                              key=lambda index: (history["val_accuracy"][index],
                                                 -history["val_loss"][index], -index)) + 1
    best_loss_epoch = int(np.argmin(history["val_loss"])) + 1
    print(f"Best observed validation accuracy: {max(history['val_accuracy']) * 100:.2f}% "
          f"(epoch {best_accuracy_epoch})")
    print(f"Best observed validation loss: {min(history['val_loss']):.4f} (epoch {best_loss_epoch})")
    print(f"Selected model: epoch {model.best_epoch}, monitored metric: {args.monitor}")
    print(f"\nFinal Validation Accuracy: {final_acc * 100:.2f}% (Loss: {final_loss:.4f})")
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (y_val, np.argmax(predictions, axis=1)), 1)
    metrics = classification_metrics(confusion, target_classes)
    print_classification_metrics(metrics)

    model.metadata["training"] = {
        "seed": args.seed, "epochs_run": len(history["loss"]), "best_epoch": model.best_epoch,
        "init_seed": init_seed, "encoding": args.encoding, "input_dim": input_dimension(args.encoding),
        "partition": partition,
        "monitor": args.monitor, "dropout": args.dropout, "weight_decay": args.weight_decay,
        "label_smoothing": label_smoothing,
        "checkpoint_tie_breaker": tie_breaker,
        "augmentation": augmentation_stats, "train_samples_before_augmentation": train_samples_before_augmentation,
        "learning_rate": args.lr, "batch_size": args.batch_size, "max_per_class": args.max_per_class,
        "train_samples": len(X_train), "validation_samples": len(X_val), "history": history,
        "validation_loss": final_loss, "validation_accuracy": final_acc,
        "best_accuracy_epoch": best_accuracy_epoch, "best_loss_epoch": best_loss_epoch,
        "best_observed_accuracy": max(history["val_accuracy"]),
        "best_observed_loss": min(history["val_loss"]), "validation_metrics": metrics,
    }
    model.save(args.output)
    print(f"Model saved to {args.output} with classes: {target_classes}")
    return model


def main(argv=None):
    parser = add_training_arguments(argparse.ArgumentParser(description="Train MyTorch Neural Network"))
    train_model(parser.parse_args(argv))


if __name__ == "__main__":
    main()

