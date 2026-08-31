"""
train.py  -  ResistAI training pipeline (baseline + transformation-aware)
=======================================================================

Fine-tunes a pretrained EfficientNet-B0 to tell REAL photos (0) from
AI-GENERATED images (1).

Two modes, selected by --augment:
  - WITHOUT --augment : the plain BASELINE (Baseline v1). No augmentation at
    all; identical code path to before, so Baseline v1 stays reproducible.
  - WITH --augment    : TRANSFORMATION-AWARE training. Each training image has
    probability --aug_prob of getting ONE randomly chosen real-world transform
    (JPEG / blur / resize / noise / colour jitter / centre crop) applied on the
    fly. Validation stays CLEAN. The label never changes.

Same model, same optimiser, same frozen train/val/test split in both modes.

What it does, step by step:
  1. Read command-line settings (data dir, output dir, epochs, sample caps, ...).
  2. Fix all random seeds so re-running gives the same result.
  3. Pick the device automatically (CUDA > MPS > CPU), or use --device to force one.
  4. Load the FROZEN train / val split from data/splits/train.csv + val.csv
     (created once by scripts/make_splits.py - this script never re-splits).
     Optionally shrink each with --max_*_samples for fast iteration.
  5. Build EfficientNet-B0 (pretrained) with a fresh 2-class head.
  6. For each epoch: train on the train set, then measure on the validation set.
  7. Print a one-line summary per epoch.
  8. Save the best checkpoint (highest validation F1) to <output_dir>/best_model.pt.
     The checkpoint is a plain state_dict and loads on any device via map_location.
  9. Append every epoch's numbers to <output_dir>/metrics.csv.

Runs on a Colab Tesla T4 (CUDA), an Apple Silicon Mac (MPS), or plain CPU with
no code changes - nothing here assumes CUDA is present.

Example (small, fast, for iterating locally):
    python -m scripts.make_splits --data_dir data/cifake     # once
    python train.py \
        --output_dir outputs/baseline \
        --epochs 3 \
        --max_train_samples 4000 \
        --max_val_samples 1000
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from src.augmentations import ALL_TRANSFORM_NAMES, RandomTransform, describe_params
from src.datasets import build_loaders_from_manifests
from src.model import build_model, count_parameters
from src.utils import get_device, set_seed


# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Train the ResistAI baseline classifier (EfficientNet-B0, "
                    "real vs AI-generated). Works on CUDA / MPS / CPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data manifests (frozen split - created once by scripts/make_splits.py)
    p.add_argument("--train_csv", default="data/splits/train.csv",
                   help="manifest of training images (image_path,label)")
    p.add_argument("--val_csv", default="data/splits/val.csv",
                   help="manifest of validation images (image_path,label)")
    p.add_argument("--output_dir", default="outputs/baseline",
                   help="folder to write best_model.pt, metrics.csv and run_config.json into")

    # Training length / size
    p.add_argument("--epochs", type=int, default=5,
                   help="number of passes over the training set")
    p.add_argument("--batch_size", type=int, default=64,
                   help="images per training/validation batch")
    p.add_argument("--lr", type=float, default=3e-4,
                   help="learning rate for the Adam optimizer")
    p.add_argument("--weight_decay", type=float, default=1e-4,
                   help="L2 weight decay for Adam (light regularisation)")
    p.add_argument("--image_size", type=int, default=224,
                   help="images are resized to this square size before the model")

    # Data subset controls (keep small while iterating)
    p.add_argument("--max_train_samples", type=int, default=None,
                   help="cap on training images from train.csv, class-balanced "
                        "(default: use all). Only for fast iteration.")
    p.add_argument("--max_val_samples", type=int, default=None,
                   help="cap on validation images from val.csv, class-balanced "
                        "(default: use all)")

    # Hardware
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
                   help="compute device; 'auto' picks CUDA > MPS > CPU. "
                        "Asking for one that is missing falls back to auto.")
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader worker processes (use 2 on macOS if it stalls)")

    # Misc
    p.add_argument("--backbone", default="efficientnet_b0",
                   help="timm model name for the backbone")
    p.add_argument("--freeze_backbone", action="store_true",
                   help="train only the new 2-class head, keep the backbone frozen "
                        "(faster, usually a bit weaker)")
    p.add_argument("--seed", type=int, default=42,
                   help="random seed for a reproducible training run "
                        "(the split itself is already frozen in the CSVs)")

    # Transformation-aware training (OFF unless --augment is given, so the
    # baseline stays reproducible)
    p.add_argument("--augment", action="store_true",
                   help="enable transformation-aware training on the TRAIN set "
                        "(validation stays clean)")
    p.add_argument("--aug_prob", type=float, default=0.7,
                   help="probability a training image gets transformed "
                        "(only used with --augment)")
    p.add_argument("--aug_transforms", default=",".join(ALL_TRANSFORM_NAMES),
                   help="comma-separated transforms to draw from: "
                        + ",".join(ALL_TRANSFORM_NAMES))
    p.add_argument("--aug_num", type=int, default=1,
                   help="how many transforms to chain on a transformed image "
                        "(spec: 1)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# One pass over a data loader
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer, device, train):
    """
    Run the model over every batch in `loader` once.

    If train=True  -> compute gradients and update the weights.
    If train=False -> just record predictions (validation).

    Returns: (average_loss, list_of_true_labels, list_of_pred_labels)
    """
    model.train() if train else model.eval()
    running_loss, seen = 0.0, 0
    y_true, y_pred = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False, desc="train" if train else "val"):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)              # shape (batch, 2), raw scores
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            seen += images.size(0)

            preds = logits.argmax(dim=1)        # pick the higher-scoring class
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    avg_loss = running_loss / max(seen, 1)
    return avg_loss, y_true, y_pred


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # 2. Reproducibility.
    set_seed(args.seed)

    # 3. Device (automatic, or forced with --device).
    device = get_device(args.device)
    print(f"[train] device = {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Transformation-aware training: build the on-the-fly augmenter (train only).
    train_augment = None
    aug_names = [s.strip() for s in args.aug_transforms.split(",") if s.strip()]
    if args.augment:
        train_augment = RandomTransform(
            prob=args.aug_prob, names=aug_names, num=args.aug_num, seed=args.seed
        )
        print(f"[train] transformation-aware: prob={args.aug_prob} num={args.aug_num} "
              f"transforms={aug_names}")
    else:
        print("[train] baseline mode (no augmentation)")

    # run_config.json - exact settings used (requirement K).
    run_config = dict(vars(args))
    run_config["model"] = args.backbone
    run_config["augmentation"] = {
        "enabled": bool(args.augment),
        "probability": args.aug_prob,
        "num_per_image": args.aug_num,
        "transform_types": aug_names if args.augment else [],
        "transform_parameters": describe_params() if args.augment else {},
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    # 4. Data - from the FROZEN split manifests (never re-split here).
    train_loader, val_loader, info = build_loaders_from_manifests(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        image_size=args.image_size,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        train_augment=train_augment,
    )
    print(
        f"[train] train {info['train_size']} (real {info['train_real']}, fake {info['train_fake']}) "
        f"from {info['train_csv']} (augmented={info['train_augmented']}) | "
        f"val {info['val_size']} (real {info['val_real']}, fake {info['val_fake']}) "
        f"from {info['val_csv']} (clean)"
    )

    # 5. Model (transfer learning: start from ImageNet weights).
    model = build_model(
        backbone=args.backbone,
        pretrained=True,
        num_classes=2,
        dropout=0.2,
    ).to(device)

    if args.freeze_backbone:
        # Freeze everything, then re-enable just the final classifier layer.
        for param in model.parameters():
            param.requires_grad = False
        head = model.get_classifier()          # timm helper -> the last Linear
        for param in head.parameters():
            param.requires_grad = True
        print("[train] backbone frozen - training classifier head only")

    print(f"[train] trainable params = {count_parameters(model):,}  (limit 2,000,000,000)")

    # Loss + optimizer. CrossEntropyLoss expects raw logits + integer labels.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 9. CSV header.
    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_loss", "val_accuracy",
             "val_precision", "val_recall", "val_f1", "seconds"]
        )

    best_f1 = -1.0
    ckpt_path = out_dir / "best_model.pt"

    # 6. Epoch loop.
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, y_true, y_pred = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        # Metrics. pos_label=1 => "AI-generated" is the positive class.
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        secs = time.time() - t0

        # 7. Progress line.
        print(
            f"[train] epoch {epoch:02d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"acc={acc:.4f}  P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}  "
            f"({secs:.0f}s)"
        )

        # 9. Append to CSV.
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{acc:.6f}",
                 f"{precision:.6f}", f"{recall:.6f}", f"{f1:.6f}", f"{secs:.1f}"]
            )

        # 8. Save best-by-F1 checkpoint.
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), ckpt_path)
            print(f"[train]   new best F1 -> saved {ckpt_path}")

    print(f"[train] done. best val F1 = {best_f1:.4f}")
    print(f"[train] checkpoint : {ckpt_path}")
    print(f"[train] metrics    : {csv_path}")


if __name__ == "__main__":
    main()
