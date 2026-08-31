"""
train_v2.py  -  ResistAI V2 training (real-world generalisation experiment)
=========================================================================
Objective C: generalise to real-world photographs and unseen AI generators -
NOT higher CIFAKE accuracy.

Differences from V1 (train.py, which is left untouched and still reproduces V1):
  * image_size 224 (native EfficientNet-B0), not the 32 -> 64 pipeline
  * NEW acquisition/domain-randomisation augmentation (src/acquisition.py),
    applied to BOTH classes, so grain / JPEG / downsampling / blur stop
    predicting the label
  * the V1 robustness augmentation group (src/augmentations.py) is still
    available via --robustness_aug for CIFAKE B-benchmark checking
  * longer schedule (~10 epochs), cosine LR, optional label smoothing
  * optional --linear_probe diagnostic (freeze backbone, train head only)
  * V2 manifests carry metadata columns; a held-out generator is excluded from
    train/val by scripts/prepare_v2_data.py, not here

Writes everything to --output_dir (default outputs/v2/). Does NOT write anywhere
under outputs/baseline_v1 or outputs/robust_v1.

Same model class (src/model.build_model, EfficientNet-B0). Same inference maths:
CrossEntropy on 2 logits, P(AI) = softmax(logits)[:, 1], class 0 = REAL / 1 = AI.
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.acquisition import AcquisitionAug, ALL_ACQ_NAMES, describe_acquisition
from src.augmentations import RandomTransform, describe_params
from src.model import build_model, count_parameters
from src.transforms import build_eval_transform, build_train_transform
from src.utils import get_device, set_seed


# ---------------------------------------------------------------------------
class V2Dataset(Dataset):
    """
    Reads a V2 manifest (image_path,label,dataset,generator,source,orig_w,orig_h).
    train transform order:  acquisition aug -> [robustness aug] -> resize/tensor/norm
    eval: clean resize/tensor/norm only.

    Acquisition + robustness aug are applied to BOTH classes identically - the
    label is read only after the pixels are transformed, so nothing can make an
    augmentation class-conditional.
    """

    def __init__(self, rows, image_size, acq=None, rob=None, train=True):
        self.rows = rows
        self.acq = acq
        self.rob = rob
        self.tfm = build_train_transform(image_size) if train else build_eval_transform(image_size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        try:
            img = ImageOps.exif_transpose(Image.open(r["image_path"])).convert("RGB")
        except Exception as e:                       # should not happen: verified up front
            raise RuntimeError(f"unreadable image in manifest: {r['image_path']}: {e}")
        if self.acq is not None:
            img = self.acq(img)          # BOTH classes
        if self.rob is not None:
            img = self.rob(img)          # BOTH classes
        return self.tfm(img), int(r["label"])


def _worker_init(_):
    import numpy as np
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    info = torch.utils.data.get_worker_info()
    for attr in ("acq", "rob"):
        obj = getattr(getattr(info, "dataset", None), attr, None)
        if obj is not None and hasattr(obj, "reseed"):
            obj.reseed(seed)


def read_manifest(path):
    df = pd.read_csv(path)
    need = {"image_path", "label"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path}: manifest needs at least columns {need}, has {list(df.columns)}")
    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(f"{path}: labels must be 0/1, found {sorted(df['label'].unique())}")
    return df.to_dict("records")


def verify_readable(rows, tag, sample=None):
    """Open images so a corrupt file fails loudly BEFORE the run, not mid-epoch.
    sample=None checks every row; an int checks a random subset (fast smoke)."""
    idxs = range(len(rows))
    if sample is not None and sample < len(rows):
        import random as _r
        idxs = _r.Random(0).sample(range(len(rows)), sample)
    bad = []
    for i in idxs:
        try:
            with Image.open(rows[i]["image_path"]) as im:
                im.verify()
        except Exception as e:
            bad.append((rows[i]["image_path"], str(e)))
    if bad:
        for p, e in bad[:10]:
            print(f"  UNREADABLE {p}: {e}")
        raise SystemExit(f"[{tag}] {len(bad)} unreadable image(s) - fix the dataset and re-run")
    print(f"[{tag}] {len(list(idxs))} image(s) checked, all readable")


# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer, device, train, scheduler=None):
    model.train() if train else model.eval()
    tot, seen, yt, yp, ys = 0.0, 0, [], [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in tqdm(loader, leave=False, desc="train" if train else "val"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            tot += loss.item() * x.size(0); seen += x.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1]
            yt += y.cpu().tolist(); yp += (probs >= 0.5).long().cpu().tolist()
            ys += probs.detach().cpu().tolist()
    auc = roc_auc_score(yt, ys) if len(set(yt)) == 2 else float("nan")
    return (tot / max(seen, 1),
            accuracy_score(yt, yp),
            precision_score(yt, yp, pos_label=1, zero_division=0),
            recall_score(yt, yp, pos_label=1, zero_division=0),
            f1_score(yt, yp, pos_label=1, zero_division=0),
            auc)


def parse_args():
    p = argparse.ArgumentParser(description="ResistAI V2 training (image_size 224, acquisition aug)",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--train_csv", default="data/splits_v2/train.csv")
    p.add_argument("--val_csv", default="data/splits_v2/val.csv")
    p.add_argument("--output_dir", default="outputs/v2")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--scheduler", default="cosine", choices=["cosine", "none"])
    p.add_argument("--warmup_frac", type=float, default=0.05)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_val_samples", type=int, default=None)
    p.add_argument("--acq_prob", type=float, default=0.6)
    p.add_argument("--acq_num", type=int, default=2)
    p.add_argument("--acq_transforms", default=",".join(ALL_ACQ_NAMES))
    p.add_argument("--robustness_aug", action="store_true",
                   help="also apply the V1 robustness group (for CIFAKE-B checks)")
    p.add_argument("--linear_probe", action="store_true",
                   help="freeze backbone, train only the classifier head (diagnostic)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verify_images", default="full", choices=["full", "sample", "off"],
                   help="check every manifest image is readable before training "
                        "('sample' = 300 random per split, for a fast smoke run)")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    # extra determinism for a reproducible Colab run (no-ops on MPS/CPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = get_device(args.device)
    out = Path(args.output_dir)
    if out.resolve() in (Path("outputs/baseline_v1").resolve(), Path("outputs/robust_v1").resolve()):
        raise SystemExit("refusing to write into a frozen V1 directory")
    out.mkdir(parents=True, exist_ok=True)
    print(f"[v2] device={device}  image_size={args.image_size}  linear_probe={args.linear_probe}")

    tr = read_manifest(args.train_csv)
    va = read_manifest(args.val_csv)
    # overlap guard - path AND (if present) content hash
    ov = {r["image_path"] for r in tr} & {r["image_path"] for r in va}
    if ov:
        raise SystemExit(f"{len(ov)} image path(s) in BOTH train and val manifests")
    if "content_hash" in tr[0] and "content_hash" in va[0]:
        ovh = {r["content_hash"] for r in tr} & {r["content_hash"] for r in va}
        if ovh:
            raise SystemExit(f"{len(ovh)} identical image(s) by content hash in BOTH train and val")
    if args.max_train_samples:
        tr = tr[: args.max_train_samples]
    if args.max_val_samples:
        va = va[: args.max_val_samples]

    if args.verify_images != "off":
        n = 300 if args.verify_images == "sample" else None
        verify_readable(tr, "train", sample=n)
        verify_readable(va, "val", sample=n)

    def comp(rows):
        df = pd.DataFrame(rows)
        return {"n": len(df), "real": int((df.label == 0).sum()), "ai": int((df.label == 1).sum()),
                "by_dataset": df.get("dataset", pd.Series(dtype=str)).value_counts().to_dict(),
                "by_generator": df.get("generator", pd.Series(dtype=str)).value_counts().to_dict()}
    print("[v2] train composition:", json.dumps(comp(tr)))
    print("[v2] val   composition:", json.dumps(comp(va)))

    acq = AcquisitionAug(prob=args.acq_prob,
                         names=[s.strip() for s in args.acq_transforms.split(",") if s.strip()],
                         num=args.acq_num, seed=args.seed)
    rob = RandomTransform(prob=0.5, num=1, seed=args.seed) if args.robustness_aug else None

    train_ds = V2Dataset(tr, args.image_size, acq=acq, rob=rob, train=True)
    val_ds = V2Dataset(va, args.image_size, acq=None, rob=None, train=False)   # val is CLEAN
    pin = torch.cuda.is_available()
    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    pin_memory=pin, worker_init_fn=_worker_init, drop_last=False)
    vl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                    pin_memory=pin)

    model = build_model(backbone="efficientnet_b0", pretrained=True, num_classes=2, dropout=0.2).to(device)
    if args.linear_probe:
        for pm in model.parameters():
            pm.requires_grad = False
        head = model.get_classifier()
        head_params = list(head.parameters())
        if not head_params:
            raise SystemExit("linear probe: get_classifier() returned no parameters")
        for pm in head_params:
            pm.requires_grad = True
        print(f"[v2] linear probe: backbone frozen, training head "
              f"({sum(p.numel() for p in head_params):,} params)")
    print(f"[v2] trainable params = {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.lr, weight_decay=args.weight_decay)
    steps = max(1, len(tl) * args.epochs)
    if args.scheduler == "cosine":
        warm = max(1, int(steps * args.warmup_frac))
        def lr_lambda(s):
            if s < warm:
                return (s + 1) / warm            # first step trains, not a no-op
            t = (s - warm) / max(1, steps - warm)
            return 0.5 * (1 + math.cos(math.pi * t))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    run_config = {
        "objective": "C: real-world / unseen-generator generalisation",
        "model": "efficientnet_b0", "pretrained": True, "image_size": args.image_size,
        "epochs": args.epochs, "batch_size": args.batch_size, "optimizer": "AdamW",
        "lr": args.lr, "weight_decay": args.weight_decay, "scheduler": args.scheduler,
        "warmup_frac": args.warmup_frac, "label_smoothing": args.label_smoothing,
        "seed": args.seed, "linear_probe": args.linear_probe,
        "train_csv": args.train_csv, "val_csv": args.val_csv,
        "checkpoint_path": str(out / "best_model.pt"),
        "augmentation_groups": {
            "acquisition": {"enabled": True, "prob": args.acq_prob, "num_per_image": args.acq_num,
                            "transforms": args.acq_transforms.split(","),
                            "spec": describe_acquisition(), "applied_to": "BOTH classes"},
            "robustness": {"enabled": bool(args.robustness_aug),
                           "spec": describe_params() if args.robustness_aug else {}},
        },
        "train_composition": comp(tr), "val_composition": comp(va),
    }
    (out / "run_config.json").write_text(json.dumps(run_config, indent=2))

    csv_path = out / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_acc", "val_precision",
                                "val_recall", "val_f1", "val_auc", "lr", "seconds"])

    best_f1, best_ep, ckpt = -1.0, 0, out / "best_model.pt"
    best_row = {}
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        trl, *_ = run_epoch(model, tl, criterion, optimizer, device, True, scheduler)
        vl_, va_, vp_, vr_, vf_, vauc_ = run_epoch(model, vl, criterion, optimizer, device, False)
        cur_lr = optimizer.param_groups[0]["lr"]
        secs = time.time() - t0
        print(f"[v2] epoch {ep:02d}/{args.epochs}  train_loss={trl:.4f}  val_loss={vl_:.4f}  "
              f"acc={va_:.4f}  P={vp_:.4f}  R={vr_:.4f}  F1={vf_:.4f}  AUC={vauc_:.4f}  "
              f"lr={cur_lr:.2e}  ({secs:.0f}s)")
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, f"{trl:.6f}", f"{vl_:.6f}", f"{va_:.6f}", f"{vp_:.6f}",
                                    f"{vr_:.6f}", f"{vf_:.6f}", f"{vauc_:.6f}", f"{cur_lr:.3e}", f"{secs:.1f}"])
        if vf_ > best_f1:
            best_f1, best_ep = vf_, ep
            best_row = {"epoch": ep, "val_loss": vl_, "val_acc": va_, "val_precision": vp_,
                        "val_recall": vr_, "val_f1": vf_, "val_auc": vauc_}
            torch.save(model.state_dict(), ckpt)
            print(f"[v2]   new best val F1 -> saved {ckpt}")

    (out / "metrics.json").write_text(json.dumps({
        "objective": "C: real-world / unseen-generator generalisation",
        "checkpoint_path": str(ckpt), "linear_probe": args.linear_probe,
        "image_size": args.image_size, "epochs": args.epochs, "seed": args.seed,
        "best_epoch": best_ep, "best_val": best_row,
        "note": "val is CLEAN in-distribution; the real signal for V2 is the "
                "realworld_test (C) + shortcut probes, not this number.",
    }, indent=2))
    print(f"[v2] done. best val F1 = {best_f1:.4f} @ epoch {best_ep}  checkpoint {ckpt}")


if __name__ == "__main__":
    main()
