"""
scripts/evaluate_realworld.py
=============================
Evaluation C: real-world / unseen-generator generalisation.

Scores a checkpoint on a held-out manifest (default data/splits_v2/realworld_test.csv)
and breaks the numbers down by:
  * overall          (accuracy, precision, recall, F1, ROC-AUC, confusion matrix,
                      mean P(AI), FP-rate on REAL, TP-rate on AI)
  * per generator            (label 1 rows: mean P(AI), detected-as-AI rate)
  * per real-photo source    (label 0 rows: mean P(AI), false-positive rate)
  * per resolution bucket    (max(orig_w, orig_h))

Read-only. Never trains, never touches V1 or the CIFAKE benchmark. This is a
separate metric from `evaluate.py` (CIFAKE A) and `scripts/evaluate_robustness.py`
(CIFAKE B) - keep the three conceptually separate.

Usage:
  python scripts/evaluate_realworld.py \
      --checkpoint outputs/v2/best_model.pt --image_size 224 \
      --test_csv data/splits_v2/realworld_test.csv \
      --out outputs/v2/realworld_metrics.json
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import build_model
from src.transforms import build_eval_transform
from src.utils import get_device


def _metrics(yt, yp, ys):
    yt = list(yt); yp = list(yp); ys = list(ys)
    both = len(set(yt)) == 2
    if yt:
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        (tn, fp), (fn, tp) = cm
    else:
        tn = fp = fn = tp = 0
    return {
        "n": len(yt),
        "accuracy": float(accuracy_score(yt, yp)) if yt else None,
        "precision": float(precision_score(yt, yp, pos_label=1, zero_division=0)) if yt else None,
        "recall": float(recall_score(yt, yp, pos_label=1, zero_division=0)) if yt else None,
        "f1": float(f1_score(yt, yp, pos_label=1, zero_division=0)) if yt else None,
        "roc_auc": float(roc_auc_score(yt, ys)) if both else None,
        "mean_p_ai": float(sum(ys) / len(ys)) if ys else None,
        "confusion_matrix": {"true_real_pred_real": int(tn), "true_real_pred_ai": int(fp),
                             "true_ai_pred_real": int(fn), "true_ai_pred_ai": int(tp)},
        "false_positive_rate_on_real": float(fp / (tn + fp)) if (tn + fp) else None,
        "true_positive_rate_on_ai": float(tp / (tp + fn)) if (tp + fn) else None,
        "false_positives": int(fp), "false_negatives": int(fn),
    }


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Real-world / unseen-generator eval (C)",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test_csv", default="data/splits_v2/realworld_test.csv")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = get_device(args.device)
    df = pd.read_csv(args.test_csv)
    if not {"image_path", "label"}.issubset(df.columns):
        raise SystemExit(f"{args.test_csv}: needs at least image_path,label; has {list(df.columns)}")
    for c in ("dataset", "generator", "source"):
        if c not in df.columns:
            df[c] = ""
    for c in ("orig_w", "orig_h"):
        if c not in df.columns:
            df[c] = 0
    df["generator"] = df["generator"].fillna("").astype(str)
    df["source"] = df["source"].fillna("").astype(str)
    print(f"[C] {len(df)} images from {args.test_csv}  "
          f"({int((df.label == 0).sum())} real, {int((df.label == 1).sum())} ai)  device={device}")

    model = build_model(backbone="efficientnet_b0", pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()
    tfm = build_eval_transform(args.image_size)

    rows = df.to_dict("records")
    probs = [float("nan")] * len(rows)          # index-aligned with df, filled in place
    buf_tensors, buf_idx = [], []

    def flush():
        if not buf_tensors:
            return
        x = torch.stack(buf_tensors).to(device)
        p = torch.softmax(model(x), dim=1)[:, 1].cpu().tolist()
        for j, v in zip(buf_idx, p):
            probs[j] = v
        buf_tensors.clear(); buf_idx.clear()

    for i, r in enumerate(rows):
        try:
            im = ImageOps.exif_transpose(Image.open(r["image_path"])).convert("RGB")
        except Exception as e:
            print(f"  unreadable {r['image_path']}: {e}")
            continue                             # probs[i] stays NaN, index stays aligned
        buf_tensors.append(tfm(im)); buf_idx.append(i)
        if len(buf_tensors) == args.batch_size:
            flush()
    flush()

    df["p_ai"] = probs
    df["pred"] = [0 if (isinstance(v, float) and math.isnan(v)) else int(v >= 0.5) for v in probs]
    n_unread = int(df["p_ai"].isna().sum())
    if n_unread:
        print(f"[C] {n_unread} image(s) unreadable - excluded from metrics")
    d = df[df["p_ai"].notna()].copy()

    out = {"checkpoint": args.checkpoint, "test_csv": args.test_csv,
           "image_size": args.image_size, "device": str(device),
           "n_unreadable": n_unread,
           "overall": _metrics(d.label.tolist(), d.pred.tolist(), d.p_ai.tolist())}

    out["by_generator"] = {}
    for g, sub in d[d.label == 1].groupby(d["generator"].replace("", "unknown")):
        out["by_generator"][str(g)] = {
            "n": int(len(sub)),
            "mean_p_ai": float(sub.p_ai.mean()),
            "detected_as_ai_rate": float((sub.pred == 1).mean()),
        }

    out["by_real_source"] = {}
    for s, sub in d[d.label == 0].groupby(d["source"].replace("", "unknown")):
        out["by_real_source"][str(s)] = {
            "n": int(len(sub)),
            "mean_p_ai": float(sub.p_ai.mean()),
            "false_positive_rate": float((sub.pred == 1).mean()),
        }

    d = d.assign(_res=d[["orig_w", "orig_h"]].max(axis=1))
    buckets = [(0, 96, "<=96"), (96, 224, "96-224"), (224, 512, "224-512"),
               (512, 1024, "512-1024"), (1024, 10 ** 9, ">1024")]
    out["by_resolution_bucket"] = {}
    for lo, hi, name in buckets:
        sub = d[(d._res > lo) & (d._res <= hi)]
        if len(sub):
            out["by_resolution_bucket"][name] = _metrics(
                sub.label.tolist(), sub.pred.tolist(), sub.p_ai.tolist())

    print(json.dumps(out, indent=2))
    dest = Path(args.out) if args.out else Path(args.checkpoint).parent / "realworld_metrics.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    df.drop(columns=[c for c in ("group", "content_hash") if c in df.columns]).to_csv(
        dest.with_suffix(".predictions.csv"), index=False)
    print(f"[C] wrote {dest} and {dest.with_suffix('.predictions.csv')}")


if __name__ == "__main__":
    main()
