"""
evaluate.py  -  score a trained ResistAI model on the HELD-OUT test split
=======================================================================

Loads a saved checkpoint and runs it over ONLY the images listed in
data/splits/test.csv - the 10% of the dataset that no model ever trains or
validates on - then prints and saves:

    accuracy, precision, recall, F1, ROC-AUC, and the 2x2 confusion matrix

    label 0 = REAL           positive class for precision/recall/F1/AUC = 1 (AI)
    label 1 = AI-GENERATED

Using the frozen manifest means the baseline model and the (later) robust model
are scored on exactly the same held-out images, so their numbers are comparable.

Works on CUDA / MPS / CPU. The checkpoint is loaded with map_location, so a
model trained on a Colab Tesla T4 evaluates fine on a Mac or on CPU.

Examples:
    python evaluate.py --checkpoint outputs/baseline/best_model.pt
    python evaluate.py --checkpoint outputs/baseline/best_model.pt --device cpu --max_samples 2000
"""

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from src.datasets import build_loader_from_manifest
from src.model import build_model
from src.utils import get_device, load_checkpoint, set_seed


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a trained ResistAI checkpoint on the held-out test "
                    "split (test.csv): accuracy, precision, recall, F1, ROC-AUC, "
                    "confusion matrix. Works on CUDA / MPS / CPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--test_csv", default="data/splits/test.csv",
                   help="manifest of the held-out test images (image_path,label)")
    p.add_argument("--checkpoint", required=True,
                   help="path to the trained state_dict checkpoint (e.g. "
                        "outputs/baseline/best_model.pt)")
    p.add_argument("--output_dir", default=None,
                   help="where to write metrics.json + confusion_matrix.csv "
                        "(default: the checkpoint's folder)")

    p.add_argument("--image_size", type=int, default=224,
                   help="resize images to this square size (must match training)")
    p.add_argument("--batch_size", type=int, default=64,
                   help="images per batch")
    p.add_argument("--max_samples", type=int, default=None,
                   help="cap on evaluated test images, class-balanced "
                        "(default: use ALL of test.csv - recommended)")
    p.add_argument("--seed", type=int, default=42,
                   help="random seed for the optional --max_samples subset")

    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
                   help="compute device; 'auto' picks CUDA > MPS > CPU")
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader worker processes (use 2 on macOS if it stalls)")
    p.add_argument("--backbone", default="efficientnet_b0",
                   help="timm model name; must match the trained checkpoint")
    return p.parse_args()


@torch.no_grad()
def collect_predictions(model, loader, device):
    """
    Run the model over the whole loader once.
    Returns three plain Python lists: true labels, predicted labels,
    and P(class == 1) which ROC-AUC needs.
    """
    y_true, y_pred, y_score = [], [], []
    for images, labels in tqdm(loader, desc="eval", leave=False):
        images = images.to(device)
        probs = torch.softmax(model(images), dim=1)[:, 1]   # P(AI-generated)
        preds = (probs >= 0.5).long()

        y_true.extend(labels.tolist())
        y_pred.extend(preds.cpu().tolist())
        y_score.extend(probs.cpu().tolist())
    return y_true, y_pred, y_score


def main():
    args = parse_args()
    set_seed(args.seed)

    device = get_device(args.device)
    print(f"[evaluate] device = {device}")

    # 1. Load ONLY the held-out test images from the frozen manifest.
    loader, info = build_loader_from_manifest(
        csv_path=args.test_csv,
        image_size=args.image_size,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )
    n_real, n_fake = info["real"], info["fake"]
    print(f"[evaluate] {info['size']} test images  (real {n_real}, fake {n_fake})  "
          f"from {info['csv']}")

    # 2. Build the model and load the trained weights onto `device`.
    model = build_model(backbone=args.backbone, pretrained=False, num_classes=2)
    model = load_checkpoint(model, args.checkpoint, device)

    # 3. Predict, then score.
    y_true, y_pred, y_score = collect_predictions(model, loader, device)

    # float(...) so the values are plain Python floats (sklearn returns numpy
    # scalars, which json.dumps cannot serialise).
    metrics = {
        "n_images": len(y_true),
        "n_real": n_real,
        "n_fake": n_fake,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }
    # ROC-AUC needs both classes present, otherwise it is undefined.
    if len(set(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    else:
        metrics["roc_auc"] = None
        print("[evaluate] only one class present - ROC-AUC skipped")

    # confusion_matrix rows = true class, columns = predicted class, order [0, 1].
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    (tn, fp), (fn, tp) = cm

    # 4. Print a readable report.
    print("\n=== ResistAI evaluation ===")
    for k in ["accuracy", "precision", "recall", "f1"]:
        print(f"  {k:9s}: {metrics[k]:.4f}")
    print(f"  roc_auc  : {metrics['roc_auc']:.4f}" if metrics["roc_auc"] is not None
          else "  roc_auc  : n/a")
    print("\n  confusion matrix")
    print("                    pred REAL   pred AI")
    print(f"    true REAL  |    {tn:8d}   {fp:7d}")
    print(f"    true AI    |    {fn:8d}   {tp:7d}")
    print(f"\n  false positives (REAL called AI): {fp}")
    print(f"  false negatives (AI called REAL): {fn}")

    # 5. Save alongside the checkpoint.
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics["confusion_matrix"] = {
        "true_real_pred_real": int(tn), "true_real_pred_ai": int(fp),
        "true_ai_pred_real": int(fn), "true_ai_pred_ai": int(tp),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(out_dir / "confusion_matrix.csv", "w") as f:
        f.write("true\\pred,REAL,AI\n")
        f.write(f"REAL,{tn},{fp}\n")
        f.write(f"AI,{fn},{tp}\n")
    print(f"\n[evaluate] wrote {out_dir/'metrics.json'} and {out_dir/'confusion_matrix.csv'}")


if __name__ == "__main__":
    main()
