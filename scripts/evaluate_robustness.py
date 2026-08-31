"""
scripts/evaluate_robustness.py
==============================
The hackathon robustness benchmark.

Stress-test BOTH trained models on the FROZEN test manifest only
(data/splits/test.csv) under CLEAN plus every required real-world
transformation, at every specified severity. Nothing is trained here.

Key methodology guarantees
--------------------------
* ONLY test.csv images are read - every path is checked against the manifest.
* Transforms are applied ON THE FLY (no transformed files written to disk).
* Both models are scored on the EXACT SAME transformed tensor for every image
  (one DataLoader per condition; each batch is fed to both models unchanged).
* Deterministic: every image's transform is seeded from
  (--seed, condition, parameter, image_index), so it is reproducible and is
  identical for both models. Re-running gives byte-identical inputs.
* This is a STRESS TEST, not random augmentation: EVERY test image receives the
  transform at the given severity. RandomTransform(prob=...) is NOT used.
* Models run in eval() mode, under torch.no_grad(). The input pipeline
  (src.transforms.build_eval_transform) is the plain clean resize->tensor->
  normalise - no augmentation inside it.

Conditions (17 total = 1 clean + 16 transformed)
------------------------------------------------
  clean
  jpeg           quality 90 / 70 / 50 / 30
  blur           Gaussian sigma 0.5 / 1.0 / 2.0
  resize         downscale 0.5x / 0.25x then upscale back
  noise          Gaussian sigma 0.02 / 0.05 / 0.10  (fraction of 255)
  color_jitter   brightness / contrast / saturation, each +/-20%
                 (per-image factor drawn once from [0.8, 1.2] with the fixed
                  seed - reproducible, same for both models, every image jittered)
  center_crop    keep 80%, resize back to model input

Outputs (written to --output_dir, default outputs/robustness_benchmark/)
  results.csv    one row per (model, condition)
  summary.json   per-condition metrics + retention/drop/gap + aggregates
  README.md      description of the benchmark and how to reproduce it

Usage
  # tiny check first (20 images, separate output dir), then STOP:
  python -m scripts.evaluate_robustness --limit 20 --output_dir outputs/_robustness_smoke

  # full benchmark (only when asked):
  python -m scripts.evaluate_robustness --device cuda --num_workers 2
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import build_model
from src.splits import read_manifest
from src.transforms import build_eval_transform
from src.utils import get_device, load_checkpoint, set_seed

import io


# ---------------------------------------------------------------------------
# Condition table.  (name, parameter_label, parameter_value)
# ---------------------------------------------------------------------------
CONDITIONS = [
    ("clean", "-", None),
    ("jpeg", "q90", 90),
    ("jpeg", "q70", 70),
    ("jpeg", "q50", 50),
    ("jpeg", "q30", 30),
    ("blur", "sigma0.5", 0.5),
    ("blur", "sigma1.0", 1.0),
    ("blur", "sigma2.0", 2.0),
    ("resize", "scale0.5", 0.5),
    ("resize", "scale0.25", 0.25),
    ("noise", "sigma0.02", 0.02),
    ("noise", "sigma0.05", 0.05),
    ("noise", "sigma0.10", 0.10),
    ("color_jitter", "brightness_pm20", ("brightness", 0.20)),
    ("color_jitter", "contrast_pm20", ("contrast", 0.20)),
    ("color_jitter", "saturation_pm20", ("saturation", 0.20)),
    ("center_crop", "keep0.80", 0.80),
]

CSV_COLUMNS = [
    "model", "transformation", "parameter",
    "accuracy", "precision", "recall", "f1", "roc_auc",
    "false_positives", "false_negatives",
    "accuracy_retention", "f1_retention",
]


# ---------------------------------------------------------------------------
# Deterministic per-image seed
# ---------------------------------------------------------------------------
def derive_seed(base_seed, *parts):
    """A stable 64-bit int from (base_seed, condition, parameter, index...)."""
    key = "|".join(str(p) for p in (base_seed, *parts))
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


# ---------------------------------------------------------------------------
# The transforms - fixed severity, mirroring src/augmentations.py pixel ops.
# (src/augmentations.py is NOT modified; these force a specific severity.)
# ---------------------------------------------------------------------------
def _jpeg(img, quality):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _blur(img, sigma):
    return img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def _resize(img, scale):
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def _noise(img, sigma_frac, npy_rng):
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    arr += npy_rng.normal(0.0, sigma_frac * 255.0, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _jitter(img, kind, factor):
    enh = {
        "brightness": ImageEnhance.Brightness,
        "contrast": ImageEnhance.Contrast,
        "saturation": ImageEnhance.Color,
    }[kind]
    return enh(img).enhance(factor)


def _center_crop(img, keep):
    w, h = img.size
    cw, ch = int(w * keep), int(h * keep)
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BILINEAR)


def apply_condition(img, name, param, base_seed, idx):
    """Apply ONE condition to a PIL image. Deterministic given (base_seed, idx)."""
    if name == "clean":
        return img
    if name == "jpeg":
        return _jpeg(img, param)
    if name == "blur":
        return _blur(img, param)
    if name == "resize":
        return _resize(img, param)
    if name == "noise":
        rng = np.random.default_rng(derive_seed(base_seed, name, param, idx))
        return _noise(img, param, rng)
    if name == "color_jitter":
        kind, amount = param
        import random
        r = random.Random(derive_seed(base_seed, name, kind, idx))
        factor = r.uniform(1.0 - amount, 1.0 + amount)
        return _jitter(img, kind, factor)
    if name == "center_crop":
        return _center_crop(img, param)
    raise KeyError(f"unknown condition {name!r}")


# ---------------------------------------------------------------------------
# Dataset that yields the transformed tensor for ONE condition
# ---------------------------------------------------------------------------
class ConditionDataset(Dataset):
    def __init__(self, samples, name, param, base_seed, tfm):
        self.samples = samples
        self.name = name
        self.param = param
        self.base_seed = base_seed
        self.tfm = tfm

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = apply_condition(img, self.name, self.param, self.base_seed, idx)
        return self.tfm(img), label


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def score(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    (tn, fp), (fn, tp) = cm
    both_classes = len(set(y_true)) == 2
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if both_classes else None,
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }


@torch.no_grad()
def run_condition(models, samples, name, param, base_seed, tfm, device, batch_size,
                  num_workers, verify=False):
    """
    Score every model on the SAME transformed test images for one condition.

    Returns: {model_name: metrics_dict}, plus (if verify) a sha256 of the first
    transformed batch so reproducibility / identical-input can be asserted.
    """
    ds = ConditionDataset(samples, name, param, base_seed, tfm)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    y_true = []
    y_pred = {m: [] for m in models}
    y_prob = {m: [] for m in models}
    first_batch_hash = None

    for bi, (x, labels) in enumerate(loader):
        x = x.to(device)
        if verify and bi == 0:
            first_batch_hash = hashlib.sha256(
                x.detach().cpu().numpy().tobytes()
            ).hexdigest()
        # EVERY model sees this exact tensor x.
        for mname, model in models.items():
            probs = torch.softmax(model(x), dim=1)[:, 1]
            y_pred[mname].extend((probs >= 0.5).long().cpu().tolist())
            y_prob[mname].extend(probs.cpu().tolist())
        y_true.extend(labels.tolist())

    results = {}
    for mname in models:
        results[mname] = score(y_true, y_pred[mname], y_prob[mname])
    return results, first_batch_hash


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(
        description="ResistAI robustness benchmark: Baseline vs Robust on the "
                    "frozen test set under every required transformation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--baseline_checkpoint", default="outputs/baseline_v1/best_model.pt")
    p.add_argument("--robust_checkpoint", default="outputs/robust_v1/best_model.pt")
    p.add_argument("--test_csv", default="data/splits/test.csv",
                   help="ONLY this manifest is read (held-out test images)")
    p.add_argument("--output_dir", default="outputs/robustness_benchmark")
    p.add_argument("--image_size", type=int, default=64,
                   help="must match how the checkpoints were trained")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None,
                   help="use only the first N test images (for a tiny dry run)")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"[robustness] device = {device}  seed = {args.seed}")

    # ---- load ONLY the frozen test manifest -------------------------------
    test_csv = Path(args.test_csv)
    samples = read_manifest(test_csv)
    manifest_paths = {p for p, _ in samples}
    if args.limit is not None:
        samples = samples[: args.limit]
    n_real = sum(1 for _, y in samples if y == 0)
    n_fake = sum(1 for _, y in samples if y == 1)

    # verification #5: nothing outside test.csv
    outside = [p for p, _ in samples if p not in manifest_paths]
    assert not outside, f"{len(outside)} image path(s) not in {test_csv}!"
    print(f"[robustness] test images: {len(samples)} (real {n_real}, fake {n_fake})  "
          f"all paths in {test_csv} (0 outside)  ✓")

    # ---- build + load both models (eval mode, no grad) -------------------
    models = {}
    for mname, ckpt in (("baseline", args.baseline_checkpoint),
                        ("robust", args.robust_checkpoint)):
        m = build_model(backbone="efficientnet_b0", pretrained=False, num_classes=2)
        m = load_checkpoint(m, ckpt, device)      # sets .eval()
        m.eval()
        models[mname] = m

    tfm = build_eval_transform(args.image_size)   # clean resize->tensor->normalise

    # ---- run every condition -------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []                 # for results.csv
    per_condition = {}        # for summary.json
    clean_metrics = {}        # model -> clean metrics (retention denominators)
    hashes = {}               # condition -> sha256 of first transformed batch
    executed_ok = 0
    t0 = time.time()

    for name, plabel, pvalue in CONDITIONS:
        tag = name if plabel == "-" else f"{name}:{plabel}"
        try:
            res, bhash = run_condition(
                models, samples, name, pvalue, args.seed, tfm, device,
                args.batch_size, args.num_workers, verify=True,
            )
            # verification #2/#3: rebuild the first batch independently and
            # confirm it is byte-identical (deterministic + identical for all).
            res2, bhash2 = run_condition(
                models, samples[: min(len(samples), args.batch_size)], name, pvalue,
                args.seed, tfm, device, args.batch_size, 0, verify=True,
            )
            same = (bhash == bhash2)
            hashes[tag] = {"sha256": bhash, "reproducible": bool(same)}
            assert same, f"{tag}: transformed batch not reproducible!"
            executed_ok += 1
            print(f"[robustness] {tag:28s} ok   first-batch sha256={bhash[:16]}…  "
                  f"reproducible={same}")
        except Exception as e:      # verification #3: report any failure
            print(f"[robustness] {tag:28s} FAILED: {e}")
            raise

        if name == "clean":
            for mname in models:
                clean_metrics[mname] = res[mname]

        cond_entry = {}
        for mname in models:
            m = res[mname]
            cacc = clean_metrics[mname]["accuracy"]
            cf1 = clean_metrics[mname]["f1"]
            acc_ret = (m["accuracy"] / cacc) if cacc else None
            f1_ret = (m["f1"] / cf1) if cf1 else None
            drop = (cacc - m["accuracy"]) if cacc is not None else None
            cond_entry[mname] = {
                **m,
                "accuracy_retention": acc_ret,
                "f1_retention": f1_ret,
                "robustness_drop": drop,
            }
            rows.append({
                "model": mname,
                "transformation": name,
                "parameter": plabel,
                "accuracy": round(m["accuracy"], 6),
                "precision": round(m["precision"], 6),
                "recall": round(m["recall"], 6),
                "f1": round(m["f1"], 6),
                "roc_auc": ("" if m["roc_auc"] is None else round(m["roc_auc"], 6)),
                "false_positives": m["false_positives"],
                "false_negatives": m["false_negatives"],
                "accuracy_retention": ("" if acc_ret is None else round(acc_ret, 6)),
                "f1_retention": ("" if f1_ret is None else round(f1_ret, 6)),
            })
        # robustness gap = robust accuracy - baseline accuracy
        cond_entry["robustness_gap_accuracy"] = (
            cond_entry["robust"]["accuracy"] - cond_entry["baseline"]["accuracy"]
        )
        cond_entry["robustness_gap_f1"] = (
            cond_entry["robust"]["f1"] - cond_entry["baseline"]["f1"]
        )
        per_condition[tag] = cond_entry

    elapsed = time.time() - t0

    # ---- aggregates ---------------------------------------------------
    def agg(model_name, include_clean):
        accs, f1s, aucs = [], [], []
        for name, plabel, _ in CONDITIONS:
            if not include_clean and name == "clean":
                continue
            tag = name if plabel == "-" else f"{name}:{plabel}"
            m = per_condition[tag][model_name]
            accs.append(m["accuracy"])
            f1s.append(m["f1"])
            if m["roc_auc"] is not None:
                aucs.append(m["roc_auc"])
        return {
            "mean_accuracy": float(np.mean(accs)),
            "mean_f1": float(np.mean(f1s)),
            "mean_roc_auc": (float(np.mean(aucs)) if aucs else None),
            "n_conditions": len(accs),
        }

    summary = {
        "seed": args.seed,
        "test_csv": str(test_csv),
        "n_test_images": len(samples),
        "n_real": n_real,
        "n_fake": n_fake,
        "image_size": args.image_size,
        "device": str(device),
        "checkpoints": {
            "baseline": args.baseline_checkpoint,
            "robust": args.robust_checkpoint,
        },
        "elapsed_seconds": round(elapsed, 1),
        "conditions_executed": f"{executed_ok}/{len(CONDITIONS)}",
        "clean": {m: clean_metrics[m] for m in models},
        "per_condition": per_condition,
        "aggregate_all_conditions": {m: agg(m, include_clean=True) for m in models},
        "aggregate_transformed_only": {m: agg(m, include_clean=False) for m in models},
        "transformed_batch_hashes": hashes,
    }

    # ---- write files ------------------------------------------------
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "README.md").write_text(_readme_text(summary, args))

    # verification #4: CSV schema
    with open(csv_path, newline="") as f:
        header = next(csv.reader(f))
    schema_ok = header == CSV_COLUMNS
    print(f"\n[verify] results.csv schema {'OK' if schema_ok else 'MISMATCH'}: {header}")
    assert schema_ok
    print(f"[verify] transformations executed: {executed_ok}/{len(CONDITIONS)} OK")
    print(f"[verify] every transformed batch reproducible (rebuild matches): "
          f"{all(h['reproducible'] for h in hashes.values())}")
    print(f"[verify] both models scored on the identical input tensor per batch "
          f"(same x)  ✓")
    print(f"[verify] image sources: {len(samples)}/{len(samples)} ∈ {test_csv} (0 outside)")

    _print_table(summary)
    print(f"\n[robustness] wrote {csv_path}")
    print(f"[robustness] wrote {out_dir/'summary.json'}")
    print(f"[robustness] wrote {out_dir/'README.md'}")


# ---------------------------------------------------------------------------
# Terminal table + README
# ---------------------------------------------------------------------------
def _print_table(summary):
    print("\n=== Robustness benchmark (accuracy | F1 | ROC-AUC) ===")
    print(f"{'condition':22s} | {'baseline':^24s} | {'robust':^24s} | gap(acc)")
    print("-" * 84)
    for tag, e in summary["per_condition"].items():
        b, r = e["baseline"], e["robust"]
        ba = f"{b['accuracy']:.3f}/{b['f1']:.3f}/" + (f"{b['roc_auc']:.3f}" if b['roc_auc'] is not None else " n/a ")
        ra = f"{r['accuracy']:.3f}/{r['f1']:.3f}/" + (f"{r['roc_auc']:.3f}" if r['roc_auc'] is not None else " n/a ")
        print(f"{tag:22s} | {ba:^24s} | {ra:^24s} | {e['robustness_gap_accuracy']:+.3f}")
    print("-" * 84)
    for label, key in (("ALL conditions", "aggregate_all_conditions"),
                       ("transformed only", "aggregate_transformed_only")):
        b = summary[key]["baseline"]
        r = summary[key]["robust"]
        print(f"mean ({label:16s}) baseline acc={b['mean_accuracy']:.3f} f1={b['mean_f1']:.3f} "
              f"auc={b['mean_roc_auc']:.3f}  |  robust acc={r['mean_accuracy']:.3f} "
              f"f1={r['mean_f1']:.3f} auc={r['mean_roc_auc']:.3f}")


def _readme_text(summary, args):
    ac = summary["aggregate_transformed_only"]
    return f"""# ResistAI robustness benchmark

Stress test of **Baseline v1** vs **Robust v1** on the frozen held-out test set
(`{summary['test_csv']}`, {summary['n_test_images']} images: {summary['n_real']} real /
{summary['n_fake']} AI). No training or validation images are used.

## Method

- Every test image gets **each transformation at each listed severity** (a stress
  test, not `RandomTransform(prob=...)`).
- Transforms are applied **on the fly**; nothing is written to disk.
- For each condition, **both models are scored on the identical transformed
  tensor** (one DataLoader per condition; each batch feeds both models).
- Deterministic: each image's transform is seeded from
  `(seed={summary['seed']}, condition, parameter, image_index)` - reproducible and
  identical for both models. `transformed_batch_hashes` in `summary.json` records
  the sha256 of the first batch per condition.
- Models run in `eval()` under `torch.no_grad()`. Input pipeline is the plain
  clean resize -> tensor -> ImageNet-normalise (`image_size={summary['image_size']}`).

## Conditions (17)

clean; jpeg q90/q70/q50/q30; blur sigma 0.5/1.0/2.0; resize 0.5x/0.25x;
noise sigma 0.02/0.05/0.10; color_jitter brightness/contrast/saturation +/-20%
(per-image factor from [0.8,1.2], fixed seed); center_crop keep 80%.

## Files

- `results.csv` - one row per (model, condition): accuracy, precision, recall,
  f1, roc_auc, false_positives, false_negatives, accuracy_retention, f1_retention.
- `summary.json` - full per-condition metrics + `robustness_drop`
  (clean_acc - transformed_acc), `robustness_gap_accuracy`
  (robust_acc - baseline_acc), and aggregates.

## Headline (transformed conditions only, n={ac['baseline']['n_conditions']})

| model | mean accuracy | mean F1 | mean ROC-AUC |
|---|---|---|---|
| baseline | {ac['baseline']['mean_accuracy']:.4f} | {ac['baseline']['mean_f1']:.4f} | {ac['baseline']['mean_roc_auc']:.4f} |
| robust | {ac['robust']['mean_accuracy']:.4f} | {ac['robust']['mean_f1']:.4f} | {ac['robust']['mean_roc_auc']:.4f} |

## Reproduce

```bash
python -m scripts.evaluate_robustness \\
    --baseline_checkpoint {args.baseline_checkpoint} \\
    --robust_checkpoint {args.robust_checkpoint} \\
    --test_csv {summary['test_csv']} \\
    --image_size {summary['image_size']} --batch_size {args.batch_size} \\
    --device {args.device} --num_workers {args.num_workers} --seed {summary['seed']}
```
"""


if __name__ == "__main__":
    main()
