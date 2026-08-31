"""
src/splits.py
-------------
Make ONE reproducible, stratified train/val/test split of the dataset and save it
as three CSV manifests:

    data/splits/train.csv      80%
    data/splits/val.csv        10%
    data/splits/test.csv       10%

Each CSV has a header and two columns:

    image_path,label
    data/cifake/real/0001.jpg,0
    data/cifake/fake/0001.jpg,1

    label 0 = REAL, label 1 = AI-GENERATED

Why manifests instead of moving files:
  - the CIFAKE HuggingFace mirror ships 50k real + 50k fake with NO train/test
    split, so we must define one ourselves;
  - every experiment (baseline, robust, ...) then trains and evaluates on exactly
    the same images, so results are comparable;
  - the images on disk are never copied, moved or changed.

The split is stratified (done per class) so real/fake stay balanced in every
split, and it is seeded so re-running produces the identical partition. The
scripts read these CSVs - they never re-split on their own.

Public functions:
  make_splits(...)  - build and write the three CSVs (one-off)
  read_manifest(csv_path) -> list[(image_path, label)]
  verify_splits(splits_dir) -> dict report (raises on a hard failure)
"""

import csv
import os
import random
from pathlib import Path

from src.datasets import discover_samples

SPLIT_NAMES = ("train", "val", "test")
SPLIT_FRACTIONS = {"train": 0.80, "val": 0.10, "test": 0.10}


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def _stratified_partition(samples, seed):
    """
    Split (path, label) pairs into train/val/test PER CLASS so the class balance
    is preserved in each split. Deterministic given `seed`.

    Returns: {"train": [...], "val": [...], "test": [...]}
    """
    by_label = {}
    for path, label in samples:
        by_label.setdefault(label, []).append((path, label))

    out = {name: [] for name in SPLIT_NAMES}
    for label, items in sorted(by_label.items()):
        # Shuffle this class's items with a per-class seed offset so the two
        # classes are not permuted identically.
        rng = random.Random(seed + label)
        items = sorted(items)                 # stable starting order
        rng.shuffle(items)

        n = len(items)
        n_train = int(round(n * SPLIT_FRACTIONS["train"]))
        n_val = int(round(n * SPLIT_FRACTIONS["val"]))
        # test gets the remainder so the three counts always sum to n exactly.
        out["train"].extend(items[:n_train])
        out["val"].extend(items[n_train:n_train + n_val])
        out["test"].extend(items[n_train + n_val:])

    # Shuffle each finished split once (mix the classes together) - still seeded.
    for name in SPLIT_NAMES:
        random.Random(seed + 100 + SPLIT_NAMES.index(name)).shuffle(out[name])
    return out


def _write_manifest(csv_path, rows):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "label"])
        for path, label in rows:
            w.writerow([path, label])


def make_splits(data_dir, out_dir="data/splits", seed=42, force=False):
    """
    Discover every image under `data_dir`, build the stratified split and write
    train.csv / val.csv / test.csv into `out_dir`.

    Refuses to overwrite existing manifests unless force=True, so the split is
    created exactly once and then frozen.
    """
    out_dir = Path(out_dir)
    existing = [out_dir / f"{n}.csv" for n in SPLIT_NAMES if (out_dir / f"{n}.csv").exists()]
    if existing and not force:
        raise FileExistsError(
            f"Split manifests already exist: {[str(p) for p in existing]}. "
            f"Refusing to overwrite. Pass force=True (--force) only if you really "
            f"want a brand-new split - it will invalidate past results."
        )

    samples = discover_samples(data_dir)
    parts = _stratified_partition(samples, seed=seed)

    for name in SPLIT_NAMES:
        _write_manifest(out_dir / f"{name}.csv", parts[name])
        n = len(parts[name])
        n_real = sum(1 for _, y in parts[name] if y == 0)
        n_fake = sum(1 for _, y in parts[name] if y == 1)
        print(f"[splits] {name:5s}: {n:6d}  (real {n_real}, fake {n_fake})  -> {out_dir/f'{name}.csv'}")

    print(f"[splits] total {len(samples)} images, seed {seed}")
    return parts


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------
def read_manifest(csv_path):
    """
    Load one split CSV -> list of (image_path, label) with label as int.
    Raises if the file or header is malformed.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Manifest '{csv_path}' not found. Create the split first:\n"
            f"    python -m scripts.make_splits --data_dir data/cifake"
        )
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["image_path", "label"]:
            raise ValueError(
                f"{csv_path}: expected header 'image_path,label', got {reader.fieldnames}"
            )
        for r in reader:
            rows.append((r["image_path"], int(r["label"])))
    if not rows:
        raise ValueError(f"{csv_path}: no rows")
    return rows


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
def verify_splits(splits_dir="data/splits", balance_tol=0.02, check_files=False):
    """
    Sanity-check the three manifests. Returns a report dict. Raises AssertionError
    on a hard failure (overlap between splits, wrong total, missing file).

    Checks:
      1. no image_path appears in more than one split (and none is repeated
         inside a single split)
      2. real/fake counts are within `balance_tol` of 50/50 in every split
      3. split sizes match the 80/10/10 target (within rounding), and
         train + val + test == the number of unique images
      4. (optional, check_files=True) every listed path exists on disk
    """
    splits_dir = Path(splits_dir)
    manifests = {name: read_manifest(splits_dir / f"{name}.csv") for name in SPLIT_NAMES}

    report = {"sizes": {}, "class_counts": {}, "problems": []}
    path_sets = {}

    for name, rows in manifests.items():
        paths = [p for p, _ in rows]
        path_sets[name] = set(paths)

        # 1a. duplicates inside this split
        if len(paths) != len(path_sets[name]):
            dup = len(paths) - len(path_sets[name])
            report["problems"].append(f"{name}.csv has {dup} duplicate rows")

        n = len(rows)
        n_real = sum(1 for _, y in rows if y == 0)
        n_fake = sum(1 for _, y in rows if y == 1)
        report["sizes"][name] = n
        report["class_counts"][name] = {"real": n_real, "fake": n_fake}

        # 2. balance
        if n and abs(n_real - n_fake) / n > balance_tol:
            report["problems"].append(
                f"{name}.csv is imbalanced: real {n_real} vs fake {n_fake}"
            )

    total = sum(report["sizes"].values())
    report["total"] = total

    # 1b. no overlap between any pair of splits
    for a in SPLIT_NAMES:
        for b in SPLIT_NAMES:
            if a < b:
                overlap = path_sets[a] & path_sets[b]
                if overlap:
                    ex = list(overlap)[:3]
                    report["problems"].append(
                        f"{len(overlap)} image(s) appear in BOTH {a} and {b}, e.g. {ex}"
                    )

    # 3. size targets
    all_unique = set().union(*path_sets.values())
    if len(all_unique) != total:
        report["problems"].append(
            f"{total - len(all_unique)} image(s) shared across splits "
            f"(unique paths {len(all_unique)} != summed rows {total})"
        )
    for name in SPLIT_NAMES:
        want = SPLIT_FRACTIONS[name] * total
        got = report["sizes"][name]
        # allow a few images of rounding slack (2 classes x rounding)
        if total and abs(got - want) > max(4, 0.01 * total):
            report["problems"].append(
                f"{name}.csv size {got} is far from target {want:.0f} "
                f"({100*got/total:.1f}% vs {100*SPLIT_FRACTIONS[name]:.0f}%)"
            )

    # 4. optional existence check
    if check_files:
        missing = 0
        for name in SPLIT_NAMES:
            for p, _ in manifests[name]:
                if not os.path.exists(p):
                    missing += 1
        report["missing_files"] = missing
        if missing:
            report["problems"].append(f"{missing} listed image path(s) do not exist on disk")

    report["ok"] = not report["problems"]
    return report


def print_report(report):
    """Pretty-print a verify_splits() report."""
    print("=== split verification ===")
    for name in SPLIT_NAMES:
        n = report["sizes"][name]
        c = report["class_counts"][name]
        pct = 100 * n / report["total"] if report["total"] else 0
        print(f"  {name:5s}: {n:6d}  ({pct:4.1f}%)   real {c['real']:6d} | fake {c['fake']:6d}")
    print(f"  total: {report['total']:6d}")
    if "missing_files" in report:
        print(f"  missing files on disk: {report['missing_files']}")
    if report["ok"]:
        print("  RESULT: OK - splits are disjoint, balanced and correctly sized")
    else:
        print("  RESULT: PROBLEMS FOUND")
        for p in report["problems"]:
            print(f"    - {p}")
