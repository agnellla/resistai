"""
src/datasets.py
---------------
Loads images off disk and hands (image_tensor, label) pairs to PyTorch.

    label 0 = REAL (authentic photo)
    label 1 = AI-GENERATED

We target the CIFAKE dataset layout, but the loader is generic. Point --data_dir
at a folder that contains class sub-folders. Both of these work:

    <data_dir>/train/REAL/*.jpg      <data_dir>/REAL/*.jpg
    <data_dir>/train/FAKE/*.jpg      <data_dir>/FAKE/*.jpg

If a "train" sub-folder exists we use it (the Kaggle CIFAKE also ships a "test"
folder which we ignore here - point evaluate.py at it for held-out testing).

Folder names are matched case-insensitively:
    real / reals / authentic          -> 0
    fake / ai / synthetic / aigenerated -> 1

This file gives you:
  - discover_samples      : scan the directory -> list of (path, label)
  - LabeledImageDataset   : a tiny torch Dataset over that list
  - build_train_val_loaders : reproducible train/val split + DataLoaders,
                              with optional caps for fast iteration
"""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.transforms import build_eval_transform, build_train_transform

# File extensions we treat as images.
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Folder name (lower-cased) -> label.
LABEL_BY_FOLDER = {
    "real": 0, "reals": 0, "authentic": 0, "0_real": 0, "real_images": 0,
    "fake": 1, "ai": 1, "synthetic": 1, "aigenerated": 1, "ai_generated": 1,
    "1_fake": 1, "fake_images": 1,
}


def _list_images(folder):
    """Return a sorted list of image paths under `folder` (recursive)."""
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(str(p) for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)


def _resolve_root(data_dir):
    """If <data_dir>/train exists, train there; otherwise use <data_dir> itself."""
    root = Path(data_dir)
    if (root / "train").is_dir():
        return root / "train"
    return root


def discover_samples(data_dir):
    """
    Walk the class sub-folders and build a list of (image_path, label).

    Raises a clear error if nothing usable is found so beginners get a helpful
    message instead of an empty training run.
    """
    root = _resolve_root(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"--data_dir '{data_dir}' is not a directory")

    samples = []
    matched_folders = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        label = LABEL_BY_FOLDER.get(child.name.lower())
        if label is None:
            continue
        paths = _list_images(child)
        samples.extend((p, label) for p in paths)
        matched_folders.append(f"{child.name} -> label {label} ({len(paths)} images)")

    if not samples:
        raise FileNotFoundError(
            f"No class folders found under '{root}'. Expected sub-folders named "
            f"like REAL / FAKE. Found: {[c.name for c in root.iterdir() if c.is_dir()]}"
        )

    print("[datasets] matched folders:")
    for line in matched_folders:
        print(f"  - {line}")
    return samples


class LabeledImageDataset(Dataset):
    """
    A minimal Dataset over a list of (path, label) pairs.

    `augment` (optional): a callable PIL image -> PIL image applied BEFORE `tfm`.
    Used for transformation-aware training; leave None for clean data
    (validation / test / the plain baseline). It only changes appearance, so
    the label is returned unchanged.
    """

    def __init__(self, samples, tfm, augment=None):
        self.samples = samples
        self.tfm = tfm
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")   # force 3 channels
        if self.augment is not None:
            img = self.augment(img)             # label stays the same
        return self.tfm(img), label


def _class_balanced_cap(samples, max_samples, rng):
    """
    Shuffle `samples` and keep at most `max_samples`, trying to keep the two
    classes balanced. Returns the (possibly smaller) list.
    """
    if max_samples is None or max_samples >= len(samples):
        rng.shuffle(samples)
        return samples

    reals = [s for s in samples if s[1] == 0]
    fakes = [s for s in samples if s[1] == 1]
    rng.shuffle(reals)
    rng.shuffle(fakes)
    half = max_samples // 2
    kept = reals[:half] + fakes[:half]
    rng.shuffle(kept)
    return kept


def build_train_val_loaders(
    data_dir,
    image_size=224,
    batch_size=64,
    val_split=0.15,
    seed=42,
    num_workers=4,
    max_train_samples=None,
    max_val_samples=None,
):
    """
    Build reproducible train and validation DataLoaders.

    Steps:
      1. discover every (path, label) under data_dir
      2. shuffle with a fixed seed, then split off `val_split` for validation
      3. optionally shrink each split (max_*_samples) so we can iterate fast
      4. wrap in Datasets - train uses the light train transform, val uses the
         deterministic eval transform

    Returns:
        (train_loader, val_loader, info_dict)
    """
    samples = discover_samples(data_dir)

    # 2. Deterministic shuffle + split.
    rng = random.Random(seed)
    rng.shuffle(samples)
    n_val = int(len(samples) * val_split)
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    # 3. Optional caps (fresh RNGs so a cap change does not reshuffle the split).
    train_samples = _class_balanced_cap(train_samples, max_train_samples, random.Random(seed + 1))
    val_samples = _class_balanced_cap(val_samples, max_val_samples, random.Random(seed + 2))

    # 4. Datasets + loaders.
    train_ds = LabeledImageDataset(train_samples, build_train_transform(image_size))
    val_ds = LabeledImageDataset(val_samples, build_eval_transform(image_size))

    # pin_memory only helps (and is only supported) for CUDA transfers; leave it
    # off for MPS / CPU so torch does not print a warning.
    pin = torch.cuda.is_available()
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)

    info = {
        "total_found": len(samples),
        "train_size": len(train_samples),
        "val_size": len(val_samples),
        "train_real": sum(1 for _, y in train_samples if y == 0),
        "train_fake": sum(1 for _, y in train_samples if y == 1),
        "val_real": sum(1 for _, y in val_samples if y == 0),
        "val_fake": sum(1 for _, y in val_samples if y == 1),
    }
    return train_loader, val_loader, info


# ===========================================================================
# Manifest-based loading  (the path used by train.py / evaluate.py)
# ---------------------------------------------------------------------------
# Instead of scanning folders and splitting on the fly, these read the frozen
# CSV manifests written once by scripts/make_splits.py. Every experiment then
# uses the exact same train / val / test images.
# ===========================================================================
def _split_counts(samples):
    """Return (n_total, n_real, n_fake) for a list of (path, label) pairs."""
    n_real = sum(1 for _, y in samples if y == 0)
    n_fake = sum(1 for _, y in samples if y == 1)
    return len(samples), n_real, n_fake


def _aug_worker_init(worker_id):
    """
    Give each DataLoader worker its own augmentation randomness. torch already
    sets a distinct base seed per worker; we forward it to numpy and to the
    Dataset's RandomTransform so workers do not all pick identical parameters.
    """
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    info = torch.utils.data.get_worker_info()
    ds = getattr(info, "dataset", None)
    aug = getattr(ds, "augment", None)
    if aug is not None and hasattr(aug, "reseed"):
        aug.reseed(seed)


def _loader(samples, tfm, batch_size, num_workers, shuffle, augment=None):
    pin = torch.cuda.is_available()   # pin_memory only supported for CUDA
    return DataLoader(
        LabeledImageDataset(samples, tfm, augment=augment),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin,
        worker_init_fn=_aug_worker_init if augment is not None else None,
    )


def build_loaders_from_manifests(
    train_csv,
    val_csv,
    image_size=224,
    batch_size=64,
    seed=42,
    num_workers=4,
    max_train_samples=None,
    max_val_samples=None,
    train_augment=None,
):
    """
    Build train + validation DataLoaders from two split manifests.

    The manifests are NOT re-split here - the rows are used exactly as written by
    scripts/make_splits.py. `max_*_samples` only take a smaller class-balanced
    subset for fast iteration (seeded, so it is still reproducible).

    `train_augment` (optional): a PIL->PIL callable applied to TRAINING images
    only (transformation-aware training). Validation stays clean. When None the
    behaviour is identical to Baseline v1.

    Returns: (train_loader, val_loader, info_dict)
    """
    from src.splits import read_manifest      # deferred: avoids an import cycle

    train_samples = read_manifest(train_csv)
    val_samples = read_manifest(val_csv)

    # Guard: the two manifests must not share any image.
    overlap = {p for p, _ in train_samples} & {p for p, _ in val_samples}
    if overlap:
        raise ValueError(
            f"{len(overlap)} image(s) are in BOTH {train_csv} and {val_csv}. "
            f"Re-create the split with scripts/make_splits.py."
        )

    train_samples = _class_balanced_cap(train_samples, max_train_samples, random.Random(seed + 1))
    val_samples = _class_balanced_cap(val_samples, max_val_samples, random.Random(seed + 2))

    train_loader = _loader(train_samples, build_train_transform(image_size),
                           batch_size, num_workers, shuffle=True, augment=train_augment)
    val_loader = _loader(val_samples, build_eval_transform(image_size),
                         batch_size, num_workers, shuffle=False)   # always clean

    t_tot, t_real, t_fake = _split_counts(train_samples)
    v_tot, v_real, v_fake = _split_counts(val_samples)
    info = {
        "train_csv": str(train_csv), "val_csv": str(val_csv),
        "train_size": t_tot, "train_real": t_real, "train_fake": t_fake,
        "val_size": v_tot, "val_real": v_real, "val_fake": v_fake,
        "train_augmented": train_augment is not None,
    }
    return train_loader, val_loader, info


def build_loader_from_manifest(
    csv_path,
    image_size=224,
    batch_size=64,
    seed=42,
    num_workers=4,
    max_samples=None,
):
    """
    Build ONE evaluation DataLoader from a single manifest (e.g. test.csv).
    No shuffling. `max_samples` optionally caps it, class-balanced and seeded.

    Returns: (loader, info_dict)
    """
    from src.splits import read_manifest      # deferred: avoids an import cycle

    samples = read_manifest(csv_path)
    if max_samples is not None:
        samples = _class_balanced_cap(samples, max_samples, random.Random(seed))

    loader = _loader(samples, build_eval_transform(image_size),
                     batch_size, num_workers, shuffle=False)

    tot, n_real, n_fake = _split_counts(samples)
    info = {"csv": str(csv_path), "size": tot, "real": n_real, "fake": n_fake}
    return loader, info
