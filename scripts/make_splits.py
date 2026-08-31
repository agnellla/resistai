"""
scripts/make_splits.py
----------------------
Create the ONE reproducible train/val/test split and freeze it as CSV manifests.

Run this exactly once, from the repo root:

    python -m scripts.make_splits --data_dir data/cifake

Writes:
    data/splits/train.csv   (80%)
    data/splits/val.csv     (10%)
    data/splits/test.csv    (10%)

Each row is "image_path,label" with label 0 = REAL, 1 = AI-GENERATED. The split
is stratified per class (balanced real/fake) and seeded, so it is identical on
every machine. It will NOT overwrite existing manifests unless you pass --force.
Images on disk are never copied or modified.

After running, check it with:
    python -m scripts.verify_splits
"""

import argparse
import sys
from pathlib import Path

# Allow "python scripts/make_splits.py" as well as "python -m scripts.make_splits".
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.splits import make_splits, print_report, verify_splits


def main():
    p = argparse.ArgumentParser(
        description="Create the frozen stratified 80/10/10 train/val/test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir", default="data/cifake",
                   help="dataset root with class sub-folders (real/ + fake/)")
    p.add_argument("--out_dir", default="data/splits",
                   help="folder to write train.csv / val.csv / test.csv into")
    p.add_argument("--seed", type=int, default=42,
                   help="random seed - fixes the partition so it is reproducible")
    p.add_argument("--force", action="store_true",
                   help="overwrite existing manifests (invalidates past results!)")
    args = p.parse_args()

    try:
        make_splits(args.data_dir, out_dir=args.out_dir, seed=args.seed, force=args.force)
    except FileExistsError as e:
        print(f"[make_splits] {e}")
        sys.exit(1)
    print()
    print_report(verify_splits(args.out_dir))


if __name__ == "__main__":
    main()
