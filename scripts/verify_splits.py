"""
scripts/verify_splits.py
------------------------
Check that the saved split manifests are valid before trusting any result.

    python -m scripts.verify_splits
    python -m scripts.verify_splits --check_files      # also confirm every path exists

Confirms:
  1. no image_path appears in more than one split (and none repeats within a split)
  2. real/fake class counts are balanced in every split
  3. split sizes are 80 / 10 / 10 (within rounding) and add up to the whole dataset

Exits with code 1 if anything is wrong, so it can gate a training run in CI or a
shell script:  python -m scripts.verify_splits && python train.py ...
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.splits import print_report, verify_splits


def main():
    p = argparse.ArgumentParser(
        description="Verify the train/val/test split manifests are disjoint, "
                    "balanced and correctly sized.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--splits_dir", default="data/splits",
                   help="folder holding train.csv / val.csv / test.csv")
    p.add_argument("--balance_tol", type=float, default=0.02,
                   help="max allowed |real-fake|/total per split")
    p.add_argument("--check_files", action="store_true",
                   help="also verify every listed image path exists on disk (slower)")
    args = p.parse_args()

    report = verify_splits(
        args.splits_dir, balance_tol=args.balance_tol, check_files=args.check_files
    )
    print_report(report)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
