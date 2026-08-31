"""
scripts/verify_v2_splits.py
===========================
Independent verification of the V2 split manifests. Re-reads the three CSVs from
disk (it trusts nothing from prepare_v2_data.py's own run) and checks:

  1. every listed image exists on disk and is readable
  2. no image_path appears in more than one split, and none is repeated in one
  3. no two rows in different splits are the SAME image by content hash
  4. every AI row (label 1) has a non-empty generator label
  5. the held-out generator(s) - passed with --heldout or taken from
     data/splits_v2/composition.json - are ABSENT from train AND val
  6. optional held-out real source(s) are absent from train AND val
  7. class counts / generator coverage per split are reported

Exit code 0 only if every hard check passes. Read-only.

    python scripts/verify_v2_splits.py --splits_dir data/splits_v2
    python scripts/verify_v2_splits.py --heldout midjourney,glide --check_files
"""

import argparse
import collections
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLS = ["image_path", "label", "dataset", "generator", "source", "orig_w", "orig_h"]


def read(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLS if c not in (r.fieldnames or [])]
        if missing:
            raise SystemExit(f"{path}: manifest missing columns {missing}; has {r.fieldnames}")
        return list(r)


def sha(path, _buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_buf), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Verify the V2 split manifests (read-only)")
    ap.add_argument("--splits_dir", default="data/splits_v2")
    ap.add_argument("--heldout", default=None,
                    help="comma-separated held-out generator names; default: read composition.json")
    ap.add_argument("--heldout_real", default=None,
                    help="comma-separated held-out real source names (optional)")
    ap.add_argument("--check_files", action="store_true",
                    help="also open every image (slow but definitive)")
    args = ap.parse_args()

    sd = Path(args.splits_dir)
    names = ["train", "val", "realworld_test"]
    m = {n: read(sd / f"{n}.csv") for n in names}

    heldout = set()
    if args.heldout:
        heldout = {g.strip() for g in args.heldout.split(",") if g.strip()}
    elif (sd / "composition.json").exists():
        heldout = set(json.loads((sd / "composition.json").read_text()).get("held_out_generators", []))
    heldout_real = {s.strip() for s in (args.heldout_real or "").split(",") if s.strip()}

    problems, notes = [], []

    # ---- per-split ----
    for n in names:
        rows = m[n]
        paths = [r["image_path"] for r in rows]
        if len(paths) != len(set(paths)):
            problems.append(f"{n}: {len(paths) - len(set(paths))} duplicate image_path rows")
        labels = collections.Counter(int(r["label"]) for r in rows)
        gens = collections.Counter(r["generator"] for r in rows if int(r["label"]) == 1)
        srcs = collections.Counter(r["source"] for r in rows if int(r["label"]) == 0)
        notes.append(f"{n}: n={len(rows)}  real={labels[0]}  ai={labels[1]}  "
                     f"generators={dict(gens)}  real_sources={dict(srcs)}")
        for r in rows:
            if int(r["label"]) == 1 and not r["generator"].strip():
                problems.append(f"{n}: AI row with empty generator: {r['image_path']}")
                break

    # ---- cross-split path disjointness ----
    P = {n: set(r["image_path"] for r in m[n]) for n in names}
    for a, b in (("train", "val"), ("train", "realworld_test"), ("val", "realworld_test")):
        inter = P[a] & P[b]
        if inter:
            problems.append(f"{len(inter)} image_path(s) in BOTH {a} and {b}, e.g. {sorted(inter)[:3]}")

    # ---- cross-split content-hash disjointness ----
    if args.check_files:
        H = {}
        for n in names:
            hs = {}
            for r in m[n]:
                p = ROOT / r["image_path"]
                if not p.exists():
                    problems.append(f"{n}: missing file {r['image_path']}")
                    continue
                try:
                    hs[sha(p)] = r["image_path"]
                except Exception as e:
                    problems.append(f"{n}: unreadable {r['image_path']}: {e}")
            H[n] = hs
        for a, b in (("train", "val"), ("train", "realworld_test"), ("val", "realworld_test")):
            inter = set(H[a]) & set(H[b])
            if inter:
                problems.append(f"{len(inter)} IDENTICAL image(s) by content hash in {a} and {b}")
    else:
        notes.append("content-hash cross-split check skipped (pass --check_files to enable)")

    # ---- held-out generators / sources absent from train+val ----
    for split in ("train", "val"):
        tg = {r["generator"] for r in m[split] if int(r["label"]) == 1}
        for g in heldout:
            if g in tg:
                problems.append(f"held-out generator {g!r} PRESENT in {split}")
        ts = {r["source"] for r in m[split] if int(r["label"]) == 0}
        for s in heldout_real:
            if s in ts:
                problems.append(f"held-out real source {s!r} PRESENT in {split}")
    if heldout:
        rw_g = {r["generator"] for r in m["realworld_test"] if int(r["label"]) == 1}
        if not heldout & rw_g:
            problems.append(f"none of the held-out generators {heldout} appear in realworld_test")
    else:
        notes.append("no held-out generator list available - skipped check 5")

    print("=== V2 split verification ===")
    for ln in notes:
        print("  " + ln)
    print(f"  held-out generators checked: {sorted(heldout) or '(none)'}")
    if problems:
        print("  RESULT: FAIL")
        for p in problems:
            print("    - " + p)
        sys.exit(1)
    print("  RESULT: OK - splits disjoint; held-out generators absent from train/val; "
          "AI rows carry generator labels.")


if __name__ == "__main__":
    main()
