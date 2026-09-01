"""
scripts/prepare_v2_data.py
==========================
Assemble the ResistAI **V2** dataset for objective C (real-world / unseen-generator
generalisation) from folders you control, with:

  * multiple REAL photo sources          (label 0)
  * multiple AI generators, each labelled (label 1)
  * >= 1 AI generator held out ENTIRELY for evaluation (never in train/val)
  * optional held-out REAL source(s) for evaluation
  * a small CIFAKE mix-in (train only) so we can measure catastrophic forgetting
  * exact-content de-duplication across everything
  * optional perceptual-hash grouping so near-duplicates cannot straddle splits

Writes (it NEVER touches the V1 artifacts):
    data/v2_images/<dataset>/<class>/*.jpg     (re-encoded, resized to <=--max_side, q92)
    data/splits_v2/train.csv
    data/splits_v2/val.csv
    data/splits_v2/realworld_test.csv          (the C evaluation set)
    data/splits_v2/composition.json
    data/splits_v2/leakage_report.json

Manifest columns:  image_path,label,dataset,generator,source,orig_w,orig_h
    label: 0 = REAL, 1 = AI      (identical convention to V1)

--------------------------------------------------------------------------------
PRIMARY INPUT: local folders  (--source folders, the default)
--------------------------------------------------------------------------------
You point the script at directories. One directory = one real source OR one
generator. Sub-directories are searched recursively.

    python scripts/prepare_v2_data.py \
        --real_dir imagenet_val=/data/real/imagenet_val \
        --real_dir raise_raw=/data/real/raise \
        --ai_dir   sd14=/data/genimage/sd14/ai \
        --ai_dir   sd15=/data/genimage/sd15/ai \
        --ai_dir   biggan=/data/genimage/biggan/ai \
        --heldout_ai_dir  midjourney=/data/genimage/midjourney/ai \
        --heldout_ai_dir  glide=/data/genimage/glide/ai \
        --heldout_real_dir phone_photos=/data/real/my_phone \
        --cifake_dir data/cifake --cifake_per_class 2500 \
        --max_side 512 --min_side 224 --val_frac 0.12 --seed 42

Recommended concrete assembly (all have per-generator labels):
  * GenImage  (https://github.com/GenImage-Dataset/GenImage) - 8 generators:
        TRAIN   : sd v1.4, sd v1.5, BigGAN, ADM
        HELD OUT: Midjourney, GLIDE            (genuinely different families)
    REAL is ImageNet; add a second real source (RAISE / DIV2K / your own phone
    photos) as --heldout_real_dir so C also tests an unseen camera domain.

--------------------------------------------------------------------------------
EXPERIMENTAL INPUT: elsaEU/ELSA_D3 streaming  (--source elsa)
--------------------------------------------------------------------------------
Kept only for completeness. It is SCIENTIFICALLY WEAKER for this experiment:
  * the 4 "generators" are all SD-family diffusion (SD1.5/2.1/SDXL/DeepFloyd-IF);
    there is no GAN / FLUX / Midjourney, so the held-out generator is not a
    genuinely different family;
  * REAL is a single source (LAION web crops), not multiple camera domains;
  * the parquet schema has changed between dataset versions and is not verified
    from this environment.
It refuses to run without --accept_weak_dataset so you cannot pick it by accident.

Requirements to run:  Pillow (already used by the project). For --source elsa you
also need `datasets` + `huggingface_hub` + network. For perceptual near-dup
grouping install `imagehash` (optional - the script degrades gracefully).
"""

import argparse
import collections
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMG_ROOT = ROOT / "data" / "v2_images"
SPLIT_DIR = ROOT / "data" / "splits_v2"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _key_val_list(pairs):
    """['a=/p', 'b=/q'] -> [('a','/p'), ('b','/q')]  (argparse action='append')."""
    out = []
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"expected NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        out.append((name.strip(), path.strip()))
    return out


def _sha256_file(path, _buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_buf), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_resized(pil, dst: Path, max_side: int):
    from PIL import Image, ImageOps
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    w, h = pil.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        pil = pil.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pil.save(dst, "JPEG", quality=92)
    return w, h            # ORIGINAL dimensions (metadata), not the resized ones


def _iter_images(folder):
    for p in sorted(Path(folder).rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


# ---------------------------------------------------------------------------
# ingest one folder  -> list of row dicts
# ---------------------------------------------------------------------------
def ingest_folder(rows, seen_hashes, name, path, *, label, dataset, generator,
                  source, args, held_out, phasher):
    from PIL import Image
    kind = "ai" if label == 1 else "real"
    n_added, n_dup, n_small, n_bad = 0, 0, 0, 0
    files = list(_iter_images(path))
    if args.limit_per_dir:
        files = files[: args.limit_per_dir]
    if not files:
        raise SystemExit(f"[{name}] no images found under {path!r}")
    for p in files:
        try:
            digest = _sha256_file(p)
            if digest in seen_hashes:
                n_dup += 1
                continue
            with Image.open(p) as im:
                w, h = im.size
                if min(w, h) < args.min_side:
                    n_small += 1
                    continue
                phash = str(phasher(im)) if phasher else None
                dst = IMG_ROOT / dataset / kind / f"{name}_{digest[:16]}.jpg"
                ow, oh = _save_resized(im, dst, args.max_side)
        except Exception as e:                       # unreadable / truncated
            n_bad += 1
            print(f"    skip {p}: {e}")
            continue
        seen_hashes.add(digest)
        rows.append(dict(
            image_path=str(dst.relative_to(ROOT)),
            label=label, dataset=dataset, generator=generator, source=source,
            orig_w=ow, orig_h=oh,
            content_hash=digest,
            group=(phash if phash else digest),      # near-dups share a group
            held_out=bool(held_out),
            is_cifake=(dataset == "CIFAKE"),
        ))
        n_added += 1
    print(f"  [{name}] {dataset}/{kind} gen={generator or '-'} src={source or '-'} "
          f"heldout={held_out}: +{n_added}  (dup {n_dup}, <{args.min_side}px {n_small}, bad {n_bad})")
    return n_added


def _pixel_hash(pil):
    """sha256 over the decoded RGB pixels - identifies pixel-identical images
    that came from an in-memory dataset (no file on disk to hash)."""
    return hashlib.sha256(pil.convert("RGB").tobytes()).hexdigest()


def _load_hf(args):
    """Open the already-downloaded Tiny-GenImage dataset. Never redownloads:
    --hf_local_dir uses load_from_disk; otherwise load_dataset hits the local
    cache (and --offline hard-disables any network call)."""
    if args.hf_local_dir:
        from datasets import load_from_disk
        ds = load_from_disk(args.hf_local_dir)
        if hasattr(ds, "keys"):                       # DatasetDict
            ds = ds[args.hf_split] if args.hf_split in ds else ds[next(iter(ds.keys()))]
        return ds
    if not args.hf_id:
        raise SystemExit("--source tiny_genimage needs --hf_id (HF dataset id) or --hf_local_dir")
    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset
    return load_dataset(args.hf_id, split=args.hf_split)


def ingest_tiny_genimage(rows, seen_hashes, args, phasher):
    """
    Assemble from the already-downloaded Tiny-GenImage HF dataset.

    Uses ACTUAL per-generator availability. Train generators are balanced to
    min(availability, --ai_per_generator, n_real // n_train_generators); the run
    FAILS LOUDLY if that floor drops below --min_per_generator instead of
    silently shipping a thin set. Held-out generators go ONLY into
    realworld_test. CIFAKE (added later) contributes equally to REAL and AI so it
    does not affect class balance.
    """
    import datasets as _d
    from PIL import Image

    ds = _load_hf(args)
    cols = list(ds.column_names)

    # ---- locate the generator + image fields ----
    gfield = args.generator_field
    if gfield == "auto":
        for cand in ("generator", "model", "label", "class", "source", "category", "name", "type"):
            if cand in cols:
                gfield = cand
                break
        else:
            raise SystemExit(f"could not auto-detect the generator field in {cols}; "
                             f"pass --generator_field")
    if gfield not in cols:
        raise SystemExit(f"--generator_field {gfield!r} not in dataset columns {cols}")
    gfeat = ds.features.get(gfield)

    def gen_name(ex):
        v = ex[gfield]
        if gfeat is not None and hasattr(gfeat, "int2str") and isinstance(v, int):
            return gfeat.int2str(v)
        return str(v)

    ifield = args.image_field
    if ifield == "auto":
        for c in cols:
            if isinstance(ds.features[c], _d.Image):
                ifield = c
                break
        else:
            raise SystemExit(f"could not auto-detect the image field in {cols}; pass --image_field")

    real_names = {s.strip().lower() for s in args.real_names.split(",") if s.strip()}
    train_gens = {s.strip() for s in args.train_generators.split(",") if s.strip()}
    held_gens = {s.strip() for s in args.heldout_generators.split(",") if s.strip()}
    if not held_gens:
        raise SystemExit("--heldout_generators must name >= 1 generator (the unseen-generator "
                         "test is the point of objective C)")
    if train_gens & held_gens:
        raise SystemExit(f"generator(s) in BOTH --train_generators and --heldout_generators: "
                         f"{sorted(train_gens & held_gens)}")

    # ---- pass 1: real availability ----
    raw_counts = collections.Counter()
    norm_avail = collections.Counter()
    for ex in ds:
        g = gen_name(ex)
        raw_counts[g] += 1
        norm_avail["REAL" if g.lower() in real_names else g] += 1
    print(f"[tiny_genimage] '{gfield}' raw value counts: {dict(raw_counts)}")
    print(f"[tiny_genimage] normalised availability   : {dict(norm_avail)}")

    if not train_gens:
        train_gens = {g for g in norm_avail if g != "REAL" and g not in held_gens}
    missing = sorted(g for g in (train_gens | held_gens) if norm_avail.get(g, 0) == 0)
    if missing:
        raise SystemExit(f"[tiny_genimage] requested generator(s) absent from the dataset: "
                         f"{missing}. Available: {sorted(norm_avail)}. Fix --train_generators / "
                         f"--heldout_generators.")
    if "REAL" not in norm_avail:
        raise SystemExit(f"[tiny_genimage] no REAL images found (real_names={sorted(real_names)}); "
                         f"pass --real_names matching one of {sorted(norm_avail)}")

    n_train_gens = len(train_gens)
    raw_per_gen = min(norm_avail[g] for g in train_gens)
    if args.ai_per_generator:
        raw_per_gen = min(raw_per_gen, args.ai_per_generator)
    n_real = min(args.n_real, norm_avail["REAL"])
    per_gen = raw_per_gen
    if args.balance_classes:
        per_gen = min(per_gen, n_real // n_train_gens)
    if per_gen < args.min_per_generator:
        raise SystemExit(
            f"[tiny_genimage] per-train-generator count would be {per_gen} "
            f"(< --min_per_generator {args.min_per_generator}). "
            f"availability={{ {', '.join(f'{g}:{norm_avail[g]}' for g in sorted(train_gens))} }}, "
            f"n_real={n_real}, n_train_generators={n_train_gens}"
            + (f", class-balance cap = n_real//n_gens = {n_real // n_train_gens}"
               if args.balance_classes else "") + ". "
            f"Options: fewer --train_generators, more REAL images, --no_balance_classes, "
            f"or lower --min_per_generator deliberately. Not building a thin set.")

    caps = {g: per_gen for g in train_gens}
    for g in held_gens:
        caps[g] = min(args.heldout_count, norm_avail[g])
        if caps[g] < args.min_per_generator:
            raise SystemExit(f"[tiny_genimage] held-out generator {g!r} has only {norm_avail[g]} "
                             f"images (< --min_per_generator {args.min_per_generator}); cannot run "
                             f"a defensible unseen-generator test on it.")
    print(f"[tiny_genimage] PLAN  REAL={n_real}  "
          f"train={{ {', '.join(f'{g}:{caps[g]}' for g in sorted(train_gens))} }}  "
          f"heldout={{ {', '.join(f'{g}:{caps[g]}' for g in sorted(held_gens))} }}  "
          f"(raw_per_gen={raw_per_gen}, class_balanced={args.balance_classes})")

    # ---- pass 2: materialise ----
    got = collections.Counter()
    n_dup = 0
    for i, ex in enumerate(ds):
        g = gen_name(ex)
        is_real = g.lower() in real_names
        key = "REAL" if is_real else g
        if is_real:
            if got["REAL"] >= n_real:
                continue
            label, gen, src, held, sub = 0, "", "GenImage-real", False, "real"
        else:
            if key not in caps or got[key] >= caps[key]:
                continue
            held = key in held_gens
            label, gen, src, sub = 1, key, key, ("heldout_ai" if held else "ai")
        try:
            im = ex[ifield]
            if not hasattr(im, "convert"):
                im = Image.open(io.BytesIO(im["bytes"])) if isinstance(im, dict) else Image.open(im)
            im = im.convert("RGB")
            digest = _pixel_hash(im)
            if digest in seen_hashes:
                n_dup += 1
                continue
            phash = str(phasher(im)) if phasher else None
            dst = IMG_ROOT / "tiny_genimage" / sub / f"{key}_{i}_{digest[:12]}.jpg"
            ow, oh = _save_resized(im, dst, args.max_side)
        except Exception as e:
            print(f"    skip row {i} ({key}): {e}")
            continue
        seen_hashes.add(digest)
        rows.append(dict(
            image_path=str(dst.relative_to(ROOT)), label=label, dataset="tiny_genimage",
            generator=gen, source=src, orig_w=ow, orig_h=oh, content_hash=digest,
            group=(phash if phash else digest), held_out=held, is_cifake=False))
        got[key] += 1
        if got["REAL"] >= n_real and all(got[k] >= v for k, v in caps.items()):
            break
    print(f"[tiny_genimage] collected {dict(got)}  (dup {n_dup})")
    short = {k: v for k, v in caps.items() if got[k] < v}
    if got["REAL"] < n_real:
        short["REAL"] = f"{got['REAL']}/{n_real}"
    if short:
        raise SystemExit(f"[tiny_genimage] could not reach the PLAN counts: {short}. "
                         f"Availability changed under us or the dataset is smaller than its "
                         f"first pass reported. Not fabricating images.")


def ingest_cifake(rows, seen_hashes, args, rng, phasher):
    root = Path(args.cifake_dir)
    real_dir, fake_dir = root / "real", root / "fake"
    if not real_dir.is_dir() or not fake_dir.is_dir():
        print(f"[cifake] {root} not found - skipping the CIFAKE mix-in")
        return
    from PIL import Image
    n = args.cifake_per_class
    for kind, d, label, gen, src in (("real", real_dir, 0, "", "CIFAR-10"),
                                     ("fake", fake_dir, 1, "SD1.4", "SD1.4")):
        pool = sorted(os.listdir(d))
        for f in rng.sample(pool, min(n, len(pool))):
            p = d / f
            try:
                digest = _sha256_file(p)
                if digest in seen_hashes:
                    continue
                with Image.open(p) as im:
                    phash = str(phasher(im)) if phasher else None
                    dst = IMG_ROOT / "CIFAKE" / kind / f
                    ow, oh = _save_resized(im, dst, args.max_side)
            except Exception as e:
                print(f"    skip {p}: {e}")
                continue
            seen_hashes.add(digest)
            rows.append(dict(
                image_path=str(dst.relative_to(ROOT)), label=label, dataset="CIFAKE",
                generator=gen, source=src, orig_w=ow, orig_h=oh, content_hash=digest,
                group=(phash if phash else digest), held_out=False, is_cifake=True))
    print(f"[cifake] mixed in up to {n}/class (TRAIN only, 32px, upsampled at model time)")


# ---------------------------------------------------------------------------
# experimental ELSA_D3 streaming path
# ---------------------------------------------------------------------------
def ingest_elsa(rows, seen_hashes, args, phasher):
    print("\n" + "!" * 78)
    print("!! --source elsa: SCIENTIFICALLY WEAKER dataset (SD-family only, single")
    print("!! real source, unverified schema). Held-out generator will NOT be a")
    print("!! genuinely different family. Proceeding only because --accept_weak_dataset.")
    print("!" * 78 + "\n")
    from datasets import load_dataset
    from PIL import Image
    gens = {0: "SD1.5", 1: "SD2.1", 2: "SDXL1.0", 3: "DeepFloydIF"}
    held = set(g.strip() for g in args.elsa_heldout.split(",") if g.strip())
    train_gens = [g for g in gens.values() if g not in held]
    ds = load_dataset("elsaEU/ELSA_D3", split="train", streaming=True)
    got_real, got_ai = 0, collections.Counter()
    for i, ex in enumerate(ds):
        if got_real < args.n_real and ex.get("image") is not None:
            im = ex["image"]
            dst = IMG_ROOT / "ELSA_D3" / "real" / f"{i}.jpg"
            ow, oh = _save_resized(im, dst, args.max_side)
            rows.append(dict(image_path=str(dst.relative_to(ROOT)), label=0,
                             dataset="ELSA_D3", generator="", source="LAION",
                             orig_w=ow, orig_h=oh, content_hash=f"elsa_real_{i}",
                             group=f"elsa_{i}", held_out=False, is_cifake=False))
            got_real += 1
        for k, gname in gens.items():
            gi = ex.get(f"image_gen{k}")
            if gi is None:
                continue
            in_train = gname in train_gens
            cap = args.ai_per_generator if in_train else args.heldout_count
            if got_ai[gname] >= cap:
                continue
            sub = "ELSA_D3_train_ai" if in_train else "ELSA_D3_heldout_ai"
            dst = IMG_ROOT / sub / "ai" / f"{gname}_{i}.jpg"
            ow, oh = _save_resized(gi, dst, args.max_side)
            rows.append(dict(image_path=str(dst.relative_to(ROOT)), label=1,
                             dataset="ELSA_D3", generator=gname, source="ELSA_D3",
                             orig_w=ow, orig_h=oh, content_hash=f"elsa_{gname}_{i}",
                             group=f"elsa_{i}", held_out=(not in_train), is_cifake=False))
            got_ai[gname] += 1
        done = (got_real >= args.n_real and
                all(got_ai[g] >= (args.ai_per_generator if g in train_gens else args.heldout_count)
                    for g in gens.values()))
        if done:
            break
        if i % 500 == 0:
            print(f"  ... scanned {i}  real {got_real}/{args.n_real}  ai {dict(got_ai)}", flush=True)
    print(f"[elsa] collected real {got_real}, ai {dict(got_ai)}  (groups shared per prompt - "
          f"content is NOT disjoint from train; this is a known ELSA limitation)")


# ---------------------------------------------------------------------------
# manifest IO
# ---------------------------------------------------------------------------
MANIFEST_COLS = ["image_path", "label", "dataset", "generator", "source", "orig_w", "orig_h"]


def write_manifest(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def summarise(name, rs):
    c = collections.Counter((r["dataset"], r["generator"], r["label"]) for r in rs)
    res = collections.Counter()
    for r in rs:
        m = max(int(r["orig_w"] or 0), int(r["orig_h"] or 0))
        b = "<=96" if m <= 96 else "96-224" if m <= 224 else "224-512" if m <= 512 \
            else "512-1024" if m <= 1024 else ">1024"
        res[b] += 1
    return {
        "name": name, "n": len(rs),
        "real": sum(1 for r in rs if r["label"] == 0),
        "ai": sum(1 for r in rs if r["label"] == 1),
        "generators": sorted({r["generator"] for r in rs if r["label"] == 1 and r["generator"]}),
        "real_sources": sorted({r["source"] for r in rs if r["label"] == 0 and r["source"]}),
        "by_dataset_generator_label": {f"{d}/{g or '-'}/{l}": n for (d, g, l), n in sorted(c.items())},
        "by_resolution_bucket": dict(res),
    }


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Assemble the ResistAI V2 dataset (objective C)",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--source", default="folders", choices=["folders", "elsa", "tiny_genimage"])
    ap.add_argument("--real_dir", action="append", metavar="SRC=PATH",
                    help="a real-photo source -> train/val pool (repeatable)")
    ap.add_argument("--ai_dir", action="append", metavar="GEN=PATH",
                    help="an AI generator -> train/val pool (repeatable)")
    ap.add_argument("--heldout_ai_dir", action="append", metavar="GEN=PATH",
                    help="an AI generator -> realworld_test ONLY (repeatable)")
    ap.add_argument("--heldout_real_dir", action="append", metavar="SRC=PATH",
                    help="a real source -> realworld_test ONLY (repeatable)")
    ap.add_argument("--cifake_dir", default="data/cifake")
    ap.add_argument("--cifake_per_class", type=int, default=2500,
                    help="CIFAKE images per class into TRAIN (aim ~20-30%% of train)")
    ap.add_argument("--max_side", type=int, default=512, help="re-encode long side cap")
    ap.add_argument("--min_side", type=int, default=224,
                    help="skip source images whose short side is below this")
    ap.add_argument("--limit_per_dir", type=int, default=None, help="cap images taken per folder")
    ap.add_argument("--val_frac", type=float, default=0.12)
    ap.add_argument("--c_real_holdback_frac", type=float, default=0.15,
                    help="if no --heldout_real_dir, hold back this fraction of pool REAL "
                         "images (group-disjoint) so realworld_test still has real photos")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_phash", action="store_true",
                    help="disable perceptual-hash near-dup grouping even if imagehash is installed")
    # generator / class balancing (folders + tiny_genimage)
    ap.add_argument("--n_real", type=int, default=10000, help="max REAL images (capped by availability)")
    ap.add_argument("--ai_per_generator", type=int, default=2000,
                    help="CEILING on images per TRAIN generator; actual = min(this, availability, "
                         "and n_real//n_train_generators when --balance_classes)")
    ap.add_argument("--heldout_count", type=int, default=2000,
                    help="CEILING on images per HELD-OUT generator (capped by availability)")
    ap.add_argument("--min_per_generator", type=int, default=1000,
                    help="hard floor: if the balanced per-generator count drops below this the "
                         "run FAILS LOUDLY instead of shipping a thin dataset")
    ap.add_argument("--balance_classes", dest="balance_classes", action="store_true", default=True,
                    help="cap train generators so total train REAL ~= total train AI (default on)")
    ap.add_argument("--no_balance_classes", dest="balance_classes", action="store_false")
    ap.add_argument("--balance_train_generators", dest="balance_train_generators",
                    action="store_true", default=True,
                    help="(folders mode) downsample each train generator to the smallest one")
    ap.add_argument("--no_balance_train_generators", dest="balance_train_generators",
                    action="store_false")
    # tiny_genimage-only
    ap.add_argument("--hf_id", default=None, help="HF dataset id for --source tiny_genimage")
    ap.add_argument("--hf_local_dir", default=None, help="local load_from_disk path (no network)")
    ap.add_argument("--hf_split", default="train")
    ap.add_argument("--generator_field", default="auto", help="column naming the generator")
    ap.add_argument("--image_field", default="auto", help="column holding the image")
    ap.add_argument("--real_names", default="real,nature,imagenet,photo,authentic",
                    help="generator-field values (case-insensitive) that mean REAL")
    ap.add_argument("--train_generators", default="",
                    help="comma list of TRAIN generators (default: all non-real, non-heldout)")
    ap.add_argument("--heldout_generators", default="",
                    help="comma list of HELD-OUT generators -> realworld_test ONLY")
    ap.add_argument("--offline", dest="offline", action="store_true", default=True,
                    help="hard-disable HF network; reuse the local cache only (default on)")
    ap.add_argument("--no_offline", dest="offline", action="store_false")
    # elsa-only
    ap.add_argument("--accept_weak_dataset", action="store_true",
                    help="required to use --source elsa")
    ap.add_argument("--elsa_heldout", default="DeepFloydIF")
    args = ap.parse_args()

    if IMG_ROOT.exists() and any(IMG_ROOT.iterdir()):
        raise SystemExit(f"{IMG_ROOT} already exists and is non-empty. Remove it first so the "
                         f"build is reproducible from scratch: rm -rf {IMG_ROOT}")
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # perceptual hasher (optional)
    phasher = None
    if not args.no_phash:
        try:
            import imagehash
            phasher = lambda im: imagehash.phash(im.convert("RGB"))  # noqa: E731
            print("[v2-data] perceptual near-dup grouping: ON (imagehash)")
        except Exception:
            print("[v2-data] imagehash not installed - near-dup grouping OFF "
                  "(exact-hash dedup still applies)")

    rows, seen = [], set()

    if args.source == "tiny_genimage":
        ingest_tiny_genimage(rows, seen, args, phasher)
    elif args.source == "elsa":
        if not args.accept_weak_dataset:
            raise SystemExit("--source elsa is scientifically weaker for objective C "
                             "(SD-family only, single real source, unverified schema). "
                             "Pass --accept_weak_dataset to proceed anyway, or use "
                             "--source folders / --source tiny_genimage.")
        ingest_elsa(rows, seen, args, phasher)
    else:
        real_dirs = _key_val_list(args.real_dir)
        ai_dirs = _key_val_list(args.ai_dir)
        held_ai = _key_val_list(args.heldout_ai_dir)
        held_real = _key_val_list(args.heldout_real_dir)
        if not real_dirs or not ai_dirs:
            raise SystemExit("need at least one --real_dir and one --ai_dir")
        if not held_ai:
            raise SystemExit("need at least one --heldout_ai_dir (the unseen-generator test "
                             "is the point of objective C)")
        gen_names = {g for g, _ in ai_dirs}
        held_names = {g for g, _ in held_ai}
        if gen_names & held_names:
            raise SystemExit(f"generator(s) {gen_names & held_names} are in BOTH --ai_dir and "
                             f"--heldout_ai_dir - a held-out generator must be unseen")
        print("[v2-data] ingesting REAL sources (train/val pool)...")
        for name, path in real_dirs:
            ingest_folder(rows, seen, name, path, label=0, dataset=f"real_{name}",
                          generator="", source=name, args=args, held_out=False, phasher=phasher)
        print("[v2-data] ingesting AI generators (train/val pool)...")
        for name, path in ai_dirs:
            ingest_folder(rows, seen, name, path, label=1, dataset=f"ai_{name}",
                          generator=name, source=name, args=args, held_out=False, phasher=phasher)
        print("[v2-data] ingesting HELD-OUT AI generators (realworld_test only)...")
        for name, path in held_ai:
            ingest_folder(rows, seen, name, path, label=1, dataset=f"heldout_ai_{name}",
                          generator=name, source=name, args=args, held_out=True, phasher=phasher)
        if held_real:
            print("[v2-data] ingesting HELD-OUT REAL sources (realworld_test only)...")
            for name, path in held_real:
                ingest_folder(rows, seen, name, path, label=0, dataset=f"heldout_real_{name}",
                              generator="", source=name, args=args, held_out=True, phasher=phasher)

    print("[v2-data] ingesting CIFAKE mix-in (train only)...")
    ingest_cifake(rows, seen, args, rng, phasher)

    # ---- balance the TRAIN pool (generator-level, then class-level) --------
    # tiny_genimage already balances during ingest; this makes folders mode and
    # any imbalanced source obey the same rule instead of failing on a count.
    def _subsample(pred, keep_per_group, key):
        groups = {}
        for r in rows:
            if pred(r):
                groups.setdefault(key(r), []).append(r)
        if not groups:
            return
        keep = set()
        for gk, items in groups.items():
            items = sorted(items, key=lambda r: r["content_hash"])
            random.Random(f"{args.seed}:{gk}").shuffle(items)
            for r in items[:keep_per_group.get(gk, len(items))]:
                keep.add(id(r))
        before = len(rows)
        rows[:] = [r for r in rows if not pred(r) or id(r) in keep]
        print(f"[v2-data] balancing pass dropped {before - len(rows)} row(s); "
              f"per-group kept={ {k: min(v, keep_per_group.get(k, v)) for k, v in {g: len(i) for g, i in groups.items()}.items()} }")

    train_ai = lambda r: r["label"] == 1 and not r["held_out"] and not r["is_cifake"]
    gen_counts = collections.Counter(r["generator"] for r in rows if train_ai(r))
    if gen_counts:
        n_gens = len(gen_counts)
        target = min(gen_counts.values())
        if args.ai_per_generator:
            target = min(target, args.ai_per_generator)
        n_real_train = sum(1 for r in rows if r["label"] == 0 and not r["held_out"]
                           and not r["is_cifake"])
        if args.balance_classes and n_real_train:
            target = min(target, n_real_train // n_gens)
        if target < args.min_per_generator:
            raise SystemExit(f"[v2-data] per-generator target {target} < --min_per_generator "
                             f"{args.min_per_generator}. availability={dict(gen_counts)}, "
                             f"n_real_train={n_real_train}, n_generators={n_gens}, "
                             f"class-balance cap={n_real_train // n_gens if n_real_train else 'n/a'}. "
                             f"Options: fewer --ai_dir generators, more REAL, --no_balance_classes, "
                             f"or lower --min_per_generator deliberately.")
        if args.balance_train_generators or args.balance_classes:
            print(f"[v2-data] train-generator availability {dict(gen_counts)} -> balancing all to {target}")
            _subsample(train_ai, {g: target for g in gen_counts}, lambda r: r["generator"])

    # ---- assemble splits ---------------------------------------------------
    held_rows = [r for r in rows if r["held_out"]]
    pool = [r for r in rows if not r["held_out"]]

    # guarantee realworld_test has REAL photos: use held-out real if supplied,
    # otherwise hold back a group-disjoint fraction of pool real (never CIFAKE).
    rw_real_from_holdout = [r for r in held_rows if r["label"] == 0]
    c_real = list(rw_real_from_holdout)
    if not rw_real_from_holdout:
        pool_real_groups = {}
        for r in pool:
            if r["label"] == 0 and not r["is_cifake"]:
                pool_real_groups.setdefault(r["group"], []).append(r)
        gkeys = list(pool_real_groups)
        rng.shuffle(gkeys)
        take = int(len(gkeys) * args.c_real_holdback_frac)
        holdback_groups = set(gkeys[:take])
        c_real = [r for g in holdback_groups for r in pool_real_groups[g]]
        pool = [r for r in pool if not (r["label"] == 0 and not r["is_cifake"]
                                        and r["group"] in holdback_groups)]
        print(f"[v2-data] no --heldout_real_dir: held back {len(c_real)} pool REAL images "
              f"({len(holdback_groups)} groups) for realworld_test "
              f"(NOTE: same source(s) as train real)")

    realworld_test = [r for r in held_rows if r["label"] == 1] + c_real

    # group-disjoint train / val split over what's left
    groups = {}
    for r in pool:
        groups.setdefault(r["group"], []).append(r)
    gkeys = list(groups)
    rng.shuffle(gkeys)
    n_val_groups = int(len(gkeys) * args.val_frac)
    val_groups = set(gkeys[:n_val_groups])
    train = [r for g in gkeys if g not in val_groups for r in groups[g]]
    val = [r for g in val_groups for r in groups[g]]
    for lst in (train, val, realworld_test):
        rng.shuffle(lst)

    write_manifest(SPLIT_DIR / "train.csv", train)
    write_manifest(SPLIT_DIR / "val.csv", val)
    write_manifest(SPLIT_DIR / "realworld_test.csv", realworld_test)

    # ---- composition + leakage report -----------------------------------
    comp = {
        "seed": args.seed, "source": args.source,
        "held_out_generators": sorted({r["generator"] for r in realworld_test
                                       if r["label"] == 1 and r["generator"]}),
        "held_out_real_sources": sorted({r["source"] for r in realworld_test
                                         if r["label"] == 0 and r["source"]
                                         and any(x["held_out"] and x["source"] == r["source"]
                                                 for x in held_rows)}),
        "max_side": args.max_side, "min_side": args.min_side,
        "splits": [summarise("train", train), summarise("val", val),
                   summarise("realworld_test", realworld_test)],
    }
    (SPLIT_DIR / "composition.json").write_text(json.dumps(comp, indent=2))

    tr_p, va_p, rw_p = ({r["image_path"] for r in s} for s in (train, val, realworld_test))
    tr_h, va_h, rw_h = ({r["content_hash"] for r in s} for s in (train, val, realworld_test))
    tr_g = {r["generator"] for r in train if r["label"] == 1}
    va_g = {r["generator"] for r in val if r["label"] == 1}
    held_gen_names = sorted({r["generator"] for r in realworld_test
                             if r["label"] == 1 and any(h["held_out"] and h["generator"] == r["generator"]
                                                        for h in held_rows)})
    problems = []
    if tr_p & va_p:               problems.append(f"{len(tr_p & va_p)} path(s) in train AND val")
    if (tr_p | va_p) & rw_p:      problems.append(f"{len((tr_p | va_p) & rw_p)} path(s) in pool AND realworld_test")
    if tr_h & va_h:               problems.append(f"{len(tr_h & va_h)} identical image(s) (hash) in train AND val")
    if (tr_h | va_h) & rw_h:      problems.append(f"{len((tr_h | va_h) & rw_h)} identical image(s) (hash) in pool AND realworld_test")
    for g in held_gen_names:
        if g in tr_g:             problems.append(f"held-out generator {g!r} present in TRAIN")
        if g in va_g:             problems.append(f"held-out generator {g!r} present in VAL")
    if any(r["label"] == 1 and not r["generator"] for r in train + val + realworld_test):
        problems.append("some AI rows have an empty generator label")
    dup_paths = [p for p, n in collections.Counter(
        r["image_path"] for r in train + val + realworld_test).items() if n > 1]
    if dup_paths:                 problems.append(f"{len(dup_paths)} duplicate image_path across manifests")
    rw_labels = {r["label"] for r in realworld_test}
    if rw_labels != {0, 1}:
        problems.append(f"realworld_test must contain BOTH classes, has labels {sorted(rw_labels)}")
    train_gen_counts = collections.Counter(r["generator"] for r in train if r["label"] == 1)
    val_gen_counts = collections.Counter(r["generator"] for r in val if r["label"] == 1)
    # check the assembled pool per generator (train+val), not the post-split train count
    pool_gen_counts = train_gen_counts + val_gen_counts
    thin = {g: n for g, n in pool_gen_counts.items()
            if g and n < args.min_per_generator and g != "SD1.4"}   # SD1.4 = CIFAKE mix-in
    if thin:
        problems.append(f"train-pool generator(s) below --min_per_generator "
                        f"{args.min_per_generator}: {thin}")
    rw_gen_counts = collections.Counter(r["generator"] for r in realworld_test if r["label"] == 1)

    leak = {
        "checked": ["path disjoint", "content-hash disjoint",
                    "held-out generators absent from train+val",
                    "AI rows have generator labels", "no duplicate paths",
                    "realworld_test has both classes",
                    "train generators >= --min_per_generator"],
        "held_out_generators": held_gen_names,
        "train_generators": sorted(tr_g), "val_generators": sorted(va_g),
        "train_generator_counts": dict(train_gen_counts),
        "realworld_test_generator_counts": dict(rw_gen_counts),
        "realworld_test_real": sum(1 for r in realworld_test if r["label"] == 0),
        "realworld_test_ai": sum(1 for r in realworld_test if r["label"] == 1),
        "min_per_generator": args.min_per_generator,
        "n_train": len(train), "n_val": len(val), "n_realworld_test": len(realworld_test),
        "n_train_real": sum(1 for r in train if r["label"] == 0),
        "n_train_ai": sum(1 for r in train if r["label"] == 1),
        "near_dup_grouping": "phash" if phasher else "exact-hash only",
        "problems": problems, "ok": not problems,
    }
    (SPLIT_DIR / "leakage_report.json").write_text(json.dumps(leak, indent=2))

    print(json.dumps(comp, indent=2))
    print(json.dumps(leak, indent=2))
    if problems:
        raise SystemExit("LEAKAGE CHECK FAILED - see data/splits_v2/leakage_report.json")
    print("[v2-data] OK: splits path- & content-disjoint; held-out generators "
          f"{held_gen_names} absent from train and val.")


if __name__ == "__main__":
    main()
