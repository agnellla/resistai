"""
scripts/shortcut_probes.py
==========================
Shortcut-regression probes. Runs on ANY set of checkpoints so V1 and V2 are
compared on the same axes. Read-only: never trains, never touches frozen artifacts.

The V1 forensic audit found the shortcut:
    smooth / low-frequency / downsampled-looking  ->  AI
    grainy / camera-textured                       ->  REAL
so an AI image collapses to REAL when you add grain or blur, and a genuine HD
photo drifts toward AI as you downsample it. These probes quantify exactly that.

Two probe sources:

  A. CIFAKE quick probe (always runs, uses data/cifake/ directly):
       - grain sweep on FAKE  sigma 0/4/8/12/16/24   (want: P(AI) stays high)
       - grain sweep on REAL  sigma 0/8/16/24        (want: P(AI) stays low)
       - blur  sweep on FAKE/REAL  radius 0/0.5/1/2
       - "resolution" sweep - NOTE CIFAKE is 32px so this is an UPSAMPLE sweep and
         only weakly informative; the real test is B.

  B. Definitive sweep on YOUR images (--real_images DIR and/or --ai_images DIR):
       - DOWNSAMPLING sweep on real HD photos: resize long side to
         1024/512/256/128/64 (BOX), upscale nothing, feed the model. If P(AI)
         climbs as the photo gets smaller, the resolution shortcut is still there.
       - GRAIN sweep on AI images: sigma 0/4/8/12/16/24. If P(AI) collapses toward
         REAL, the grain shortcut is still there.
       - GRAIN sweep on real HD photos (diagnostic).
     Desired V2 behaviour: NOT a perfectly flat curve, but substantially less
     sensitive than Baseline V1 and no dramatic AI->REAL flip.

  C. SANITY set (--images DIR): score each file once at native resolution
     (diagnostic only - never training/tuning data).

Usage:
  python scripts/shortcut_probes.py \
      --checkpoints "baseline_v1=outputs/baseline_v1/best_model.pt:64" \
                    "robust_v1=outputs/robust_v1/best_model.pt:64" \
                    "v2=outputs/v2/best_model.pt:224" \
      --real_images data/probe/real_hd --ai_images data/probe/ai \
      --images data/probe/sanity \
      --n 64 --out reports/shortcut_probes_v2.json

Each --checkpoints entry is  name=path[:image_size]  (image_size defaults to 64).
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import build_model
from src.transforms import build_eval_transform
from src.utils import get_device

CIFAKE_REAL = Path("data/cifake/real")
CIFAKE_FAKE = Path("data/cifake/fake")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

GRAIN_SIGMAS = (0, 4, 8, 12, 16, 24)
BLUR_RADII = (0, 0.5, 1.0, 2.0)
DOWNSAMPLE_LONG_SIDES = (1024, 512, 256, 128, 64)

_RNG_NP = np.random.default_rng(0)


def load_model(ckpt, device):
    m = build_model(backbone="efficientnet_b0", pretrained=False, num_classes=2)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.to(device).eval()
    return m


@torch.no_grad()
def p_ai(model, tfm, pil, device):
    x = tfm(pil.convert("RGB")).unsqueeze(0).to(device)
    return torch.softmax(model(x), dim=1)[0, 1].item()


def mean_p_ai(model, tfm, images, fn, device):
    if not images:
        return None
    return float(np.mean([p_ai(model, tfm, fn(im), device) for im in images]))


# --- perturbations -------------------------------------------------------------
def add_grain(im, sigma):
    if sigma == 0:
        return im
    a = np.asarray(im.convert("RGB"), np.float32) + _RNG_NP.normal(0, sigma, (im.size[1], im.size[0], 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def gblur(im, r):
    return im if r == 0 else im.convert("RGB").filter(ImageFilter.GaussianBlur(r))


def downsample_long_side(im, target):
    """Resize DOWN so the long side == target (BOX / area filter). Never upsamples."""
    w, h = im.size
    if max(w, h) <= target:
        return im
    s = target / max(w, h)
    return im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BOX)


def _load_dir(path, n, seed):
    files = sorted(str(p) for p in Path(path).rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise SystemExit(f"no images under {path!r}")
    random.Random(seed).shuffle(files)
    files = files[:n]
    return [Image.open(f).convert("RGB") for f in files], files


def sweep_block(models, imgs, device, *, grain=True, blur=True, downsample=False):
    """Return {model_name: {sweep_name: {step: mean_p_ai}}} for one image list."""
    out = {}
    for name, (model, tfm, _) in models.items():
        b = {"clean_mean_p_ai": mean_p_ai(model, tfm, imgs, lambda im: im, device)}
        if grain:
            b["grain_sweep"] = {f"sigma_{s}": mean_p_ai(model, tfm, imgs,
                                lambda im, s=s: add_grain(im, s), device) for s in GRAIN_SIGMAS}
            g = b["grain_sweep"]
            b["grain_p_ai_drop_0_to_24"] = (g["sigma_0"] - g["sigma_24"]
                                            if g["sigma_0"] is not None else None)
        if blur:
            b["blur_sweep"] = {f"radius_{r}": mean_p_ai(model, tfm, imgs,
                               lambda im, r=r: gblur(im, r), device) for r in BLUR_RADII}
        if downsample:
            b["downsample_sweep"] = {f"long_side_{t}": mean_p_ai(model, tfm, imgs,
                                     lambda im, t=t: downsample_long_side(im, t), device)
                                     for t in DOWNSAMPLE_LONG_SIDES}
            d = b["downsample_sweep"]
            keys = [f"long_side_{t}" for t in DOWNSAMPLE_LONG_SIDES]
            if d[keys[0]] is not None:
                b["downsample_p_ai_rise_1024_to_64"] = d[keys[-1]] - d[keys[0]]
        out[name] = b
    return out


def main():
    ap = argparse.ArgumentParser(description="Shortcut-regression probes (read-only)",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--checkpoints", nargs="+", required=True, help="name=path[:image_size] entries")
    ap.add_argument("--n", type=int, default=48, help="images per class per probe")
    ap.add_argument("--real_images", default=None, help="dir of genuine HD photos for the definitive sweep")
    ap.add_argument("--ai_images", default=None, help="dir of AI images for the definitive grain sweep")
    ap.add_argument("--images", default=None, help="dir for the one-shot sanity set")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--out", default="reports/shortcut_probes.json")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    device = get_device(args.device)
    models = {}
    for entry in args.checkpoints:
        name, rest = entry.split("=", 1)
        path, _, size = rest.partition(":")
        size = int(size) if size else 64
        if not Path(path).exists():
            print(f"[probe] SKIP {name}: {path} not found")
            continue
        models[name] = (load_model(path, device), build_eval_transform(size), size)
        print(f"[probe] loaded {name}  ({path}, image_size {size})  device={device}")
    if not models:
        sys.exit("no checkpoints loaded")

    results = {"n_per_class": args.n, "seed": args.seed, "device": str(device), "models": {}}

    # ---- A. CIFAKE quick probe ------------------------------------------------
    rng = random.Random(args.seed)
    reals = [Image.open(CIFAKE_REAL / f).convert("RGB")
             for f in rng.sample(os.listdir(CIFAKE_REAL), args.n)]
    fakes = [Image.open(CIFAKE_FAKE / f).convert("RGB")
             for f in rng.sample(os.listdir(CIFAKE_FAKE), args.n)]
    print(f"[probe] A: {len(reals)} CIFAKE real + {len(fakes)} CIFAKE fake "
          f"(from data/cifake/, NOT from any split manifest)\n")
    cif_fake = sweep_block(models, fakes, device, grain=True, blur=True, downsample=False)
    cif_real = sweep_block(models, reals, device, grain=True, blur=True, downsample=False)
    for name in models:
        results["models"][name] = {
            "image_size": models[name][2],
            "cifake_quick_probe": {
                "note": "CIFAKE is 32px; informative for grain/blur, NOT for downsampling.",
                "fake": cif_fake[name], "real": cif_real[name],
            }
        }
        f = cif_fake[name]
        print(f"--- {name} (image_size {models[name][2]}) : CIFAKE quick probe ---")
        print(f"  clean  real P(AI) {cif_real[name]['clean_mean_p_ai']:.3f} | "
              f"fake P(AI) {f['clean_mean_p_ai']:.3f}")
        print("  grain FAKE : " + "  ".join(f"s{k.split('_')[1]}={v:.3f}"
                                            for k, v in f["grain_sweep"].items())
              + f"   (drop {f['grain_p_ai_drop_0_to_24']:+.3f})")
        print("  blur  FAKE : " + "  ".join(f"r{k.split('_')[1]}={v:.3f}"
                                            for k, v in f["blur_sweep"].items()))
        print()

    # ---- B. definitive sweep on user images --------------------------------
    if args.real_images:
        real_hd, real_files = _load_dir(args.real_images, args.n, args.seed)
        print(f"[probe] B: {len(real_hd)} real HD photos from {args.real_images}")
        blk = sweep_block(models, real_hd, device, grain=True, blur=False, downsample=True)
        for name in models:
            results["models"][name]["real_hd_downsample_and_grain"] = blk[name]
            d = blk[name]["downsample_sweep"]
            print(f"--- {name} : real HD photo sweep ---")
            print("  downsample P(AI): " + "  ".join(f"{k.split('_')[-1]}px={v:.3f}"
                                                     for k, v in d.items())
                  + f"   (rise 1024->64 {blk[name].get('downsample_p_ai_rise_1024_to_64', float('nan')):+.3f})")
            g = blk[name]["grain_sweep"]
            print("  grain      P(AI): " + "  ".join(f"s{k.split('_')[1]}={v:.3f}"
                                                    for k, v in g.items()))
            print()
        results["real_hd_files"] = real_files

    if args.ai_images:
        ai_imgs, ai_files = _load_dir(args.ai_images, args.n, args.seed + 1)
        print(f"[probe] B: {len(ai_imgs)} AI images from {args.ai_images}")
        blk = sweep_block(models, ai_imgs, device, grain=True, blur=True, downsample=True)
        for name in models:
            results["models"][name]["ai_images_grain_blur_downsample"] = blk[name]
            g = blk[name]["grain_sweep"]
            print(f"--- {name} : AI image sweep ---")
            print("  grain P(AI): " + "  ".join(f"s{k.split('_')[1]}={v:.3f}"
                                                for k, v in g.items())
                  + f"   (drop 0->24 {blk[name]['grain_p_ai_drop_0_to_24']:+.3f})")
            b = blk[name]["blur_sweep"]
            print("  blur  P(AI): " + "  ".join(f"r{k.split('_')[1]}={v:.3f}"
                                                for k, v in b.items()))
            print()
        results["ai_files"] = ai_files

    if not args.real_images and not args.ai_images:
        results["note_definitive"] = ("no --real_images / --ai_images given: only the CIFAKE "
                                      "quick probe ran. For the definitive downsampling/grain "
                                      "curves, pass folders of genuine HD photos and AI images.")

    # ---- C. sanity set ---------------------------------------------------
    if args.images and Path(args.images).is_dir():
        files = sorted(str(p) for p in Path(args.images).rglob("*") if p.suffix.lower() in IMG_EXTS)
        san = []
        for fpath in files:
            row = {"file": fpath}
            try:
                im = Image.open(fpath).convert("RGB")
                row["orig_size"] = list(im.size)
                for name, (model, tfm, _) in models.items():
                    row[name] = round(p_ai(model, tfm, im, device), 4)
            except Exception as e:
                row["error"] = str(e)
            san.append(row)
        results["sanity_set"] = san
        print("=== sanity set (diagnostic only) ===")
        for r in san:
            print("  ", r)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"[probe] wrote {args.out}")


if __name__ == "__main__":
    main()
