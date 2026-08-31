"""
scripts/test_inference.py
=========================
Forensic diagnostic for the inference / live-demo pipeline. NOT part of the
benchmark. Prints, for each image, exactly what the model sees and outputs.

    python scripts/test_inference.py IMG [IMG ...] \\
        --checkpoint outputs/robust_v1/best_model.pt --image_size 64

For each image:
    filename
    original (WxH) + mode + EXIF-orientation tag if present
    preprocessed tensor shape + dtype + value range
    raw logits [real, AI]
    P(REAL)  (= softmax logit[0])
    P(AI)    (= softmax logit[1])   <- this is what the demo shows
    predicted class at threshold 0.5

Class map (from src/model.py, src/datasets.py, training): 0 = REAL, 1 = AI.
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import build_model
from src.transforms import build_eval_transform


def load_model(ckpt, device):
    m = build_model(backbone="efficientnet_b0", pretrained=False, num_classes=2)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.to(device).eval()
    return m


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="Inference-pipeline diagnostic")
    p.add_argument("images", nargs="+")
    p.add_argument("--checkpoint", default="outputs/robust_v1/best_model.pt")
    p.add_argument("--image_size", type=int, default=64,
                   help="MUST match the checkpoint's training size (64 for v1)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--exif", action="store_true",
                   help="apply EXIF orientation before preprocessing")
    args = p.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    tfm = build_eval_transform(args.image_size)

    print(f"checkpoint : {args.checkpoint}")
    print(f"image_size : {args.image_size}   device: {device}   exif-transpose: {args.exif}")
    print(f"class map  : 0 = REAL, 1 = AI-generated   threshold: P(AI) >= 0.5\n")

    for path in args.images:
        img = Image.open(path)
        orient = img.getexif().get(274, None)   # 274 = EXIF Orientation tag
        raw_mode, raw_size = img.mode, img.size
        img = img.convert("RGB")
        if args.exif:
            img = ImageOps.exif_transpose(img)

        x = tfm(img).unsqueeze(0).to(device)
        logits = model(x)[0]
        probs = torch.softmax(logits, dim=0)
        p_real, p_ai = probs[0].item(), probs[1].item()
        pred = "AI" if p_ai >= 0.5 else "REAL"

        print(f"file            : {path}")
        print(f"  original      : {raw_size[0]}x{raw_size[1]}  mode={raw_mode}"
              f"  exif_orientation={orient}")
        print(f"  tensor        : shape={tuple(x.shape)} dtype={x.dtype}"
              f"  range=[{x.min().item():.3f}, {x.max().item():.3f}]")
        print(f"  raw logits    : real={logits[0].item():+.4f}  AI={logits[1].item():+.4f}")
        print(f"  P(REAL)       : {p_real:.4f}")
        print(f"  P(AI)         : {p_ai:.4f}")
        print(f"  predicted     : {pred}\n")


if __name__ == "__main__":
    main()
