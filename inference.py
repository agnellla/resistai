"""
inference.py  (competition deliverable)
---------------------------------------
Take a directory of images, classify each one, write a JSON file.

Output format (one entry per image):

    [
      {"image_path": "test_images/img_001.jpg", "pred": 1},
      {"image_path": "test_images/img_002.png", "pred": 0},
      ...
    ]

    pred = 0  -> real / authentic
    pred = 1  -> AI-generated

Run:
    python inference.py --images path/to/dir --checkpoint checkpoints/model.pt --out predictions.json

This script is intentionally standalone and dependency-light so the judges can
run it without touching the rest of the repo.
"""

import argparse
import json
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image, ImageOps

from src.model import build_model
from src.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.utils import get_device, load_config

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder):
    """Every image file under `folder`, recursively, sorted for stable output."""
    folder = Path(folder)
    return sorted(str(p) for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)


def build_transform(image_size):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description="ResistAI inference -> JSON (real=0, AI-generated=1). CUDA / MPS / CPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--images", required=True, help="directory of images to classify")
    parser.add_argument("--checkpoint", default="checkpoints/model.pt",
                        help="path to a trained state_dict checkpoint")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="YAML config for model settings (backbone, image size, ...)")
    parser.add_argument("--image_size", type=int, default=None,
                        help="override config image size; MUST match the "
                             "checkpoint's training size (64 for Baseline/Robust v1)")
    parser.add_argument("--out", default="predictions.json",
                        help="output JSON file path")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
                        help="compute device; 'auto' picks CUDA > MPS > CPU")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(args.device)
    image_size = args.image_size if args.image_size is not None else cfg["data"]["image_size"]
    print(f"[inference] image_size = {image_size}  (must match the checkpoint's training size)")

    # Build the model and load trained weights.
    model = build_model(
        backbone=cfg["model"]["backbone"],
        pretrained=False,
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
    )
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    tfm = build_transform(image_size)
    paths = list_images(args.images)
    print(f"[inference] {len(paths)} images found in {args.images}")

    results = []
    for path in paths:
        try:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img)   # respect camera rotation
            img = img.convert("RGB")
        except Exception as e:
            # Unreadable file -> default to real (0) and keep going.
            print(f"[inference] could not read {path}: {e}")
            results.append({"image_path": path, "pred": 0})
            continue

        x = tfm(img).unsqueeze(0).to(device)
        prob_ai = torch.softmax(model(x), dim=1)[0, 1].item()
        pred = int(prob_ai >= 0.5)
        results.append({"image_path": path, "pred": pred})

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[inference] wrote {len(results)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
