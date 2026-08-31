"""
src/transforms.py
-----------------
All image transformations live here.

Two jobs:
1. Give the "normalisation" pipeline that turns a PIL image into a tensor the
   model expects (resize -> tensor -> ImageNet mean/std).
2. Give the six real-world corruptions the competition tests robustness against:
       jpeg, blur, resize_updown, noise, color_jitter, center_crop
   plus "clean" (no corruption) as the reference point.

The same corruption functions are used in two places:
  - src/evaluate.py -> to measure robustness
  - the (later) transformation-aware training run -> to degrade training images

The current baseline train.py does NOT use these corruptions.

Everything works on PIL.Image in, PIL.Image out, so corruptions can be chained
and stay easy to eyeball.
"""

import io

import numpy as np
import torchvision.transforms as T
from PIL import Image, ImageFilter

# ImageNet statistics - EfficientNet-B0 was pretrained with these.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# 1. Model input pipeline (PIL image -> normalised tensor)
# ---------------------------------------------------------------------------
def build_eval_transform(image_size):
    """Deterministic pipeline used at validation / inference time."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_train_transform(image_size):
    """
    Training pipeline. Light, standard augmentation only.
    The heavier robustness corruptions are applied separately (see
    apply_random_corruption) when train_aware is turned on in the config.
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# 2. Real-world corruptions  (PIL image -> PIL image)
# ---------------------------------------------------------------------------
def corrupt_clean(img):
    """No change. The reference point for robustness comparisons."""
    return img


def corrupt_jpeg(img, quality=35):
    """Re-encode as JPEG at low quality to add compression artefacts."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def corrupt_blur(img, radius=2.0):
    """Gaussian blur - simulates soft focus / downscaled sources."""
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def corrupt_resize_updown(img, scale=0.5):
    """Shrink the image then blow it back up - destroys fine detail."""
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def corrupt_noise(img, sigma=15.0):
    """Add Gaussian pixel noise (sigma in 0-255 units)."""
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    arr += np.random.normal(0.0, sigma, arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def corrupt_color_jitter(img):
    """Shift brightness / contrast / saturation / hue a little."""
    jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
    return jitter(img)


def corrupt_center_crop(img, keep=0.8):
    """Crop the middle `keep` fraction, then resize back to original size."""
    w, h = img.size
    cw, ch = int(w * keep), int(h * keep)
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BILINEAR)


# Name -> function lookup. Keys must match configs/default.yaml -> eval.transforms.
CORRUPTIONS = {
    "clean": corrupt_clean,
    "jpeg": corrupt_jpeg,
    "blur": corrupt_blur,
    "resize_updown": corrupt_resize_updown,
    "noise": corrupt_noise,
    "color_jitter": corrupt_color_jitter,
    "center_crop": corrupt_center_crop,
}


def apply_corruption(img, name):
    """Apply one named corruption to a PIL image."""
    if name not in CORRUPTIONS:
        raise KeyError(f"unknown corruption '{name}', choose from {list(CORRUPTIONS)}")
    return CORRUPTIONS[name](img)


def apply_random_corruption(img, p=0.5):
    """
    Used by transformation-aware training: with probability `p` pick one random
    corruption (never 'clean') and apply it, so the model sees degraded images
    while it learns.
    """
    if np.random.rand() > p:
        return img
    name = np.random.choice([k for k in CORRUPTIONS if k != "clean"])
    return apply_corruption(img, name)
