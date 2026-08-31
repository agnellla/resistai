"""
src/augmentations.py
--------------------
Transformation-aware training augmentations.

Goal: during training, randomly degrade some images the way real-world
post-processing would (JPEG re-save, blur, downscale, noise, colour shift,
crop) so the model learns features that survive those operations. It is used
ONLY for training - validation and test data stay clean.

Design (matches the hackathon spec):
  - all transforms run ON THE FLY (PIL image in -> PIL image out); nothing is
    written to disk
  - each training image has probability `prob` of being transformed at all
  - when transformed, ONE transform is picked at random (configurable count)
  - the label never changes - these are appearance changes, not class changes

The transform functions take (img, rng) where `rng` is a random.Random so the
parameter choice is reproducible from the training seed.
"""

import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# ---------------------------------------------------------------------------
# Parameter banks (exact values from the hackathon transformation list)
# ---------------------------------------------------------------------------
JPEG_QUALITIES = [90, 70, 50, 30]          # 1. JPEG compression
BLUR_SIGMAS = [0.5, 1.0, 2.0]              # 2. Gaussian blur (radius = sigma)
RESIZE_SCALES = [0.5, 0.25]               # 3. resize down then back up
NOISE_SIGMAS = [0.02, 0.05, 0.10]         # 4. Gaussian noise (fraction of 255)
COLOR_JITTER = 0.20                        # 5. brightness/contrast/saturation +/-20%
CROP_FRACTION = 0.80                       # 6. centre-crop keep 80%, then resize back


# ---------------------------------------------------------------------------
# Individual transforms  (PIL RGB image -> PIL RGB image)
# ---------------------------------------------------------------------------
def t_jpeg(img, rng):
    """Re-encode as JPEG at a random low quality -> compression artefacts."""
    quality = int(rng.choice(JPEG_QUALITIES))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def t_blur(img, rng):
    """Gaussian blur with a random sigma."""
    sigma = float(rng.choice(BLUR_SIGMAS))
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def t_resize(img, rng):
    """Shrink to `scale` then upscale back to the original size (detail loss)."""
    scale = float(rng.choice(RESIZE_SCALES))
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def t_noise(img, rng):
    """Add Gaussian pixel noise; sigma is a fraction of the 0-255 range."""
    sigma = float(rng.choice(NOISE_SIGMAS)) * 255.0
    npy_rng = np.random.default_rng(rng.getrandbits(32))
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    arr += npy_rng.normal(0.0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def t_color_jitter(img, rng):
    """Multiply brightness, then contrast, then saturation by random +/-20% factors."""
    for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        factor = 1.0 + rng.uniform(-COLOR_JITTER, COLOR_JITTER)
        img = enhancer(img).enhance(factor)
    return img


def t_center_crop(img, rng):
    """Centre-crop to CROP_FRACTION of each side, then resize back to original size."""
    w, h = img.size
    cw, ch = int(w * CROP_FRACTION), int(h * CROP_FRACTION)
    left, top = (w - cw) // 2, (h - ch) // 2
    cropped = img.crop((left, top, left + cw, top + ch))
    return cropped.resize((w, h), Image.BILINEAR)


# name -> function. Names are what you pass on the command line.
TRANSFORMS = {
    "jpeg": t_jpeg,
    "blur": t_blur,
    "resize": t_resize,
    "noise": t_noise,
    "color_jitter": t_color_jitter,
    "center_crop": t_center_crop,
}
ALL_TRANSFORM_NAMES = list(TRANSFORMS)


def describe_params():
    """Return the parameter banks as a plain dict (for run_config.json)."""
    return {
        "jpeg_qualities": JPEG_QUALITIES,
        "blur_sigmas": BLUR_SIGMAS,
        "resize_scales": RESIZE_SCALES,
        "noise_sigmas_frac255": NOISE_SIGMAS,
        "color_jitter_range": COLOR_JITTER,
        "center_crop_fraction": CROP_FRACTION,
    }


# ---------------------------------------------------------------------------
# The callable used by the training Dataset
# ---------------------------------------------------------------------------
class RandomTransform:
    """
    With probability `prob`, apply `num` randomly chosen transform(s) to a PIL
    image. Otherwise return it untouched. The label is handled by the Dataset
    and is never passed in here - it cannot change.

    Args:
        prob:  probability that an image gets transformed at all (e.g. 0.7)
        names: which transforms are allowed (default: all six)
        num:   how many transforms to chain when an image IS transformed
               (spec: start with 1)
        seed:  base seed for the parameter RNG (reproducible)
    """

    def __init__(self, prob=0.7, names=None, num=1, seed=42):
        self.prob = float(prob)
        self.names = list(names) if names else list(ALL_TRANSFORM_NAMES)
        unknown = [n for n in self.names if n not in TRANSFORMS]
        if unknown:
            raise KeyError(f"unknown transform(s) {unknown}; choose from {ALL_TRANSFORM_NAMES}")
        self.num = max(1, int(num))
        # Instance RNG so behaviour is reproducible; DataLoader workers reseed
        # it via worker_init_fn (see src/datasets.py) so workers differ.
        import random
        self._rng = random.Random(seed)

    def reseed(self, seed):
        import random
        self._rng = random.Random(seed)

    def __call__(self, img):
        if self._rng.random() >= self.prob:
            return img
        k = min(self.num, len(self.names))
        for name in self._rng.sample(self.names, k=k):
            img = TRANSFORMS[name](img, self._rng)
        return img
