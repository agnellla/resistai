"""
src/acquisition.py
==================
V2 acquisition / domain-randomisation augmentation.

The V1 forensic audit found a shortcut:
    camera-like grain / high-frequency micro-texture  ->  REAL
    smooth / clean / downsampled                      ->  AI
i.e. the model used *acquisition characteristics* (grain, JPEG history,
downsampling footprint, sharpness) as if they were the class label.

This module applies acquisition transforms to **BOTH REAL and AI images**
during training, so that:
    grain != label,  JPEG quality != label,
    downsampling history != label,  blur != label.
After this, the model can only separate the classes using something that
survives the randomisation - ideally generator-artifact evidence.

This is a SEPARATE augmentation group from the robustness transforms in
src/augmentations.py (which stay unchanged for the frozen CIFAKE B benchmark).

PIL RGB image in -> PIL RGB image out. Deterministic given the seed passed to
AcquisitionAug (per-DataLoader-worker reseed, same pattern as RandomTransform).
"""

import io
import random

import numpy as np
from PIL import Image, ImageFilter

_INTERP = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS, Image.BOX, Image.HAMMING]


def a_recompress(img, rng):
    """Random JPEG re-encode, quality 30-95 (applied to real AND ai)."""
    q = rng.randint(30, 95)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def a_downup(img, rng):
    """
    Random downscale to a random fraction with a random interpolation kernel,
    then back up with a (possibly different) random kernel. Randomises the
    'downsampling history' cue so it stops predicting the label.
    """
    w, h = img.size
    frac = rng.uniform(0.25, 0.9)
    # keep the intermediate at least 8px on the short side so tiny inputs
    # (e.g. the 32px CIFAKE mix-in) are randomised, not obliterated
    min_frac = 8.0 / max(1, min(w, h))
    frac = max(frac, min(min_frac, 1.0))
    k_down = rng.choice(_INTERP)
    k_up = rng.choice(_INTERP)
    small = img.resize((max(1, int(w * frac)), max(1, int(h * frac))), k_down)
    return small.resize((w, h), k_up)


def a_grain(img, rng):
    """Mild-to-moderate synthetic grain, sigma 2-14 in 0-255 units (real AND ai)."""
    sigma = rng.uniform(2.0, 14.0)
    npy = np.random.default_rng(rng.getrandbits(32))
    arr = np.asarray(img.convert("RGB"), np.float32) + npy.normal(0.0, sigma, (img.size[1], img.size[0], 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def a_blur(img, rng):
    """Mild Gaussian blur, radius 0.3-1.2 (real AND ai)."""
    return img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.2)))


def a_sharpen_or_soften(img, rng):
    """Small unsharp / smoothing variation so 'sharpness' isn't class-predictive."""
    if rng.random() < 0.5:
        return img.filter(ImageFilter.UnsharpMask(radius=rng.uniform(0.5, 1.5),
                                                  percent=rng.randint(40, 120)))
    return img.filter(ImageFilter.SMOOTH_MORE)


ACQUISITION_TRANSFORMS = {
    "recompress": a_recompress,
    "downup": a_downup,
    "grain": a_grain,
    "blur": a_blur,
    "sharpness": a_sharpen_or_soften,
}
ALL_ACQ_NAMES = list(ACQUISITION_TRANSFORMS)


def describe_acquisition():
    return {
        "recompress": "JPEG re-encode, quality uniform[30,95]",
        "downup": "downscale to frac uniform[0.25,0.9] with random kernel, upscale back with random kernel",
        "grain": "additive Gaussian grain, sigma uniform[2,14] of 255",
        "blur": "Gaussian blur, radius uniform[0.3,1.2]",
        "sharpness": "random UnsharpMask or SMOOTH_MORE",
        "kernels": ["nearest", "bilinear", "bicubic", "lanczos", "box", "hamming"],
    }


class AcquisitionAug:
    """
    With probability `prob`, apply `num` randomly chosen acquisition transforms
    (chained). Applied to EVERY class equally. Label never touched.

    prob default 0.6, num default 2  (spec: ~50-70%, don't destroy every image).
    """

    def __init__(self, prob=0.6, names=None, num=2, seed=42):
        self.prob = float(prob)
        self.names = list(names) if names else list(ALL_ACQ_NAMES)
        bad = [n for n in self.names if n not in ACQUISITION_TRANSFORMS]
        if bad:
            raise KeyError(f"unknown acquisition transform(s) {bad}")
        self.num = max(1, int(num))
        self._rng = random.Random(seed)

    def reseed(self, seed):
        self._rng = random.Random(seed)

    def __call__(self, img):
        if self._rng.random() >= self.prob:
            return img
        k = min(self.num, len(self.names))
        for name in self._rng.sample(self.names, k=k):
            img = ACQUISITION_TRANSFORMS[name](img, self._rng)
        return img
