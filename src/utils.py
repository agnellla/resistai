"""
src/utils.py
------------
Small helper functions shared by the other scripts.
Nothing clever here on purpose - just the plumbing every ML project needs.
"""

import os
import random

import numpy as np
import torch
import yaml


def load_config(path="configs/default.yaml"):
    """Read the YAML config file into a plain Python dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    """
    Fix every random source we use so two runs give the same result.
    Makes debugging and 'baseline vs robust' comparisons fair.

    All the torch calls below are safe to run even when that backend is
    missing - they are no-ops when there is no CUDA / MPS device.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)              # no-op without CUDA
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)              # no-op without Apple MPS


def _mps_available():
    """True only on Apple Silicon with a working MPS build of torch."""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def get_device(prefer=None):
    """
    Return the torch device to run on.

    prefer:
        None / "auto" -> pick the fastest available: CUDA > MPS > CPU
        "cuda"        -> NVIDIA GPU (e.g. a Colab Tesla T4)
        "mps"         -> Apple Silicon GPU (M1/M2/M3 Macs)
        "cpu"         -> force CPU (works everywhere, slow)

    If you ask for a device that is not present, we print a warning and fall
    back to auto-detection instead of crashing. Nothing here assumes CUDA.
    """
    choice = (prefer or "auto").lower()

    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[utils] --device cuda requested but no CUDA GPU found - auto-detecting instead")
    elif choice == "mps":
        if _mps_available():
            return torch.device("mps")
        print("[utils] --device mps requested but MPS not available - auto-detecting instead")
    elif choice not in ("auto", ""):
        print(f"[utils] unknown --device '{prefer}' - auto-detecting instead")

    # Auto-detect.
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(model, path):
    """Save model weights to disk. Creates the folder if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[utils] saved checkpoint -> {path}")


def load_checkpoint(model, path, device):
    """
    Load weights saved by save_checkpoint back into a model object.

    `map_location=device` is the important bit: a checkpoint trained on a CUDA
    Tesla T4 in Colab loads fine on a Mac (MPS) or on CPU, and vice versa,
    because the tensors are remapped to `device` as they are read.
    """
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"[utils] loaded checkpoint <- {path}  (mapped to {device})")
    return model
