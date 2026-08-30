"""Small shared helpers."""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fix random seeds across python/numpy/torch.

    Non-negotiable for this project (see CLAUDE.md): without this, the
    train/val split behavior, weight init for the new stem+head, and
    augmentation order all vary run to run, on top of whatever timing
    variance the benchmarking harness already has to contend with. Fixing
    seeds removes one axis of noise so differences we measure are
    attributable to the thing we're actually testing (quantization).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
