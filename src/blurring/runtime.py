"""Cross-cutting runtime helpers for training/inference with torch: reproducibility and device choice."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed every RNG the pipeline touches so splits, augmentation and init are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name: str = "auto", mps_memory_fraction: float = 0.0) -> torch.device:
    """Pick CUDA, then Apple MPS, then CPU. MPS falls back to CPU for any op it lacks.

    On Apple Silicon the GPU shares system RAM, so ``mps_memory_fraction`` caps how much
    the MPS allocator may hold; past the cap PyTorch raises OOM instead of starving the OS.
    """
    if mps_memory_fraction > 0:
        # must be set before the first MPS allocation; PyTorch requires low <= high watermark
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", str(mps_memory_fraction))
        os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", str(mps_memory_fraction))
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    return torch.device("cpu")
