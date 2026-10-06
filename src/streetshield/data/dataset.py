"""PyTorch datasets: annotated records for training/eval, and raw image paths for batch blurring."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from streetshield.data.annotations import Record

Transform = Callable[[Image.Image, np.ndarray], tuple[torch.Tensor, np.ndarray]]


def load_rgb(path: str | Path) -> Image.Image:
    """Some source PNGs are RGBA or palette images; the detector needs 3 channels."""
    with Image.open(path) as im:
        return im.convert("RGB")


class DetectionDataset(Dataset):
    """Yields ``(image_tensor, target)`` in the format torchvision detectors train on."""

    def __init__(self, records: list[Record], classes: list[str], transform: Transform):
        self.records = records
        # torchvision reserves label 0 for background, so our classes start at 1
        self.class_to_idx = {name: i + 1 for i, name in enumerate(classes)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        record = self.records[idx]
        image = load_rgb(record.image_path)
        boxes = np.asarray(record.boxes, dtype=np.float32).reshape(-1, 4)
        tensor, boxes = self.transform(image, boxes)
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor([self.class_to_idx[l] for l in record.labels], dtype=torch.int64),
            "image_id": torch.tensor(idx),
        }
        return tensor, target


class ImagePathDataset(Dataset):
    """CPU-side preprocessing for the batch blurring pipeline: decode, keep the original for blurring."""

    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, np.ndarray, str]:
        image = load_rgb(self.paths[idx])
        return F.to_tensor(image), np.asarray(image), str(self.paths[idx])


def collate_detection(batch: list) -> tuple:
    """Detection images differ in size, so keep them as tuples instead of stacking."""
    return tuple(zip(*batch))
