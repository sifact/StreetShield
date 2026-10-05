"""Box-aware image augmentation: every geometric change to the image is applied to its boxes too."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as F

from blurring.config import AugmentConfig


def _rand() -> float:
    # torch RNG (not Python's) so each DataLoader worker gets its own seeded stream
    return float(torch.rand(()))


def hflip(image: Image.Image, boxes: np.ndarray) -> tuple[Image.Image, np.ndarray]:
    width = image.width
    boxes = boxes.copy()
    boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
    return F.hflip(image), boxes


def crop_keeping_boxes(image: Image.Image, boxes: np.ndarray) -> tuple[Image.Image, np.ndarray]:
    """Random crop whose window still contains every box.

    Cutting a face in half would teach the detector that half-faces are background,
    which is the worst failure for a privacy system, so we never crop through a box.
    """
    w, h = image.size
    if len(boxes):
        min_x, min_y = boxes[:, 0].min(), boxes[:, 1].min()
        max_x, max_y = boxes[:, 2].max(), boxes[:, 3].max()
    else:
        min_x, min_y, max_x, max_y = w / 2, h / 2, w / 2, h / 2
    left = int(_rand() * min_x)
    top = int(_rand() * min_y)
    right = int(max_x + _rand() * (w - max_x))
    bottom = int(max_y + _rand() * (h - max_y))
    if right - left < 32 or bottom - top < 32:
        return image, boxes
    boxes = boxes.copy()
    boxes[:, [0, 2]] -= left
    boxes[:, [1, 3]] -= top
    return image.crop((left, top, right, bottom)), boxes


class TrainTransform:
    """Online augmentation for training: colour jitter, box-safe crop, horizontal flip."""

    def __init__(self, cfg: AugmentConfig):
        self.cfg = cfg
        self.jitter = ColorJitter(cfg.brightness, cfg.contrast, cfg.saturation)

    def __call__(self, image: Image.Image, boxes: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        image = self.jitter(image)
        if _rand() < self.cfg.crop_prob:
            image, boxes = crop_keeping_boxes(image, boxes)
        if _rand() < self.cfg.hflip_prob:
            image, boxes = hflip(image, boxes)
        return F.to_tensor(image), boxes


class EvalTransform:
    """No augmentation. Resizing and normalisation happen inside the detector itself."""

    def __call__(self, image: Image.Image, boxes: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        return F.to_tensor(image), boxes
