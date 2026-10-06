"""Box geometry shared by data, evaluation and inference.

Internally every box is ``[x1, y1, x2, y2]`` (what torchvision expects). The chapter's
``[x, y, w, h]`` format is only used at the edges (reports in, detections out).
"""

from __future__ import annotations

import numpy as np


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return np.concatenate([boxes[:, :2], boxes[:, :2] + boxes[:, 2:]], axis=1)


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return np.concatenate([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]], axis=1)


def box_area(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)


def pairwise_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix of shape (len(a), len(b)): intersection area / union area (chapter Fig. 3.9)."""
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(bottom_right - top_left, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def pad_and_clip(boxes: np.ndarray, padding: float, width: int, height: int) -> np.ndarray:
    """Grow each box by ``padding`` of its size on every side, then clip to the image."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    wh = boxes[:, 2:] - boxes[:, :2]
    boxes[:, :2] -= wh * padding
    boxes[:, 2:] += wh * padding
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    return boxes
