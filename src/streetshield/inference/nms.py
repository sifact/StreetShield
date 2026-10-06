"""Non-maximum suppression: collapse the RPN's many overlapping boxes into one box per object (chapter Fig. 3.11)."""

from __future__ import annotations

import numpy as np

from streetshield.boxes import pairwise_iou


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    """Keep the highest-scoring box, drop every box overlapping it by more than ``iou_thresh``, repeat.

    Returns kept indices, highest score first.
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    order = np.argsort(-np.asarray(scores), kind="stable")
    keep = []
    while order.size:
        best = order[0]
        keep.append(best)
        if order.size == 1:
            break
        ious = pairwise_iou(boxes[best][None], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thresh]
    return np.asarray(keep, dtype=np.int64)


def batched_nms(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, iou_thresh: float) -> np.ndarray:
    """NMS run independently per class, so a face box never suppresses an overlapping plate box."""
    labels = np.asarray(labels)
    keep = [np.flatnonzero(labels == c)[nms(boxes[labels == c], scores[labels == c], iou_thresh)] for c in np.unique(labels)]
    if not keep:
        return np.zeros(0, dtype=np.int64)
    keep = np.concatenate(keep)
    return keep[np.argsort(-np.asarray(scores)[keep], kind="stable")]
