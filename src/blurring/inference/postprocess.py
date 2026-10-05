"""Turn raw detector outputs into final detections. Pure NumPy, shared by the torch and ONNX backends."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from blurring.inference.nms import batched_nms


@dataclass
class Detections:
    boxes: np.ndarray  # (N, 4) xyxy
    scores: np.ndarray  # (N,)
    labels: list[str]


def postprocess(raw: dict[str, np.ndarray], classes: list[str], score_thresh: float, nms_iou: float) -> Detections:
    """Drop low-confidence boxes, then run our own NMS.

    The detector already applies per-class NMS internally at a looser IoU; this pass
    uses the serving threshold so the blurring step gets one box per object.
    """
    keep = raw["scores"] >= score_thresh
    boxes, scores, labels = raw["boxes"][keep], raw["scores"][keep], raw["labels"][keep]
    order = batched_nms(boxes, scores, labels, nms_iou)
    return Detections(boxes[order], scores[order], [classes[int(l) - 1] for l in labels[order]])
