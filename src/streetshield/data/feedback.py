"""Data pipeline for retraining: user reports and hard negative mining (chapter Fig. 3.12).

Two signals come back from production:
  * user reports: an annotator corrects an image that was not blurred properly, giving
    the true boxes. These become new labelled training records.
  * hard examples: images where the current model fires confidently on background
    (false positives) or misses an object (false negatives). They are oversampled in
    the next training round so the model sees its own mistakes more often.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from streetshield.boxes import pairwise_iou, xywh_to_xyxy
from streetshield.data.annotations import Record
from streetshield.data.dataset import load_rgb


def load_reports(path: Path) -> list[Record]:
    """Read annotator-verified reports from JSONL.

    Each line: ``{"image_path": ..., "boxes": [[x, y, w, h], ...], "labels": ["face", ...]}``
    using the chapter's ``[x, y, w, h]`` format.
    """
    records = []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            item = json.loads(line)
            width, height = load_rgb(item["image_path"]).size
            boxes = xywh_to_xyxy(np.asarray(item.get("boxes", []), dtype=np.float32)).tolist()
            records.append(
                Record(item["image_path"], width, height, boxes, list(item.get("labels", [])), source="user_report")
            )
    return records


def find_hard_examples(
    records: list[Record],
    predictions: list[dict[str, np.ndarray]],
    classes: list[str],
    score_thresh: float,
    match_iou: float,
) -> list[int]:
    """Indices of records where the model has a confident false positive or any missed object.

    ``predictions[i]`` holds ``boxes`` (xyxy), ``scores`` and ``labels`` (1-based class ids).
    """
    hard = []
    for i, (record, pred) in enumerate(zip(records, predictions)):
        keep = pred["scores"] >= score_thresh
        pred_boxes, pred_labels = pred["boxes"][keep], pred["labels"][keep]
        gt_boxes = np.asarray(record.boxes, dtype=np.float32).reshape(-1, 4)
        gt_labels = np.asarray([classes.index(l) + 1 for l in record.labels], dtype=np.int64)

        iou = pairwise_iou(pred_boxes, gt_boxes)
        same_class = pred_labels[:, None] == gt_labels[None, :]
        matched = (iou >= match_iou) & same_class
        has_false_positive = bool(len(pred_boxes)) and not matched.any(axis=1).all()
        has_miss = bool(len(gt_boxes)) and not matched.any(axis=0).all()
        if has_false_positive or has_miss:
            hard.append(i)
    return hard


def build_retraining_set(
    train: list[Record], reports: list[Record], hard_examples: list[Record], oversample: int
) -> list[Record]:
    """New training split = original + corrected reports + extra copies of hard examples."""
    return train + reports + [r for r in hard_examples for _ in range(oversample)]
