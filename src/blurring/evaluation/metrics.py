"""Object-detection metrics from the chapter: precision at an IoU threshold, AP and mAP.

Inputs are per-image lists:
  ground_truth[i] = {"boxes": (N, 4) xyxy, "labels": (N,) 1-based class ids}
  predictions[i]  = {"boxes": (M, 4) xyxy, "scores": (M,), "labels": (M,)}
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from blurring.boxes import pairwise_iou

COCO_IOU_THRESHOLDS = tuple(np.round(np.arange(0.5, 0.96, 0.05), 2))


@dataclass
class ClassMatches:
    """Detections of one class sorted by score, each marked true/false positive."""

    scores: np.ndarray
    is_tp: np.ndarray
    num_gt: int


def match_class(
    ground_truth: list[dict], predictions: list[dict], class_id: int, iou_thresh: float, min_score: float = 0.0
) -> ClassMatches:
    """Greedy matching, highest score first; each ground-truth box can be claimed once.

    A second detection on an already-matched object is a false positive. That is how
    duplicate boxes (what NMS removes) show up as lower precision.
    """
    entries: list[tuple[float, int, np.ndarray]] = []
    gt_by_image, num_gt = {}, 0
    for i, (gt, pred) in enumerate(zip(ground_truth, predictions)):
        gt_boxes = np.asarray(gt["boxes"]).reshape(-1, 4)[np.asarray(gt["labels"]) == class_id]
        gt_by_image[i] = (gt_boxes, np.zeros(len(gt_boxes), dtype=bool))
        num_gt += len(gt_boxes)
        keep = (np.asarray(pred["labels"]) == class_id) & (np.asarray(pred["scores"]) >= min_score)
        for box, score in zip(np.asarray(pred["boxes"]).reshape(-1, 4)[keep], np.asarray(pred["scores"])[keep]):
            entries.append((float(score), i, box))

    entries.sort(key=lambda e: -e[0])
    is_tp = np.zeros(len(entries), dtype=bool)
    for k, (_, i, box) in enumerate(entries):
        gt_boxes, taken = gt_by_image[i]
        if not len(gt_boxes):
            continue
        ious = pairwise_iou(box[None], gt_boxes)[0]
        ious[taken] = -1.0
        j = int(ious.argmax())
        if ious[j] >= iou_thresh:
            taken[j] = True
            is_tp[k] = True
    scores = np.array([e[0] for e in entries], dtype=np.float32)
    return ClassMatches(scores=scores, is_tp=is_tp, num_gt=num_gt)


def average_precision(matches: ClassMatches, method: str = "all_point") -> float:
    """Area under the precision-recall curve for one class at one IoU threshold.

    ``all_point`` is the modern VOC/COCO-style interpolation; ``11_point`` is the
    VOC2008 variant the chapter cites. Returns NaN when the class has no ground truth.
    """
    if matches.num_gt == 0:
        return float("nan")
    if len(matches.is_tp) == 0:
        return 0.0
    tp = np.cumsum(matches.is_tp)
    fp = np.cumsum(~matches.is_tp)
    recall = tp / matches.num_gt
    precision = tp / np.maximum(tp + fp, 1)

    if method == "11_point":
        return float(np.mean([precision[recall >= r].max() if (recall >= r).any() else 0.0 for r in np.linspace(0, 1, 11)]))
    if method != "all_point":
        raise ValueError(f"Unknown AP method {method!r}")
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]  # precision envelope: best precision at >= this recall
    steps = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[steps + 1] - mrec[steps]) * mpre[steps + 1]))


def precision_recall_at(matches: ClassMatches) -> tuple[int, int, int]:
    """(true positives, total detections, total ground truth) for already score-filtered matches."""
    return int(matches.is_tp.sum()), len(matches.is_tp), matches.num_gt


def _nanmean(values: list[float]) -> float:
    vals = [v for v in values if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def evaluate_detections(
    ground_truth: list[dict],
    predictions: list[dict],
    classes: list[str],
    precision_iou_thresholds: list[float],
    operating_score: float,
) -> dict:
    """Full offline report: per-class AP, mAP@0.5, mAP@0.75, COCO mAP@[.5:.95], and
    precision/recall at the operating score for each of the chapter's IoU thresholds.

    Recall is reported alongside precision because for privacy a missed face (low
    recall) is worse than an extra blurred patch (low precision).
    """
    per_class: dict[str, dict[str, float]] = {}
    for class_id, name in enumerate(classes, start=1):
        aps = {t: average_precision(match_class(ground_truth, predictions, class_id, t)) for t in COCO_IOU_THRESHOLDS}
        op = match_class(ground_truth, predictions, class_id, 0.5, min_score=operating_score)
        tp, n_det, n_gt = precision_recall_at(op)
        per_class[name] = {
            "AP50": aps[0.5],
            "AP75": aps[0.75],
            "AP": _nanmean(list(aps.values())),
            "AP50_voc11": average_precision(match_class(ground_truth, predictions, class_id, 0.5), "11_point"),
            "precision@0.5": tp / n_det if n_det else float("nan"),
            "recall@0.5": tp / n_gt if n_gt else float("nan"),
            "num_gt": n_gt,
        }

    precision_by_iou = {}
    for t in precision_iou_thresholds:
        tp = n_det = n_gt = 0
        for class_id in range(1, len(classes) + 1):
            a, b, c = precision_recall_at(match_class(ground_truth, predictions, class_id, t, operating_score))
            tp, n_det, n_gt = tp + a, n_det + b, n_gt + c
        precision_by_iou[str(t)] = {
            "precision": tp / n_det if n_det else float("nan"),
            "recall": tp / n_gt if n_gt else float("nan"),
            "correct": tp,
            "detections": n_det,
            "ground_truth": n_gt,
        }

    return {
        "mAP50": _nanmean([c["AP50"] for c in per_class.values()]),
        "mAP75": _nanmean([c["AP75"] for c in per_class.values()]),
        "mAP": _nanmean([c["AP"] for c in per_class.values()]),
        "per_class": per_class,
        "at_operating_score": {"score": operating_score, "by_iou": precision_by_iou},
        "num_images": len(ground_truth),
    }
