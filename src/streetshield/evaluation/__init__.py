"""Evaluation stage: IoU matching, precision, AP and mAP."""

from streetshield.evaluation.evaluator import evaluate_model, ground_truth_from_records, predict_records
from streetshield.evaluation.metrics import (
    COCO_IOU_THRESHOLDS,
    average_precision,
    evaluate_detections,
    match_class,
)

__all__ = [
    "COCO_IOU_THRESHOLDS",
    "average_precision",
    "evaluate_detections",
    "evaluate_model",
    "ground_truth_from_records",
    "match_class",
    "predict_records",
]
