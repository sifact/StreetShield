"""Run a detector over annotated records and score it with the offline metrics."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from blurring.config import EvalConfig
from blurring.data import DetectionDataset, EvalTransform, Record, collate_detection
from blurring.evaluation.metrics import evaluate_detections


def ground_truth_from_records(records: list[Record], classes: list[str]) -> list[dict]:
    idx = {name: i + 1 for i, name in enumerate(classes)}
    return [
        {"boxes": np.asarray(r.boxes, dtype=np.float32).reshape(-1, 4), "labels": np.asarray([idx[l] for l in r.labels])}
        for r in records
    ]


@torch.no_grad()
def predict_records(
    model: torch.nn.Module,
    records: list[Record],
    classes: list[str],
    device: torch.device,
    batch_size: int = 4,
    num_workers: int = 0,
    progress: bool = True,
) -> list[dict[str, np.ndarray]]:
    """Raw detector outputs (after the model's own per-class NMS) as NumPy, in record order."""
    model.eval()
    loader = DataLoader(
        DetectionDataset(records, classes, EvalTransform()),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_detection,
    )
    outputs: list[dict[str, np.ndarray]] = []
    for images, _ in tqdm(loader, desc="predict", disable=None if progress else True, leave=False):
        for out in model([img.to(device) for img in images]):
            outputs.append({k: v.cpu().numpy() for k, v in out.items()})
    return outputs


def evaluate_model(
    model: torch.nn.Module,
    records: list[Record],
    classes: list[str],
    cfg: EvalConfig,
    device: torch.device,
    batch_size: int = 4,
    num_workers: int = 0,
) -> dict:
    predictions = predict_records(model, records, classes, device, batch_size, num_workers)
    return evaluate_detections(
        ground_truth_from_records(records, classes),
        predictions,
        classes,
        cfg.precision_iou_thresholds,
        cfg.operating_score_thresh,
    )
