"""ONNX Runtime backend: same detections as the PyTorch predictor without the ~2 GB torch install.

This is what runs on the Azure App Service Free tier (1 GB RAM, 1 GB disk), where torch can't fit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from streetshield.config import InferenceConfig
from streetshield.inference.postprocess import Detections, postprocess


class OnnxPredictor:
    """Runs the exported detector (``modeling.export_onnx``) with onnxruntime on CPU."""

    def __init__(self, model_path: Path, cfg: InferenceConfig, threads: int = 0):
        import onnxruntime as ort  # imported here so the torch path never needs onnxruntime

        options = ort.SessionOptions()
        # no arena / memory-pattern caching: keeps resident memory low (the Free tier has 1 GB), at a small speed cost
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        meta = json.loads(Path(model_path).with_suffix(".json").read_text())
        self.classes: list[str] = meta["classes"]
        self.meta = meta
        self.cfg = cfg

    @property
    def backend(self) -> str:
        return "onnxruntime:cpu"

    def detect_array(self, pixels: np.ndarray, score_thresh: float | None = None, nms_iou: float | None = None) -> Detections:
        score = self.cfg.score_thresh if score_thresh is None else score_thresh
        iou = self.cfg.nms_iou if nms_iou is None else nms_iou
        # same preprocessing as torchvision's to_tensor: CHW float in [0, 1]; resize/normalise is inside the graph
        image = np.ascontiguousarray(pixels.transpose(2, 0, 1), dtype=np.float32) / 255.0
        boxes, labels, scores = self.session.run(["boxes", "labels", "scores"], {"image": image})
        return postprocess({"boxes": boxes, "labels": labels, "scores": scores}, self.classes, score, iou)
