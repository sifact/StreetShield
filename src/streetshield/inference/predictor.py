"""PyTorch backend: load a trained checkpoint and produce final detections (score filter + NMS)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import functional as F

from streetshield.config import InferenceConfig
from streetshield.inference.postprocess import Detections, postprocess
from streetshield.modeling import load_detector


class Predictor:
    """Detector + post-processing, the "detect then refine with NMS" part of the blurring service."""

    def __init__(self, model: torch.nn.Module, classes: list[str], cfg: InferenceConfig, device: torch.device):
        self.model, self.classes, self.cfg, self.device = model, classes, cfg, device

    @property
    def backend(self) -> str:
        return f"torch:{self.device}"

    @classmethod
    def from_checkpoint(cls, path: Path, cfg: InferenceConfig, device: torch.device) -> "Predictor":
        model, classes, _ = load_detector(path, device)
        return cls(model, classes, cfg, device)

    @torch.no_grad()
    def detect(
        self, images: list[torch.Tensor], score_thresh: float | None = None, nms_iou: float | None = None
    ) -> list[Detections]:
        """Detect objects; ``score_thresh``/``nms_iou`` override the config per call (used by the API)."""
        score = self.cfg.score_thresh if score_thresh is None else score_thresh
        iou = self.cfg.nms_iou if nms_iou is None else nms_iou
        outputs = self.model([img.to(self.device) for img in images])
        return [postprocess({k: v.cpu().numpy() for k, v in out.items()}, self.classes, score, iou) for out in outputs]

    def detect_array(self, pixels: np.ndarray, score_thresh: float | None = None, nms_iou: float | None = None) -> Detections:
        """Single HxWx3 uint8 image; the interface the API service uses for both backends."""
        return self.detect([F.to_tensor(pixels)], score_thresh, nms_iou)[0]
