"""Two-stage detector (Faster R-CNN): conv backbone -> region proposal network -> classifier (chapter Fig. 3.7)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torchvision.models import detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from streetshield.config import ModelConfig

ARCHITECTURES = {
    "fasterrcnn_mobilenet_v3_large_fpn": detection.fasterrcnn_mobilenet_v3_large_fpn,
    "fasterrcnn_resnet50_fpn_v2": detection.fasterrcnn_resnet50_fpn_v2,
    "fasterrcnn_resnet50_fpn": detection.fasterrcnn_resnet50_fpn,
}


def build_detector(cfg: ModelConfig, num_classes: int) -> torch.nn.Module:
    """Build a Faster R-CNN and swap its classifier head for our classes (+ background).

    Starting from COCO weights matters with ~1.4k images: the backbone and RPN already
    know what "an object" looks like, so we only need to teach it faces vs plates.
    """
    if cfg.arch not in ARCHITECTURES:
        raise ValueError(f"Unknown arch {cfg.arch!r}; choose from {sorted(ARCHITECTURES)}")
    model = ARCHITECTURES[cfg.arch](
        weights="DEFAULT" if cfg.pretrained else None,
        weights_backbone="DEFAULT" if cfg.pretrained else None,
        min_size=cfg.min_size,
        max_size=cfg.max_size,
        box_score_thresh=cfg.box_score_thresh,
        box_nms_thresh=cfg.box_nms_thresh,
        box_detections_per_img=cfg.detections_per_img,
    )
    if cfg.fixed_size:
        # one input shape for every image: MPS compiles and caches a graph per distinct shape,
        # which leaks memory with variable-size images. Boxes are mapped back to original size.
        model.transform.fixed_size = tuple(cfg.fixed_size)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def save_checkpoint(model: torch.nn.Module, path: Path, model_cfg: ModelConfig, classes: list[str], **extra: Any) -> None:
    """Store weights with the config needed to rebuild the model, so serving never guesses the architecture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "model_cfg": vars(model_cfg), "classes": classes, **extra}, path
    )


def load_detector(path: Path, device: torch.device) -> tuple[torch.nn.Module, list[str], dict[str, Any]]:
    """Rebuild the detector from a checkpoint. Returns ``(model_in_eval_mode, classes, checkpoint)``."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model_cfg = ModelConfig(**{**ckpt["model_cfg"], "pretrained": False})
    model = build_detector(model_cfg, len(ckpt["classes"]) + 1)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["classes"], ckpt
