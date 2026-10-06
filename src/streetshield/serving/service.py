"""Business logic behind the API: decode an upload, detect, blur, encode. No HTTP concerns here."""

from __future__ import annotations

import base64
import io
import json
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from streetshield.boxes import pad_and_clip, xyxy_to_xywh
from streetshield.config import Config
from streetshield.inference import Detections, OnnxPredictor, blur_regions
from streetshield.serving.schemas import BlurParams, BlurResponse, Detection, ModelInfo


class InvalidImageError(ValueError):
    """The upload is not a decodable image."""


def decode_image(data: bytes, max_side: int) -> tuple[Image.Image, tuple[int, int]]:
    """Decode, honour EXIF rotation (phone photos), convert to RGB and cap the longest side.

    Returns the working image and the original (w, h).
    """
    try:
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("File is not a supported image") from exc
    original = image.size
    if max(original) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image, original


def encode_jpeg_data_url(pixels: np.ndarray, quality: int) -> str:
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class Detector(Protocol):
    """What the service needs from a backend; both the torch Predictor and OnnxPredictor provide it."""

    classes: list[str]

    @property
    def backend(self) -> str: ...

    def detect_array(self, pixels: np.ndarray, score_thresh: float | None = None, nms_iou: float | None = None) -> Detections: ...


class BlurService:
    """Holds the loaded model and serialises inference; one instance per server process."""

    def __init__(self, predictor: Detector, cfg: Config, model_meta: dict):
        self.predictor = predictor
        self.cfg = cfg
        self.model_meta = model_meta  # arch / epoch / val, from the checkpoint or the ONNX sidecar
        # MPS/CUDA models are not safe to call from several request threads at once
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, cfg: Config) -> "BlurService":
        """Load the configured backend. torch is only imported for the torch backend."""
        if cfg.serving.backend == "onnx":
            predictor = OnnxPredictor(cfg.resolve(cfg.serving.onnx_model), cfg.inference, cfg.serving.onnx_threads)
            return cls(predictor, cfg, predictor.meta)

        from streetshield.inference.predictor import Predictor
        from streetshield.modeling import load_detector
        from streetshield.runtime import get_device

        device = get_device(cfg.train.device, cfg.train.mps_memory_fraction)
        model, classes, ckpt = load_detector(cfg.resolve(cfg.paths.checkpoint), device)
        meta = {"arch": ckpt["model_cfg"]["arch"], "epoch": ckpt.get("epoch"), "val": ckpt.get("val")}
        return cls(Predictor(model, classes, cfg.inference, device), cfg, meta)

    def default_params(self) -> BlurParams:
        inf = self.cfg.inference
        return BlurParams(score_thresh=inf.score_thresh, nms_iou=inf.nms_iou, box_padding=inf.box_padding, method=inf.blur)

    def blur(self, data: bytes, params: BlurParams) -> BlurResponse:
        start = time.perf_counter()
        image, (orig_w, orig_h) = decode_image(data, self.cfg.serving.max_side)
        pixels = np.asarray(image)
        with self._lock:
            dets = self.predictor.detect_array(pixels, params.score_thresh, params.nms_iou)
        h, w = pixels.shape[:2]
        blurred = blur_regions(pixels, pad_and_clip(dets.boxes, params.box_padding, w, h), params.method)

        boxes = xyxy_to_xywh(dets.boxes).astype(float)
        detections = [
            Detection(box=[round(v, 1) for v in box], label=label, score=round(float(score), 4))
            for box, label, score in zip(boxes.tolist(), dets.labels, dets.scores)
        ]
        counts = {c: 0 for c in self.predictor.classes} | dict(Counter(dets.labels))
        return BlurResponse(
            width=w,
            height=h,
            original_width=orig_w,
            original_height=orig_h,
            detections=detections,
            counts=counts,
            image=encode_jpeg_data_url(blurred, self.cfg.serving.jpeg_quality),
            elapsed_ms=round((time.perf_counter() - start) * 1000, 1),
            params=params,
        )

    def model_info(self) -> ModelInfo:
        artifacts = self.cfg.resolve(self.cfg.paths.artifacts_dir)
        return ModelInfo(
            status="ready",
            device=self.predictor.backend,
            arch=self.model_meta.get("arch", "unknown"),
            classes=self.predictor.classes,
            checkpoint_epoch=self.model_meta.get("epoch"),
            val=self.model_meta.get("val"),
            test=_read_json(artifacts / "metrics_test.json"),
            history=_read_json(artifacts / "train_history.json") or [],
            defaults=self.default_params(),
        )


def _read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None
