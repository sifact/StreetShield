"""Typed configuration loaded from one YAML file, so no path or hyperparameter is hard-coded elsewhere."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


@dataclass
class PathsConfig:
    manifest: str = "data/processed/manifest.json"
    artifacts_dir: str = "artifacts"
    checkpoint: str = "artifacts/model.pt"


@dataclass
class SourceConfig:
    name: str
    images_dir: str
    annots_dir: str


@dataclass
class DataConfig:
    classes: list[str] = field(default_factory=lambda: ["face", "plate"])
    sources: list[SourceConfig] = field(default_factory=list)
    label_map: dict[str, str] = field(default_factory=dict)
    val_frac: float = 0.15
    test_frac: float = 0.15


@dataclass
class AugmentConfig:
    hflip_prob: float = 0.5
    crop_prob: float = 0.3
    brightness: float = 0.25
    contrast: float = 0.25
    saturation: float = 0.25


@dataclass
class ModelConfig:
    arch: str = "fasterrcnn_mobilenet_v3_large_fpn"
    pretrained: bool = True
    fixed_size: list[int] | None = field(default_factory=lambda: [1024, 640])  # [w, h]
    min_size: int = 640
    max_size: int = 1066
    box_score_thresh: float = 0.05
    box_nms_thresh: float = 0.5
    detections_per_img: int = 100


@dataclass
class TrainConfig:
    epochs: int = 12
    batch_size: int = 2
    lr: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 1e-4
    warmup_iters: int = 200
    lr_milestones: list[int] = field(default_factory=lambda: [8, 11])
    lambda_reg: float = 1.0
    num_workers: int = 1
    device: str = "auto"
    epochs_per_process: int = 1
    mps_memory_fraction: float = 0.4


@dataclass
class EvalConfig:
    precision_iou_thresholds: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.7])
    operating_score_thresh: float = 0.5


@dataclass
class InferenceConfig:
    score_thresh: float = 0.4
    nms_iou: float = 0.4
    box_padding: float = 0.15
    blur: str = "gaussian"
    batch_size: int = 4
    num_workers: int = 2


@dataclass
class ServingConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    max_upload_mb: int = 20
    max_side: int = 2048
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:5173"])
    jpeg_quality: int = 92
    backend: str = "torch"  # torch | onnx
    onnx_model: str = "artifacts/model.onnx"
    onnx_threads: int = 0  # 0 = onnxruntime default
    onnx_proposals: int = 300  # RPN proposals baked into the ONNX export (torchvision default 1000)
    static_dir: str | None = None  # built React app to serve at "/" (single-origin deployment)


@dataclass
class FeedbackConfig:
    hard_negative_score: float = 0.5
    match_iou: float = 0.3
    oversample: int = 3


@dataclass
class Config:
    seed: int = 42
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)

    @property
    def num_classes(self) -> int:
        """Detector classes including the background slot torchvision reserves at index 0."""
        return len(self.data.classes) + 1

    def resolve(self, path: str | Path) -> Path:
        """Relative config paths are relative to the project root, not the shell's cwd."""
        p = Path(path)
        return p if p.is_absolute() else PROJECT_ROOT / p


def _build(cls: type, raw: dict[str, Any]) -> Any:
    """Recursively turn a YAML dict into the matching dataclass, rejecting unknown keys early."""
    known = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise KeyError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in raw.items():
        default = cls.__dataclass_fields__[name].default_factory  # type: ignore[attr-defined]
        sample = default() if callable(default) else None
        if is_dataclass(sample) and isinstance(value, dict):
            kwargs[name] = _build(type(sample), value)
        else:
            kwargs[name] = value
    obj = cls(**kwargs)
    if isinstance(obj, DataConfig):
        obj.sources = [s if isinstance(s, SourceConfig) else SourceConfig(**s) for s in obj.sources]
    return obj


def load_config(path: str | Path | None = None) -> Config:
    """Load a YAML config (default: configs/default.yaml) into a typed Config."""
    with open(path or DEFAULT_CONFIG) as fh:
        raw = yaml.safe_load(fh) or {}
    return _build(Config, raw)
