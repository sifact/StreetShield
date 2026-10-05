"""Request/response models for the HTTP API (the contract the React UI codes against)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BlurParams(BaseModel):
    score_thresh: float = Field(0.4, ge=0.0, le=1.0)
    nms_iou: float = Field(0.4, ge=0.0, le=1.0)
    box_padding: float = Field(0.15, ge=0.0, le=1.0)
    method: Literal["gaussian", "pixelate"] = "gaussian"


class Detection(BaseModel):
    box: list[float]  # [x, y, w, h] in the returned image's pixels (chapter format)
    label: str
    score: float


class BlurResponse(BaseModel):
    width: int
    height: int
    original_width: int
    original_height: int
    detections: list[Detection]
    counts: dict[str, int]
    image: str  # data URL of the blurred image
    elapsed_ms: float
    params: BlurParams


class ModelInfo(BaseModel):
    status: str
    device: str
    arch: str
    classes: list[str]
    checkpoint_epoch: int | None
    val: dict[str, float] | None
    test: dict | None
    history: list[dict]
    defaults: BlurParams
