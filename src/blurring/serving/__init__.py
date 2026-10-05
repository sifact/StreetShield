"""Serving stage: HTTP API over the blurring pipeline, used by the React UI in web/."""

from blurring.serving.app import create_app
from blurring.serving.schemas import BlurParams, BlurResponse, Detection, ModelInfo
from blurring.serving.service import BlurService, InvalidImageError, decode_image

__all__ = [
    "BlurParams",
    "BlurResponse",
    "BlurService",
    "Detection",
    "InvalidImageError",
    "ModelInfo",
    "create_app",
    "decode_image",
]
