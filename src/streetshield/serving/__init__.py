"""Serving stage: HTTP API over the blurring pipeline, used by the React UI in web/."""

from streetshield.serving.app import create_app
from streetshield.serving.schemas import BlurParams, BlurResponse, Detection, ModelInfo
from streetshield.serving.service import BlurService, InvalidImageError, decode_image

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
