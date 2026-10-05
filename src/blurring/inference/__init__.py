"""Inference stage: NMS, blurring, predictors (torch and ONNX) and the batch blurring pipeline.

Torch-dependent names are imported lazily, so the slim ONNX deployment (no torch installed)
can still ``from blurring.inference import ...`` the NumPy-only parts.
"""

from importlib import import_module

from blurring.inference.blur import BLUR_METHODS, blur_regions
from blurring.inference.nms import batched_nms, nms
from blurring.inference.onnx_predictor import OnnxPredictor
from blurring.inference.postprocess import Detections, postprocess

_LAZY = {
    "Predictor": "blurring.inference.predictor",
    "run_batch": "blurring.inference.batch_pipeline",
    "list_images": "blurring.inference.batch_pipeline",
}


def __getattr__(name: str):
    if name in _LAZY:
        return getattr(import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'blurring.inference' has no attribute {name!r}")


__all__ = [
    "BLUR_METHODS",
    "Detections",
    "OnnxPredictor",
    "Predictor",
    "batched_nms",
    "blur_regions",
    "list_images",
    "nms",
    "postprocess",
    "run_batch",
]
