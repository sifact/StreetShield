"""Data stage: parse raw annotations, build splits, augment, and serve tensors to the model."""

from streetshield.data.annotations import Record, load_voc_dir, parse_voc
from streetshield.data.dataset import DetectionDataset, ImagePathDataset, collate_detection, load_rgb
from streetshield.data.feedback import build_retraining_set, find_hard_examples, load_reports
from streetshield.data.manifest import SPLITS, build_manifest, load_manifest, save_manifest, summarize
from streetshield.data.transforms import EvalTransform, TrainTransform

__all__ = [
    "SPLITS",
    "DetectionDataset",
    "EvalTransform",
    "ImagePathDataset",
    "Record",
    "TrainTransform",
    "build_manifest",
    "build_retraining_set",
    "collate_detection",
    "find_hard_examples",
    "load_manifest",
    "load_reports",
    "load_rgb",
    "load_voc_dir",
    "parse_voc",
    "save_manifest",
    "summarize",
]
