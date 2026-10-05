"""Data stage: parse raw annotations, build splits, augment, and serve tensors to the model."""

from blurring.data.annotations import Record, load_voc_dir, parse_voc
from blurring.data.dataset import DetectionDataset, ImagePathDataset, collate_detection, load_rgb
from blurring.data.feedback import build_retraining_set, find_hard_examples, load_reports
from blurring.data.manifest import SPLITS, build_manifest, load_manifest, save_manifest, summarize
from blurring.data.transforms import EvalTransform, TrainTransform

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
