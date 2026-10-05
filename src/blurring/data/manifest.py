"""Merge all annotation sources into one manifest with reproducible train/val/test splits."""

from __future__ import annotations

import json
import logging
import os
import random
from collections import Counter
from pathlib import Path

from blurring.config import PROJECT_ROOT, Config
from blurring.data.annotations import Record, load_voc_dir

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")
Manifest = dict[str, list[Record]]


def split_records(records: list[Record], val_frac: float, test_frac: float, seed: int) -> Manifest:
    """Shuffle with a fixed seed and cut into train/val/test."""
    shuffled = records[:]
    random.Random(seed).shuffle(shuffled)
    n_test = round(len(shuffled) * test_frac)
    n_val = round(len(shuffled) * val_frac)
    return {
        "test": shuffled[:n_test],
        "val": shuffled[n_test : n_test + n_val],
        "train": shuffled[n_test + n_val :],
    }


def build_manifest(cfg: Config) -> Manifest:
    """Split each source separately, then concatenate.

    Splitting per source keeps the face/plate mix the same in every split, so val
    and test metrics are not skewed by one dataset dominating a split by chance.
    """
    manifest: Manifest = {s: [] for s in SPLITS}
    for source in cfg.data.sources:
        records = load_voc_dir(
            cfg.resolve(source.annots_dir), cfg.resolve(source.images_dir), cfg.data.label_map, source.name
        )
        if not records:
            logger.warning("Source %s produced no records; is it downloaded?", source.name)
        for split, items in split_records(records, cfg.data.val_frac, cfg.data.test_frac, cfg.seed).items():
            manifest[split].extend(items)
    return manifest


def summarize(manifest: Manifest) -> dict[str, dict[str, int]]:
    """Image and per-class box counts per split, logged after prepare-data."""
    out = {}
    for split, records in manifest.items():
        counts = Counter(label for r in records for label in r.labels)
        out[split] = {"images": len(records), **dict(sorted(counts.items()))}
    return out


def save_manifest(manifest: Manifest, path: Path, root: Path = PROJECT_ROOT) -> None:
    """Image paths are stored relative to ``root`` so the manifest survives moving the project."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def rel(r: Record) -> dict:
        d = r.to_dict()
        d["image_path"] = os.path.relpath(d["image_path"], root)
        return d

    with open(path, "w") as fh:
        json.dump({k: [rel(r) for r in v] for k, v in manifest.items()}, fh, indent=1)


def load_manifest(path: Path, root: Path = PROJECT_ROOT) -> Manifest:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `blurring prepare-data` first.")
    with open(path) as fh:
        raw = json.load(fh)
    manifest = {k: [Record.from_dict(r) for r in v] for k, v in raw.items()}
    for records in manifest.values():
        for r in records:
            r.image_path = str(root / r.image_path)
    return manifest
