"""Batch prediction pipeline (chapter Fig. 3.12): preprocess -> detect -> NMS -> blur -> store.

Preprocessing (image decoding) runs in CPU DataLoader workers while detection runs on
the GPU in the main process; the chapter separates these so each can scale on its own.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from blurring.boxes import pad_and_clip, xyxy_to_xywh
from blurring.config import InferenceConfig
from blurring.data import ImagePathDataset, collate_detection
from blurring.data.annotations import IMAGE_EXTENSIONS
from blurring.inference.blur import blur_regions
from blurring.inference.predictor import Predictor

logger = logging.getLogger(__name__)


def list_images(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def run_batch(
    predictor: Predictor, input_dir: Path, output_dir: Path, cfg: InferenceConfig, overwrite: bool = False
) -> dict:
    """Blur every image under ``input_dir`` into ``output_dir`` (same relative paths).

    Writes ``detections.jsonl`` with boxes in the chapter's ``[x, y, w, h]`` format, so
    blurred images can be audited and user reports can reference exact detections.
    Already-processed images are skipped unless ``overwrite``, so a crashed run can resume.
    """
    paths = list_images(input_dir)
    todo = [p for p in paths if overwrite or not (output_dir / p.relative_to(input_dir)).exists()]
    logger.info("%d images found, %d to process", len(paths), len(todo))
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        ImagePathDataset(todo), batch_size=cfg.batch_size, num_workers=cfg.num_workers, collate_fn=collate_detection
    )
    counts = {"images": 0, "detections": 0}
    with open(output_dir / "detections.jsonl", "a") as log:
        for tensors, originals, sources in tqdm(loader, desc="blur"):
            for dets, original, source in zip(predictor.detect(list(tensors)), originals, sources):
                h, w = original.shape[:2]
                padded = pad_and_clip(dets.boxes, cfg.box_padding, w, h)
                blurred = blur_regions(original, padded, cfg.blur)

                rel = Path(source).relative_to(input_dir)
                dest = output_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(blurred).save(dest)

                log.write(json.dumps({
                    "image": str(rel),
                    "boxes": np.round(xyxy_to_xywh(dets.boxes).astype(float), 1).tolist(),
                    "labels": dets.labels,
                    "scores": np.round(dets.scores.astype(float), 3).tolist(),
                }) + "\n")
                counts["images"] += 1
                counts["detections"] += len(dets.labels)
    return counts
