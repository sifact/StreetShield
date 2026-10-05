"""Fine-tune the detector: forward -> combined loss -> backward, validate each epoch, keep the best checkpoint."""

from __future__ import annotations

import json
import logging
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from blurring.config import Config
from blurring.data import DetectionDataset, Record, TrainTransform, collate_detection, load_manifest
from blurring.evaluation import evaluate_model
from blurring.modeling import build_detector, save_checkpoint
from blurring.runtime import get_device, seed_everything
from blurring.training.losses import combine_losses

logger = logging.getLogger(__name__)

SELECTION_METRIC = "mAP50"  # more stable than mAP@[.5:.95] on a ~200-image val split


def _make_loader(records: list[Record], cfg: Config, epoch: int) -> DataLoader:
    # seeded per epoch, so shuffling is reproducible even when epochs run in separate processes
    generator = torch.Generator().manual_seed(cfg.seed + epoch)
    return DataLoader(
        DetectionDataset(records, cfg.data.classes, TrainTransform(cfg.augment)),
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_detection,
        generator=generator,
    )


def _release_cached_memory(device: torch.device) -> None:
    """Return the allocator's cached blocks between train and eval, so peak memory doesn't stack."""
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def _warmup_factor(iteration: int, warmup_iters: int) -> float:
    """Linear warmup: a full LR on freshly replaced heads can blow up the first few batches."""
    if iteration >= warmup_iters:
        return 1.0
    return 0.001 + (1 - 0.001) * iteration / warmup_iters


def _state_path(cfg: Config) -> Path:
    return cfg.resolve(cfg.paths.artifacts_dir) / "last_state.pt"


def train(
    cfg: Config,
    limit: int | None = None,
    epochs: int | None = None,
    resume: bool = False,
    max_epochs_this_run: int | None = None,
) -> dict:
    """Train on the manifest's train split and select the checkpoint with the best val mAP@0.5.

    After every epoch the full training state is saved to ``last_state.pt``. With ``resume``
    training continues from it, and ``max_epochs_this_run`` stops early so the CLI can run
    each epoch in a fresh process (see ``train.epochs_per_process``). ``limit`` caps images
    per split and ``epochs`` overrides the config; both exist for quick smoke runs.
    """
    seed_everything(cfg.seed)
    device = get_device(cfg.train.device, cfg.train.mps_memory_fraction)
    manifest = load_manifest(cfg.resolve(cfg.paths.manifest))
    train_records, val_records = manifest["train"][:limit], manifest["val"][:limit]
    num_epochs = epochs or cfg.train.epochs

    model = build_detector(cfg.model, cfg.num_classes).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg.train.lr, momentum=cfg.train.momentum, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.train.lr_milestones, gamma=0.1)

    artifacts = cfg.resolve(cfg.paths.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    history, best, iteration, first_epoch = [], -1.0, 0, 1
    if resume and _state_path(cfg).exists():
        state = torch.load(_state_path(cfg), map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, best, iteration, first_epoch = state["history"], state["best"], state["iteration"], state["epoch"] + 1
    last_epoch = num_epochs if max_epochs_this_run is None else min(num_epochs, first_epoch + max_epochs_this_run - 1)
    logger.info(
        "Device %s | train %d images | val %d images | epochs %d-%d of %d",
        device, len(train_records), len(val_records), first_epoch, last_epoch, num_epochs,
    )

    for epoch in range(first_epoch, last_epoch + 1):
        torch.manual_seed(cfg.seed + epoch)  # augmentation stream independent of process boundaries
        model.train()
        sums, batches, start = defaultdict(float), 0, time.time()
        for images, targets in tqdm(_make_loader(train_records, cfg, epoch), desc=f"epoch {epoch}/{num_epochs}", leave=False):
            for group in optimizer.param_groups:
                group["lr"] = scheduler.get_last_lr()[0] * _warmup_factor(iteration, cfg.train.warmup_iters)
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            total, parts = combine_losses(model(images, targets), cfg.train.lambda_reg)
            if not math.isfinite(parts["total"]):
                logger.warning("Non-finite loss at iteration %d; skipping batch", iteration)
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()

            for k, v in parts.items():
                sums[k] += v
            batches += 1
            iteration += 1
        scheduler.step()

        train_losses = {k: v / max(batches, 1) for k, v in sums.items()}
        report = evaluate_model(model, val_records, cfg.data.classes, cfg.eval, device, cfg.train.batch_size)
        _release_cached_memory(device)
        entry = {
            "epoch": epoch,
            "seconds": round(time.time() - start, 1),
            "lr": scheduler.get_last_lr()[0],
            "train_loss": train_losses,
            "val": {k: report[k] for k in ("mAP50", "mAP75", "mAP")},
            "val_per_class_AP50": {c: v["AP50"] for c, v in report["per_class"].items()},
        }
        history.append(entry)
        logger.info(
            "epoch %d | loss %.3f (cls %.3f, reg %.3f) | val mAP50 %.3f mAP %.3f | %s",
            epoch, train_losses.get("total", float("nan")), train_losses.get("l_cls", float("nan")),
            train_losses.get("l_reg", float("nan")), report["mAP50"], report["mAP"], entry["val_per_class_AP50"],
        )

        score = report[SELECTION_METRIC]
        if not math.isnan(score) and score > best:
            best = score
            save_checkpoint(
                model, cfg.resolve(cfg.paths.checkpoint), cfg.model, cfg.data.classes, epoch=epoch, val=entry["val"]
            )
            logger.info("Saved new best checkpoint (val %s %.3f)", SELECTION_METRIC, score)

        with open(artifacts / "train_history.json", "w") as fh:
            json.dump(history, fh, indent=2)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "iteration": iteration,
                "best": best,
                "history": history,
            },
            _state_path(cfg),
        )

    return {"best_val_" + SELECTION_METRIC: best, "history": history, "done": last_epoch >= num_epochs}
