"""Command-line entry point: prepare-data, train, evaluate, blur, export-onnx, serve, mine."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

from streetshield.config import Config, load_config
from streetshield.logging_utils import setup_logging
from streetshield.runtime import get_device, seed_everything

logger = logging.getLogger("streetshield")


def cmd_prepare_data(cfg: Config, args: argparse.Namespace) -> None:
    from streetshield.data import build_manifest, save_manifest, summarize

    manifest = build_manifest(cfg)
    path = cfg.resolve(cfg.paths.manifest)
    save_manifest(manifest, path)
    logger.info("Wrote %s\n%s", path, json.dumps(summarize(manifest), indent=2))


def cmd_train(cfg: Config, args: argparse.Namespace) -> None:
    """Train in-process, or supervise one fresh subprocess per ``train.epochs_per_process`` epochs.

    PyTorch's Apple MPS backend leaks host memory during detection training (~5 MB per
    step, measured); it is only returned when the process exits. Restarting from the saved
    state every epoch bounds peak memory without changing the training result.
    """
    from streetshield.training import train

    per_process = cfg.train.epochs_per_process
    if args.worker or per_process <= 0:
        result = train(
            cfg, limit=args.limit, epochs=args.epochs, resume=args.resume or args.worker,
            max_epochs_this_run=per_process if args.worker else None,
        )
        logger.info("Best val mAP50 so far: %.3f", result["best_val_mAP50"])
        return

    artifacts = cfg.resolve(cfg.paths.artifacts_dir)
    state, history_path = artifacts / "last_state.pt", artifacts / "train_history.json"
    if not args.resume:
        state.unlink(missing_ok=True)
        history_path.unlink(missing_ok=True)
    total = args.epochs or cfg.train.epochs
    cmd = [sys.executable, "-m", "streetshield.cli", *(["--config", args.config] if args.config else []), "train", "--worker"]
    cmd += [*(["--limit", str(args.limit)] if args.limit else []), *(["--epochs", str(args.epochs)] if args.epochs else [])]
    while True:
        done = len(json.loads(history_path.read_text())) if history_path.exists() else 0
        if done >= total:
            break
        logger.info("Starting worker process at epoch %d/%d", done + 1, total)
        subprocess.run(cmd, check=True)
    best = max((e["val"]["mAP50"] for e in json.loads(history_path.read_text())), default=float("nan"))
    logger.info("Training finished. Best val mAP50: %.3f", best)


def cmd_evaluate(cfg: Config, args: argparse.Namespace) -> None:
    from streetshield.data import load_manifest
    from streetshield.evaluation import evaluate_model
    from streetshield.modeling import load_detector

    device = get_device(cfg.train.device, cfg.train.mps_memory_fraction)
    model, classes, ckpt = load_detector(Path(args.checkpoint or cfg.resolve(cfg.paths.checkpoint)), device)
    records = load_manifest(cfg.resolve(cfg.paths.manifest))[args.split][: args.limit]
    report = evaluate_model(model, records, classes, cfg.eval, device, cfg.inference.batch_size)
    report.update(split=args.split, checkpoint_epoch=ckpt.get("epoch"))
    out = cfg.resolve(cfg.paths.artifacts_dir) / f"metrics_{args.split}.json"
    out.write_text(json.dumps(report, indent=2))
    logger.info("Wrote %s\n%s", out, json.dumps(report, indent=2))


def cmd_blur(cfg: Config, args: argparse.Namespace) -> None:
    from streetshield.inference import Predictor, run_batch

    if args.score_thresh is not None:
        cfg.inference.score_thresh = args.score_thresh
    if args.method:
        cfg.inference.blur = args.method
    device = get_device(cfg.train.device, cfg.train.mps_memory_fraction)
    predictor = Predictor.from_checkpoint(Path(args.checkpoint or cfg.resolve(cfg.paths.checkpoint)), cfg.inference, device)
    counts = run_batch(predictor, Path(args.input), Path(args.output), cfg.inference, overwrite=args.overwrite)
    logger.info("Blurred %d images, %d regions -> %s", counts["images"], counts["detections"], args.output)


def cmd_mine(cfg: Config, args: argparse.Namespace) -> None:
    """Data pipeline: user reports + hard examples from the train split -> retraining manifest."""
    from streetshield.data import build_retraining_set, find_hard_examples, load_manifest, load_reports, save_manifest
    from streetshield.evaluation import predict_records
    from streetshield.modeling import load_detector

    device = get_device(cfg.train.device, cfg.train.mps_memory_fraction)
    model, classes, _ = load_detector(Path(args.checkpoint or cfg.resolve(cfg.paths.checkpoint)), device)
    manifest = load_manifest(cfg.resolve(cfg.paths.manifest))
    train_records = manifest["train"]

    predictions = predict_records(model, train_records, classes, device, cfg.inference.batch_size)
    hard_idx = find_hard_examples(
        train_records, predictions, classes, cfg.feedback.hard_negative_score, cfg.feedback.match_iou
    )
    hard = [train_records[i] for i in hard_idx]
    reports = load_reports(Path(args.reports)) if args.reports else []

    manifest["train"] = build_retraining_set(train_records, reports, hard, cfg.feedback.oversample)
    out = Path(args.output) if args.output else cfg.resolve(cfg.paths.artifacts_dir) / "manifest_retrain.json"
    save_manifest(manifest, out)
    logger.info(
        "%d hard examples, %d user reports -> %d train records in %s", len(hard), len(reports), len(manifest["train"]), out
    )


def cmd_export_onnx(cfg: Config, args: argparse.Namespace) -> None:
    """Export the checkpoint to ONNX, traced on the val image with the most annotated objects."""
    from torchvision.transforms import functional as F

    from streetshield.data import load_manifest, load_rgb
    from streetshield.modeling import export_onnx

    val = load_manifest(cfg.resolve(cfg.paths.manifest))["val"]
    record = max(val, key=lambda r: len(r.labels))
    out = Path(args.output) if args.output else cfg.resolve(cfg.serving.onnx_model)
    checkpoint = Path(args.checkpoint or cfg.resolve(cfg.paths.checkpoint))
    export_onnx(checkpoint, out, F.to_tensor(load_rgb(record.image_path)), cfg.serving.onnx_proposals)
    logger.info("Exported %s (%.1f MB), traced on %s", out, out.stat().st_size / 1e6, Path(record.image_path).name)


def cmd_serve(cfg: Config, args: argparse.Namespace) -> None:
    """Start the HTTP API the React UI talks to."""
    import uvicorn

    from streetshield.serving import create_app

    uvicorn.run(create_app(cfg), host=args.host or cfg.serving.host, port=args.port or cfg.serving.port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="streetshield", description=__doc__)
    parser.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare-data", help="Parse raw VOC datasets into a split manifest").set_defaults(func=cmd_prepare_data)

    p = sub.add_parser("train", help="Fine-tune the detector")
    p.add_argument("--limit", type=int, help="Cap images per split (smoke runs)")
    p.add_argument("--epochs", type=int, help="Override train.epochs")
    p.add_argument("--resume", action="store_true", help="Continue from artifacts/last_state.pt")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)  # internal: one supervised chunk
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="Offline metrics (AP, mAP, precision@IoU) on a split")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--checkpoint")
    p.add_argument("--limit", type=int)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("blur", help="Batch-blur every image in a folder")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--score-thresh", type=float)
    p.add_argument("--method", choices=["gaussian", "pixelate"])
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_blur)

    p = sub.add_parser("export-onnx", help="Export the checkpoint to ONNX for torch-free serving")
    p.add_argument("--checkpoint")
    p.add_argument("--output")
    p.set_defaults(func=cmd_export_onnx)

    p = sub.add_parser("serve", help="Run the HTTP API for the web UI")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("mine", help="Build a retraining manifest from hard examples and user reports")
    p.add_argument("--reports", help="JSONL of annotator-verified user reports")
    p.add_argument("--checkpoint")
    p.add_argument("--output")
    p.set_defaults(func=cmd_mine)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    args.func(cfg, args)


if __name__ == "__main__":
    main()
