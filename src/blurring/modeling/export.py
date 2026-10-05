"""Export the trained detector to ONNX so serving can run without PyTorch."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import torch

from blurring.modeling.detector import load_detector

OPSET = 17


def export_onnx(checkpoint: Path, out_path: Path, sample: torch.Tensor, proposals: int | None = None) -> Path:
    """Write ``out_path`` (.onnx) plus a ``.json`` sidecar with the class names.

    The graph includes torchvision's resize/normalise and box rescaling, so it takes a raw
    CHW float image of any size and returns boxes in that image's pixel coordinates.

    ``sample`` must be a real image on which the model detects several objects: tracing on
    noise (zero detections) bakes the empty-result shapes into the graph and the exported
    model then fails on any image with detections.
    """
    model, classes, ckpt = load_detector(checkpoint, torch.device("cpu"))
    if proposals:
        # fewer RPN proposals = smaller second stage; 300 vs the default 1000 changed test mAP@0.5 by -0.0005
        model.rpn._pre_nms_top_n["testing"] = proposals
        model.rpn._post_nms_top_n["testing"] = proposals
    with torch.no_grad():
        if len(model([sample])[0]["boxes"]) < 2:
            raise ValueError("Export sample yields <2 detections; pick an image with several objects")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # tracer warnings about data-dependent shapes are expected here
        torch.onnx.export(
            model,
            ([sample],),
            str(out_path),
            opset_version=OPSET,
            input_names=["image"],
            output_names=["boxes", "labels", "scores"],
            dynamic_axes={"image": {1: "height", 2: "width"}, "boxes": {0: "n"}, "labels": {0: "n"}, "scores": {0: "n"}},
            dynamo=False,
        )
    meta = {"classes": classes, "arch": ckpt["model_cfg"]["arch"], "epoch": ckpt.get("epoch"), "val": ckpt.get("val"), "opset": OPSET,
            "proposals": proposals or model.rpn._post_nms_top_n["testing"]}
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return out_path
