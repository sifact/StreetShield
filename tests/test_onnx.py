"""ONNX backend must match the torch backend. Uses the real exported model, so it skips until
`streetshield export-onnx` has been run (export needs a trained checkpoint and a real image)."""

import numpy as np
import pytest
import torch

from streetshield.config import load_config

cfg = load_config()
ONNX = cfg.resolve(cfg.serving.onnx_model)
CKPT = cfg.resolve(cfg.paths.checkpoint)
pytestmark = pytest.mark.skipif(not (ONNX.exists() and CKPT.exists()), reason="needs trained + exported model")


def test_onnx_matches_torch_on_real_image():
    from streetshield.data import load_manifest, load_rgb
    from streetshield.inference import OnnxPredictor, Predictor

    record = max(load_manifest(cfg.resolve(cfg.paths.manifest))["test"], key=lambda r: len(r.labels))
    pixels = np.asarray(load_rgb(record.image_path))
    onnx = OnnxPredictor(ONNX, cfg.inference)
    torch_pred = Predictor.from_checkpoint(CKPT, cfg.inference, torch.device("cpu"))
    # compare like with like: the export may cap RPN proposals (see modeling/export.py)
    for d in (torch_pred.model.rpn._pre_nms_top_n, torch_pred.model.rpn._post_nms_top_n):
        d["testing"] = onnx.meta["proposals"]
    onnx_dets, torch_dets = onnx.detect_array(pixels), torch_pred.detect_array(pixels)

    assert len(onnx_dets.labels) == len(torch_dets.labels) > 0
    assert onnx_dets.labels == torch_dets.labels
    np.testing.assert_allclose(onnx_dets.boxes, torch_dets.boxes, atol=1.0)
    np.testing.assert_allclose(onnx_dets.scores, torch_dets.scores, atol=1e-3)
