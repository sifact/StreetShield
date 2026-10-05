"""Combine the detector's loss terms the way the chapter defines it: L = L_cls + lambda * L_reg."""

from __future__ import annotations

import torch

# torchvision Faster R-CNN returns four terms; two are "is this the right class" and two are "is the box right"
CLASSIFICATION_TERMS = ("loss_classifier", "loss_objectness")
REGRESSION_TERMS = ("loss_box_reg", "loss_rpn_box_reg")


def combine_losses(loss_dict: dict[str, torch.Tensor], lambda_reg: float) -> tuple[torch.Tensor, dict[str, float]]:
    """Return the total loss to backprop and a float breakdown for logging."""
    missing = set(CLASSIFICATION_TERMS + REGRESSION_TERMS) - set(loss_dict)
    if missing:
        raise KeyError(f"Detector did not return loss terms: {sorted(missing)}")
    l_cls = sum(loss_dict[k] for k in CLASSIFICATION_TERMS)
    l_reg = sum(loss_dict[k] for k in REGRESSION_TERMS)
    total = l_cls + lambda_reg * l_reg
    parts = {k: float(v.detach()) for k, v in loss_dict.items()}
    parts.update(l_cls=float(l_cls.detach()), l_reg=float(l_reg.detach()), total=float(total.detach()))
    return total, parts
