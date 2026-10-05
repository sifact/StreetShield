"""Irreversibly obscure image regions. The blur must be strong enough that faces and plates can't be read."""

from __future__ import annotations

import cv2
import numpy as np

BLUR_METHODS = ("gaussian", "pixelate")
PIXEL_BLOCKS = 8  # a pixelated region keeps at most 8x8 "pixels" of information, too few to read a plate


def _gaussian(region: np.ndarray) -> np.ndarray:
    # kernel scales with region size so a 400px face is as unreadable as a 40px one
    h, w = region.shape[:2]
    k = max(3, (max(h, w) // 2) | 1)
    return cv2.GaussianBlur(region, (k, k), sigmaX=0)


def _pixelate(region: np.ndarray) -> np.ndarray:
    h, w = region.shape[:2]
    small = cv2.resize(region, (max(1, min(PIXEL_BLOCKS, w)), max(1, min(PIXEL_BLOCKS, h))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def blur_regions(image: np.ndarray, boxes: np.ndarray, method: str = "gaussian") -> np.ndarray:
    """Return a copy of an HxWx3 uint8 image with every ``[x1, y1, x2, y2]`` box blurred."""
    if method not in BLUR_METHODS:
        raise ValueError(f"Unknown blur method {method!r}; choose from {BLUR_METHODS}")
    blur = _gaussian if method == "gaussian" else _pixelate
    out = image.copy()
    h, w = out.shape[:2]
    for x1, y1, x2, y2 in np.asarray(boxes).reshape(-1, 4):
        x1, y1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
        x2, y2 = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        out[y1:y2, x1:x2] = blur(np.ascontiguousarray(out[y1:y2, x1:x2]))
    return out
