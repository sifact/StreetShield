import numpy as np

from streetshield.boxes import pad_and_clip, pairwise_iou, xywh_to_xyxy, xyxy_to_xywh


def test_xywh_roundtrip():
    xywh = np.array([[10, 10, 25, 50], [120, 180, 40, 70]], dtype=np.float32)
    np.testing.assert_allclose(xyxy_to_xywh(xywh_to_xyxy(xywh)), xywh)
    np.testing.assert_allclose(xywh_to_xyxy(xywh)[0], [10, 10, 35, 60])


def test_iou_identical_disjoint_and_half():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [0, 0, 10, 5]])
    np.testing.assert_allclose(pairwise_iou(a, b), [[1.0, 0.0, 0.5]])


def test_iou_empty_shapes():
    assert pairwise_iou(np.zeros((0, 4)), np.array([[0, 0, 1, 1]])).shape == (0, 1)


def test_pad_and_clip_stays_in_image():
    out = pad_and_clip(np.array([[0, 0, 10, 10]]), 0.5, width=12, height=12)
    np.testing.assert_allclose(out, [[0, 0, 12, 12]])
