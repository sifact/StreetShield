import numpy as np

from blurring.inference import batched_nms, nms, postprocess


def test_nms_keeps_best_of_overlapping_cluster():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.8, 0.9, 0.7])
    assert nms(boxes, scores, 0.5).tolist() == [1, 2]


def test_nms_empty():
    assert nms(np.zeros((0, 4)), np.zeros(0), 0.5).size == 0


def test_batched_nms_does_not_suppress_across_classes():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    keep = batched_nms(boxes, np.array([0.9, 0.8]), np.array([1, 2]), 0.5)
    assert sorted(keep.tolist()) == [0, 1]


def test_postprocess_filters_scores_and_names_classes():
    raw = {
        "boxes": np.array([[0, 0, 10, 10], [1, 1, 11, 11], [30, 30, 40, 40]], dtype=np.float32),
        "scores": np.array([0.9, 0.85, 0.1]),
        "labels": np.array([1, 1, 2]),
    }
    dets = postprocess(raw, ["face", "plate"], score_thresh=0.4, nms_iou=0.4)
    assert dets.labels == ["face"]
    assert dets.scores.tolist() == [0.9]
