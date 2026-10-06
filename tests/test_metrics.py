import numpy as np
import pytest

from streetshield.evaluation import average_precision, evaluate_detections, match_class

GT = [{"boxes": np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32), "labels": np.array([1, 2])}]


def _pred(boxes, scores, labels):
    return [{"boxes": np.asarray(boxes, dtype=np.float32), "scores": np.asarray(scores), "labels": np.asarray(labels)}]


def test_perfect_detections_give_ap_one():
    report = evaluate_detections(GT, _pred(GT[0]["boxes"], [0.9, 0.8], [1, 2]), ["face", "plate"], [0.5], 0.5)
    assert report["mAP50"] == pytest.approx(1.0)
    assert report["mAP"] == pytest.approx(1.0)
    assert report["at_operating_score"]["by_iou"]["0.5"]["precision"] == pytest.approx(1.0)


def test_duplicate_detection_is_false_positive():
    m = match_class(GT, _pred([[0, 0, 10, 10], [0, 0, 10, 10]], [0.9, 0.8], [1, 1]), 1, 0.5)
    assert m.is_tp.tolist() == [True, False]


def test_wrong_class_does_not_match():
    m = match_class(GT, _pred([[0, 0, 10, 10]], [0.9], [2]), 2, 0.5)
    assert m.is_tp.tolist() == [False]


def test_precision_depends_on_iou_threshold():
    # one detection shifted so IoU with the face box is 0.6
    shifted = [[0, 0, 10, 6]]
    report = evaluate_detections(GT, _pred(shifted, [0.9], [1]), ["face", "plate"], [0.5, 0.7], 0.5)
    assert report["at_operating_score"]["by_iou"]["0.5"]["precision"] == pytest.approx(1.0)
    assert report["at_operating_score"]["by_iou"]["0.7"]["precision"] == pytest.approx(0.0)


def test_ap_ranks_false_positive_first():
    # a confident false positive before the true positive halves precision at that recall
    m = match_class(GT, _pred([[50, 50, 60, 60], [0, 0, 10, 10]], [0.9, 0.8], [1, 1]), 1, 0.5)
    assert average_precision(m) == pytest.approx(0.5)
    assert average_precision(m, "11_point") == pytest.approx(0.5)


def test_ap_nan_without_ground_truth():
    m = match_class(GT, _pred([], [], []), 3, 0.5)
    assert np.isnan(average_precision(m))
