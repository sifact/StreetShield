# metrics.py

Implements the chapter's offline metrics for object detection: precision at an IoU threshold, AP and mAP.

- `match_class` sorts detections by score and greedily matches each to the best unclaimed ground-truth box of the same class; it is a true positive only if IoU ≥ threshold, so duplicate boxes count as false positives.
- Precision = correct detections / total detections, computed at the chapter's IoU thresholds (0.1, 0.5, 0.7) to show how much the score depends on the threshold.
- `average_precision` integrates the precision-recall curve for one class (all-point interpolation, plus the VOC2008 11-point variant the chapter cites).
- Reports mAP@0.5, mAP@0.75 and COCO-style mAP@[.5:.95], which averages AP across IoU thresholds — the chapter's "average across thresholds" idea in its standard form.
- Recall is reported next to precision at the operating score, because for privacy a missed face is worse than an extra blurred patch.
