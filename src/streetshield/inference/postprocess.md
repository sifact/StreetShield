# postprocess.py

Backend-independent post-processing that turns raw detector output into the boxes that get blurred.

- Defines `Detections` (xyxy boxes, scores, class names), the type every predictor returns and the blur step consumes.
- Drops detections below the serving score threshold; the default 0.4 is deliberately low because a missed face is worse than an extra blurred patch.
- Runs explicit per-class NMS at the serving IoU so each object ends up with exactly one box.
- Maps the model's 1-based class ids back to names (`face`, `plate`).
- Pure NumPy, shared by the torch and ONNX backends, which is what guarantees both serve identical results.
