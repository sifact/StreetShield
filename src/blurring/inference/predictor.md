# predictor.py

PyTorch backend: turns the trained checkpoint into final detections ready for blurring.

- `Predictor.from_checkpoint` rebuilds the model from the checkpoint alone (architecture, classes, weights) and puts it in eval mode on the chosen device (MPS, CUDA or CPU).
- `detect` batches image tensors through the model under `torch.no_grad()`; `detect_array` is the single-image NumPy entry point the API service shares with the ONNX backend.
- Score threshold and NMS IoU come from the `inference` config but can be overridden per call, which is how the UI sliders work.
- Post-processing (score filter + per-class NMS) is delegated to `postprocess.py`, so torch and ONNX produce identical final boxes.
- Used locally for the batch pipeline and `serve` with `serving.backend: torch`; deployment uses `onnx_predictor.py` instead.
