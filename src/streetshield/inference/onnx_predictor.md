# onnx_predictor.py

Torch-free backend that runs the exported detector with ONNX Runtime, used for the Azure Free-tier deployment.

- Loads `model.onnx` and its `.json` sidecar (class names, arch, epoch, val metrics) into a CPU `InferenceSession`.
- Preprocessing is just HWC uint8 → CHW float in [0, 1]; resizing, normalisation and box rescaling are inside the exported graph.
- Shares `postprocess.py` with the torch predictor, so score filtering and NMS are identical.
- Verified against PyTorch on all 208 test images: same mAP@0.5 (0.935), mAP@0.75 and mAP@[.5:.95], median box IoU 1.0.
- Lets the deployed app run in about 250 MB of dependencies instead of a ~2 GB torch install, and in ~0.05 s per image on an M4.
