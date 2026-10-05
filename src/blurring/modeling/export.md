# export.py

Exports the trained Faster R-CNN to ONNX so it can be served without PyTorch.

- Uses the TorchScript-based exporter (opset 17) with dynamic height/width, so the graph accepts any image size.
- The exported graph contains the whole detector: resize to the fixed 1024×640 input, normalisation, backbone, RPN, ROI heads and box rescaling to original pixels.
- Must be traced on a real image with several detections; tracing on noise (zero boxes) bakes empty-result shapes into the graph and it then crashes on real images. `export_onnx` refuses samples with fewer than 2 detections.
- Optionally caps RPN proposals (`serving.onnx_proposals`, default 300 vs torchvision's 1000), which cuts memory for the 1 GB Free tier while changing test mAP@0.5 by only −0.0005.
- Writes a `.json` sidecar with class names, architecture, epoch and val metrics, which the API reports in `/api/model`; run via `blurring export-onnx`, which traces on the val image with the most objects.
