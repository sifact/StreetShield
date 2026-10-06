# StreetShield

Detects human faces and license plates in street images and blurs them before they are published. This is the system from *Machine Learning System Design Interview*, Chapter 3 (see [PROJECT_BRIEF.md](PROJECT_BRIEF.md)).

**Pipeline:** two-stage detector (Faster R-CNN, COCO-pretrained, fine-tuned) → NMS → padded Gaussian blur or pixelation, run as an offline batch job. User reports and hard examples feed a retraining manifest.

## Quickstart

```bash
uv sync --extra dev
# download the data: see data/README.md
uv run streetshield prepare-data                              # merge + split -> data/processed/manifest.json
uv run streetshield train                                     # best val mAP@0.5 -> artifacts/model.pt
uv run streetshield evaluate --split test                     # -> artifacts/metrics_test.json
uv run streetshield blur --input street_photos/ --output blurred/   # + blurred/detections.jsonl
uv run streetshield mine --reports reports.jsonl              # -> artifacts/manifest_retrain.json
uv run streetshield serve                                     # HTTP API for the web UI
uv run pytest
```

## Web UI

A dark-theme React app (Vite + TypeScript, in `web/`) on top of a FastAPI backend (`src/streetshield/serving/`).

```bash
uv run streetshield serve                      # API on http://127.0.0.1:8000 (loads artifacts/model.pt)
cd web && npm install && npm run dev       # UI on http://localhost:5173 (proxies /api to :8000)
```

- **Upload:** drag and drop, browse, or paste (⌘V) several images. They are processed one at a time, with the selected image first.
- **View:** a before/after compare slider plus Blurred and Original modes, with detection boxes you can toggle.
- **Results:** per-detection confidences, and a download of the blurred JPEG.
- **Settings:** confidence threshold, box padding, NMS IoU and blur style. Changes re-run the current image after 350 ms with no further edits.
- **Model panel:** the test metrics and the validation curve for each epoch.

API endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Health check |
| `GET /api/model` | Checkpoint info and metrics |
| `POST /api/blur` | Multipart `file`, plus optional `score_thresh`, `nms_iou`, `box_padding`, `method`. Returns the boxes in `[x, y, w, h]` and the blurred image as a data URL |

Uploads larger than 2048 px are downscaled first.

## Deploy to Azure (App Service Free tier)

The Free (F1) plan has 1 GB RAM, 1 GB disk and 60 CPU-minutes per day. PyTorch alone doesn't fit, so the deployment is torch-free:

- **ONNX model.** `streetshield export-onnx` writes `artifacts/model.onnx`. It matches PyTorch exactly on the test set (mAP@0.5 0.935, median box IoU 1.0).
- **Fewer proposals.** The export caps RPN proposals at 300 instead of 1000. That cut peak memory from ~835 MB to ~490 MB and changed test mAP@0.5 by −0.0005.
- **One app.** The API serves the built UI from the same origin, using `configs/azure.yaml`.

```bash
uv run streetshield export-onnx                              # artifacts/model.onnx (+ .json sidecar)
./deploy/azure/build_package.sh                          # build/streetshield-azure.zip (~80 MB)
SUBSCRIPTION=<subscription-id> ./deploy/azure/deploy.sh  # RG + F1 plan + web app + zip deploy
```

`deploy.sh` creates whatever is missing: resource group `rg-streetshield`, plan `plan-streetshield` (F1 Linux) and web app `streetshield`.
- To reuse an existing plan, set `PLAN` and `PLAN_RESOURCE_GROUP`.
- The free tier has no "Always On", so the first request after the app idles takes a while.

All paths and hyperparameters live in [configs/default.yaml](configs/default.yaml). Pass `--config other.yaml` to override them.

## Results (test split, 208 held-out images)

Fine-tuned `fasterrcnn_mobilenet_v3_large_fpn` for 12 epochs on an Apple M4 (about 3 min per epoch). The best val checkpoint was epoch 10.

| Class | AP@0.5 | AP@0.75 | AP@[.5:.95] | Precision* | Recall* | Ground-truth boxes |
|---|---|---|---|---|---|---|
| face | 0.910 | 0.687 | 0.584 | 0.913 | 0.874 | 494 |
| plate | 0.961 | 0.877 | 0.690 | 0.879 | 0.946 | 92 |
| **mean (mAP)** | **0.935** | **0.782** | **0.637** | | | |

\*Precision and recall at IoU 0.5 and score ≥ 0.5. The chapter's IoU-threshold example for precision: 0.923 at IoU 0.1, 0.907 at 0.5, 0.818 at 0.7.
The full report is in `artifacts/metrics_test.json`, and per-epoch history in `artifacts/train_history.json`.

Face recall (0.87) is the weakest number and the one that matters most for privacy. Serving uses a lower score threshold (0.4) to favour recall.

## Memory on Apple Silicon

PyTorch's MPS backend leaks host RAM during detection training (measured at about 5 MB per step, and it doesn't happen on CPU). The leak is far worse when input shapes vary. Mitigations in the config:
- `model.fixed_size` puts every image into one input shape.
- `train.batch_size: 2`.
- `train.mps_memory_fraction` caps GPU memory.
- `train.epochs_per_process: 1` runs each epoch in a fresh process that resumes from `artifacts/last_state.pt`.

With these, the training worker peaks at about 4–6.5 GB instead of growing without limit. An interrupted run continues with `streetshield train --resume`.

## Layout

```
src/streetshield/
  config.py, cli.py, runtime.py, boxes.py   cross-cutting: config, CLI, seeds/device, box geometry
  data/        annotations (VOC -> Record), manifest (splits), transforms (box-aware aug), dataset, feedback (reports + hard negatives)
  modeling/    detector (Faster R-CNN build/save/load)
  training/    losses (L = L_cls + λ·L_reg), trainer
  evaluation/  metrics (IoU matching, precision@IoU, AP, mAP), evaluator
  inference/   nms, blur, predictor, batch_pipeline
  serving/     schemas (API contract), service (decode -> detect -> blur), app (FastAPI routes), asgi
deploy/azure/  build_package.sh, deploy.sh, requirements.txt (torch-free runtime)
web/           React UI (components/, api.ts typed client)
```

Each core module has a `<module>.md` next to it explaining what it does.

## Chapter → code

| Chapter concept | Where |
|---|---|
| Two-stage network (conv layers → RPN → classifier) | `modeling/detector.py` |
| Annotated dataset `[x, y, w, h]` + class | `data/annotations.py`, `boxes.py` |
| Preprocessing + image augmentation (boxes transformed too) | `data/transforms.py` |
| Loss `L = L_cls + λ·L_reg` | `training/losses.py` |
| IoU, precision at IoU thresholds, AP, mAP | `boxes.py`, `evaluation/metrics.py` |
| NMS | `inference/nms.py` |
| Batch prediction pipeline (CPU preprocessing / GPU blurring service) | `inference/batch_pipeline.py` |
| Data pipeline: user reports, hard negative mining | `data/feedback.py`, `streetshield mine` |
| Online metric: user reports | `detections.jsonl` gives an auditable record that reports can point at |

## Deviations from the chapter

- **Augmentation is online, not offline.** The dataset is small, so storing 10x copies buys nothing.
- **Box regression uses smooth-L1** (torchvision's default) rather than MSE.
- **AP is the area under the precision-recall curve** per class, and COCO-style mAP@[.5:.95] averages it across IoU thresholds. Precision at the chapter's example thresholds (0.1, 0.5, 0.7) is also reported.
- **Recall is reported alongside precision,** because for privacy a missed face matters more than an extra blurred patch.
