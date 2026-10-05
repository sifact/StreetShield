# batch_pipeline.py

The chapter's batch prediction pipeline: preprocess → detect → NMS → blur → store.

- CPU DataLoader workers decode images (preprocessing) while the GPU/MPS runs detection, mirroring the chapter's split between a CPU-bound preprocessing service and a GPU blurring service.
- For each image it detects objects, refines them with NMS, pads the boxes, blurs them and saves the result under the same relative path in the output folder (the "blurred Street View images" store).
- Appends one JSON line per image to `detections.jsonl` with `[x, y, w, h]` boxes, labels and scores, so every blur can be audited and user reports can point at exact detections.
- Skips images that already have an output unless `--overwrite`, so a crashed or interrupted batch run can resume where it stopped.
- All knobs (threshold, NMS IoU, padding, blur method, batch size, workers) come from the `inference` config block.
