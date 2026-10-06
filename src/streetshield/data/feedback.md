# feedback.py

The retraining data pipeline from the chapter: user reports and hard negative mining.

- `load_reports` reads annotator-verified user reports (JSONL, `[x, y, w, h]` boxes as in the chapter) and turns them into new training records.
- `find_hard_examples` runs the IoU matching from evaluation per image and flags images with a confident false positive (a hard negative) or any missed object.
- Matching requires both IoU ≥ `match_iou` and the same class, so a plate predicted as a face still counts as a mistake.
- `build_retraining_set` appends the reports and `oversample` copies of each hard example to the training split, so the next model sees its own errors more often.
- Everything is driven by the `feedback` block in the config, and the `streetshield mine` CLI command writes the new manifest.
