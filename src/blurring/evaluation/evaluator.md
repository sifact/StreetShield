# evaluator.py

Runs a trained detector over a split and turns its outputs into the metrics report.

- `predict_records` batches images through the model in eval mode with no gradients and returns NumPy boxes, scores and labels in record order.
- Uses the same `DetectionDataset` and `EvalTransform` as training, so evaluation preprocessing can't drift from training.
- `ground_truth_from_records` converts manifest records into the per-image box/label arrays the metrics expect.
- `evaluate_model` is the single call used by the trainer (val mAP each epoch, for picking the best checkpoint) and by the `evaluate` CLI (test set).
- Predictions are also reused by hard negative mining (`data/feedback.py`), so the model's mistakes are found with the same code that scores it.
