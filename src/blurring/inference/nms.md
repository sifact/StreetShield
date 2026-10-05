# nms.py

Non-maximum suppression, the post-processing step the chapter singles out for serving.

- The RPN proposes many overlapping boxes around each object; NMS keeps the most confident one and removes the rest.
- `nms` repeatedly takes the highest-scoring box and drops every remaining box whose IoU with it is above `iou_thresh`.
- `batched_nms` runs NMS separately per class, so a face box never suppresses a plate box it happens to overlap.
- Uses the shared `boxes.pairwise_iou`, so NMS and evaluation agree on what "overlap" means.
- Plain NumPy and unit-tested against hand-made cases, since NMS is a common interview question worth being able to show line by line.
