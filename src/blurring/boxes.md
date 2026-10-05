# boxes.py

Bounding-box geometry shared by every pipeline stage, so IoU is defined exactly once.

- Converts between the chapter's `[x, y, w, h]` format (used for reports and detection output) and the `[x1, y1, x2, y2]` format torchvision needs internally.
- `pairwise_iou` computes the full IoU matrix (intersection / union, chapter Fig. 3.9) that evaluation, NMS and hard negative mining all rely on.
- Handles empty inputs and zero-area boxes without dividing by zero, which happens with images that contain no objects.
- `pad_and_clip` grows detected boxes by a fraction of their size before blurring, so the edges of a face or plate are not left sharp.
- Pure NumPy with no torch dependency, so it is cheap to unit-test and safe to import from any sub-package without circular imports.
