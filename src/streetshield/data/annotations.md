# annotations.py

Parses Pascal VOC XML from different GitHub datasets into one `Record` format the rest of the pipeline uses.

- Defines `Record`: image path, size, `[x1, y1, x2, y2]` boxes and class names, matching the chapter's annotated-dataset table (Table 3.1).
- Maps each dataset's raw class names onto our two classes through `label_map` (e.g. `with_mask`, `without_mask`, `mask_weared_incorrect` → `face`; `license-plate` → `plate`) and drops anything unmapped.
- Clips boxes to the image and discards boxes thinner than 1 px, which would otherwise produce degenerate regression targets.
- Finds the image from the XML `<filename>` and falls back to the XML stem, which copes with odd names such as the plate set's `video.mp4#t=1055.jpg`.
- Skips (with a warning) annotations whose image is missing, so a partial download shrinks the dataset rather than crashing training.
