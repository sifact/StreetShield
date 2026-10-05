# blur.py

Obscures detected regions so faces and plates can't be recognised, which is the actual privacy deliverable.

- `blur_regions` returns a blurred copy of the image and never edits the input in place, so originals stay untouched for audits and retraining.
- Gaussian blur uses a kernel that scales with the box (about half its longest side), so large close-up faces are blurred as strongly as small distant ones.
- Pixelation is the alternative: it squeezes each region to at most 8x8 blocks, which leaves too little information to read a plate.
- Box coordinates are rounded outward and clipped to the image, and regions under 2 px are skipped, so edge cases can't crash a batch job.
- Boxes are padded before blurring (`boxes.pad_and_clip`, `inference.box_padding`) so hair and plate edges are covered too.
