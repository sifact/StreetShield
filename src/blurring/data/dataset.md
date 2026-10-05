# dataset.py

PyTorch datasets that feed the detector during training/evaluation and the batch blurring pipeline.

- `DetectionDataset` turns a `Record` into `(image_tensor, {"boxes", "labels", "image_id"})`, the format torchvision detectors train on.
- Class ids start at 1 because the two-stage detector reserves 0 for background.
- `load_rgb` converts RGBA and palette PNGs to 3-channel RGB, since some source images are not RGB.
- `ImagePathDataset` is the CPU preprocessing step of the serving pipeline: it decodes images in DataLoader workers and keeps the original pixels for blurring.
- `collate_detection` keeps variable-sized images as tuples rather than stacking them into one tensor.
