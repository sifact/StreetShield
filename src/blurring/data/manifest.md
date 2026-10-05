# manifest.py

Merges every annotation source into one JSON manifest with reproducible train/val/test splits.

- Loads each configured source (faces, plates train, plates valid) through `annotations.load_voc_dir`.
- Splits each source separately with a seeded shuffle, so the face/plate ratio is the same in train, val and test.
- Writes image paths relative to the project root, so the manifest stays valid if the repo moves.
- `summarize` reports image and per-class box counts per split, which is the first sanity check after `prepare-data`.
- `load_manifest` is the single entry point that training, evaluation and mining use to get their records.
