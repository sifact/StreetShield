# Data

Two small public datasets from GitHub, merged into two classes (`face`, `plate`). Both are Pascal VOC XML.

| Source | Repo | Images | Our class | License |
|---|---|---|---|---|
| Faces | [iAmEthanMai/mask-detection-dataset](https://github.com/iAmEthanMai/mask-detection-dataset) (mirror of Kaggle "Face Mask Detection") | 853 | `with_mask`, `without_mask`, `mask_weared_incorrect` → `face` | MIT (original CC0) |
| Plates | [RobertLucian/license-plate-dataset](https://github.com/RobertLucian/license-plate-dataset) (Romanian dashcam frames, 1280x720) | 534 | `license-plate` → `plate` | MIT |

Download (sparse clones, about 900 MB on disk):

```bash
mkdir -p data/raw && cd data/raw
git clone --depth 1 --filter=blob:none --sparse https://github.com/iAmEthanMai/mask-detection-dataset.git faces
(cd faces && git sparse-checkout set data/xml data/images)
git clone --depth 1 --filter=blob:none --sparse https://github.com/RobertLucian/license-plate-dataset.git plates
(cd plates && git sparse-checkout set dataset/train dataset/valid)
```

Then `uv run streetshield prepare-data` writes `data/processed/manifest.json` with seeded 70/15/15 splits per source.

Known limitations:
- About 85% of the faces wear masks, so lower-face appearance is skewed.
- The plates are European, so other plate formats may be detected less well.
- Neither set is real Street View imagery. The chapter's fairness and bias concerns apply and are not measured here.
