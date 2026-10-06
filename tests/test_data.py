import json

import numpy as np
import torch
from PIL import Image

from streetshield.config import AugmentConfig
from streetshield.data import (
    DetectionDataset,
    Record,
    TrainTransform,
    build_retraining_set,
    find_hard_examples,
    load_manifest,
    load_reports,
    parse_voc,
    save_manifest,
)
from streetshield.data.manifest import split_records
from streetshield.data.transforms import crop_keeping_boxes, hflip

VOC = """<annotation><filename>{name}</filename><size><width>100</width><height>80</height></size>
<object><name>with_mask</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>30.5</xmax><ymax>40</ymax></bndbox></object>
<object><name>license-plate</name><bndbox><xmin>50</xmin><ymin>60</ymin><xmax>120</xmax><ymax>70</ymax></bndbox></object>
<object><name>dog</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>5</xmax><ymax>5</ymax></bndbox></object>
<object><name>without_mask</name><bndbox><xmin>5</xmin><ymin>5</ymin><xmax>5.5</xmax><ymax>9</ymax></bndbox></object>
</annotation>"""
LABEL_MAP = {"with_mask": "face", "without_mask": "face", "license-plate": "plate"}


def _write_sample(tmp_path, name="clip.mp4#t=1.jpg"):
    Image.new("RGBA", (100, 80), (200, 100, 50, 255)).save(tmp_path / name, format="PNG")
    xml = tmp_path / "a.xml"
    xml.write_text(VOC.format(name=name))
    return xml


def test_parse_voc_maps_clips_and_drops(tmp_path):
    record = parse_voc(_write_sample(tmp_path), tmp_path, LABEL_MAP, "test")
    assert record.labels == ["face", "plate"]  # dog unmapped, sliver box dropped
    assert record.boxes[1] == [50.0, 60.0, 100.0, 70.0]  # clipped to width
    assert record.image_path.endswith("clip.mp4#t=1.jpg")


def test_parse_voc_missing_image_returns_none(tmp_path):
    xml = tmp_path / "b.xml"
    xml.write_text(VOC.format(name="missing.jpg"))
    assert parse_voc(xml, tmp_path, LABEL_MAP) is None


def test_dataset_item_is_rgb_with_1_based_labels(tmp_path):
    record = parse_voc(_write_sample(tmp_path), tmp_path, LABEL_MAP)
    image, target = DetectionDataset([record], ["face", "plate"], TrainTransform(AugmentConfig()))[0]
    assert image.shape[0] == 3
    assert target["labels"].tolist() == [1, 2]
    assert target["boxes"].shape == (2, 4)


def test_hflip_mirrors_boxes():
    _, boxes = hflip(Image.new("RGB", (100, 50)), np.array([[10, 0, 30, 10]], dtype=np.float32))
    np.testing.assert_allclose(boxes, [[70, 0, 90, 10]])


def test_crop_keeps_every_box_inside():
    torch.manual_seed(0)
    boxes = np.array([[20, 20, 40, 40], [60, 10, 90, 30]], dtype=np.float32)
    for _ in range(50):
        image, out = crop_keeping_boxes(Image.new("RGB", (120, 100)), boxes)
        assert (out[:, :2] >= 0).all() and (out[:, 2] <= image.width).all() and (out[:, 3] <= image.height).all()
        np.testing.assert_allclose(out[:, 2:] - out[:, :2], boxes[:, 2:] - boxes[:, :2])


def test_split_is_deterministic_and_disjoint():
    records = [Record(f"img{i}.jpg", 1, 1) for i in range(100)]
    a, b = split_records(records, 0.15, 0.15, 42), split_records(records, 0.15, 0.15, 42)
    assert [r.image_path for r in a["test"]] == [r.image_path for r in b["test"]]
    assert sum(len(v) for v in a.values()) == 100 and len(a["test"]) == 15


def test_manifest_roundtrip_uses_relative_paths(tmp_path):
    manifest = {"train": [Record(str(tmp_path / "x.jpg"), 10, 10, [[0, 0, 5, 5]], ["face"])], "val": [], "test": []}
    save_manifest(manifest, tmp_path / "m.json", root=tmp_path)
    assert json.loads((tmp_path / "m.json").read_text())["train"][0]["image_path"] == "x.jpg"
    assert load_manifest(tmp_path / "m.json", root=tmp_path)["train"][0].image_path == str(tmp_path / "x.jpg")


def test_reports_and_hard_examples(tmp_path):
    Image.new("RGB", (64, 48)).save(tmp_path / "r.jpg")
    (tmp_path / "reports.jsonl").write_text(
        json.dumps({"image_path": str(tmp_path / "r.jpg"), "boxes": [[1, 2, 10, 20]], "labels": ["face"]}) + "\n"
    )
    reports = load_reports(tmp_path / "reports.jsonl")
    assert reports[0].boxes == [[1, 2, 11, 22]] and reports[0].width == 64

    records = [Record("a", 1, 1, [[0, 0, 10, 10]], ["face"]), Record("b", 1, 1, [[0, 0, 10, 10]], ["face"])]
    preds = [
        {"boxes": np.array([[0, 0, 10, 10]], np.float32), "scores": np.array([0.9]), "labels": np.array([1])},
        {"boxes": np.array([[50, 50, 60, 60]], np.float32), "scores": np.array([0.9]), "labels": np.array([1])},
    ]
    assert find_hard_examples(records, preds, ["face", "plate"], 0.5, 0.3) == [1]
    assert len(build_retraining_set(records, reports, [records[1]], oversample=3)) == 6
