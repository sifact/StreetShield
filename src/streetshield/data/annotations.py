"""Parse Pascal VOC annotations from heterogeneous sources into one record format."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
MIN_BOX_SIDE = 1.0  # pixels; thinner boxes are annotation noise and break the box regression targets


@dataclass
class Record:
    """One annotated image. Boxes are ``[x1, y1, x2, y2]`` in pixels; labels are our class names."""

    image_path: str
    width: int
    height: int
    boxes: list[list[float]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    source: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Record":
        return cls(**d)


def _find_image(xml_path: Path, filename: str | None, images_dir: Path) -> Path | None:
    """Prefer the filename stored in the XML; fall back to the XML stem with any image extension."""
    if filename and (images_dir / filename).exists():
        return images_dir / filename
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{xml_path.stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def parse_voc(xml_path: Path, images_dir: Path, label_map: dict[str, str], source: str = "") -> Record | None:
    """Read one VOC XML file, map raw class names to ours and drop objects we don't blur.

    Returns ``None`` when the image file is missing, so a partially downloaded dataset
    degrades to fewer samples instead of crashing training later.
    """
    root = ET.parse(xml_path).getroot()
    image_path = _find_image(xml_path, root.findtext("filename"), images_dir)
    if image_path is None:
        logger.warning("No image for %s; skipping", xml_path.name)
        return None

    width = int(float(root.findtext("size/width", "0")))
    height = int(float(root.findtext("size/height", "0")))
    record = Record(image_path=str(image_path), width=width, height=height, source=source)

    for obj in root.iter("object"):
        label = label_map.get((obj.findtext("name") or "").strip())
        if label is None:
            continue
        bb = obj.find("bndbox")
        x1, y1, x2, y2 = (float(bb.findtext(k, "0")) for k in ("xmin", "ymin", "xmax", "ymax"))
        if width and height:
            x1, x2 = max(0.0, x1), min(float(width), x2)
            y1, y2 = max(0.0, y1), min(float(height), y2)
        if x2 - x1 < MIN_BOX_SIDE or y2 - y1 < MIN_BOX_SIDE:
            continue
        record.boxes.append([x1, y1, x2, y2])
        record.labels.append(label)
    return record


def load_voc_dir(annots_dir: Path, images_dir: Path, label_map: dict[str, str], source: str) -> list[Record]:
    """Parse every XML file in a directory, sorted so the order (and therefore the split) is stable."""
    records = [parse_voc(p, images_dir, label_map, source) for p in sorted(annots_dir.glob("*.xml"))]
    return [r for r in records if r is not None]
