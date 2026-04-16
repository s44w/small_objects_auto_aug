from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class YoloBBox:
    """A single YOLO normalized bounding box."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


def parse_yolo_label_line(line: str, line_number: int | None = None) -> YoloBBox:
    """
    Parse a YOLO line with format: class_id x_center y_center width height.

    Values are expected to be normalized to [0, 1].
    """
    chunks = line.strip().split()
    if len(chunks) < 5:
        suffix = f" at line {line_number}" if line_number is not None else ""
        raise ValueError(f"Invalid YOLO label format{suffix}: '{line.strip()}'")

    class_id = int(float(chunks[0]))
    x_center, y_center, width, height = map(float, chunks[1:5])
    return YoloBBox(
        class_id=class_id,
        x_center=x_center,
        y_center=y_center,
        width=width,
        height=height,
    )


def load_yolo_labels(path: str | Path) -> list[YoloBBox]:
    """Load YOLO labels from a txt file. Missing files are treated as empty labels."""
    label_path = Path(path)
    if not label_path.exists():
        return []

    boxes: list[YoloBBox] = []
    with label_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            clean = line.strip()
            if not clean:
                continue
            boxes.append(parse_yolo_label_line(clean, line_number=line_number))
    return boxes


def yolo_bbox_to_xywh_px(
    bbox: YoloBBox, image_width: int, image_height: int
) -> tuple[float, float, float, float]:
    """Convert normalized YOLO bbox to absolute COCO-style xywh in pixels."""
    width_px = bbox.width * image_width
    height_px = bbox.height * image_height
    x_min = (bbox.x_center * image_width) - width_px / 2.0
    y_min = (bbox.y_center * image_height) - height_px / 2.0
    return x_min, y_min, width_px, height_px


def yolo_bbox_to_xyxy_px(
    bbox: YoloBBox, image_width: int, image_height: int
) -> tuple[float, float, float, float]:
    """Convert normalized YOLO bbox to absolute xyxy in pixels."""
    x_min, y_min, width_px, height_px = yolo_bbox_to_xywh_px(
        bbox=bbox,
        image_width=image_width,
        image_height=image_height,
    )
    x_max = x_min + width_px
    y_max = y_min + height_px
    return x_min, y_min, x_max, y_max


def yolo_bbox_area_px(bbox: YoloBBox, image_width: int, image_height: int) -> float:
    """Return bbox area in pixel units."""
    _, _, width_px, height_px = yolo_bbox_to_xywh_px(
        bbox=bbox,
        image_width=image_width,
        image_height=image_height,
    )
    return max(0.0, width_px) * max(0.0, height_px)

