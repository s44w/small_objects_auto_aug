from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np


VISDRONE_CLASS_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]


def _yolo_line(class_id: int, x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> str:
    xc = ((x1 + x2) / 2.0) / width
    yc = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"


def create_visdrone_tiny_fixture(
    output_root: str | Path,
    train_images: int = 12,
    val_images: int = 4,
    width: int = 320,
    height: int = 240,
    clean_output: bool = True,
) -> Path:
    """
    Create a tiny VisDrone-like YOLO dataset for fast module smoke tests.

    The fixture uses the VisDrone class list and deliberately small boxes, so
    analyzer/rule/eval paths exercise the same small-object logic as real data.
    """
    output_root = Path(output_root)
    if clean_output and output_root.exists():
        shutil.rmtree(output_root, ignore_errors=True)
    output_root.mkdir(parents=True, exist_ok=True)

    counts = {"train": int(train_images), "val": int(val_images)}
    for split, num_images in counts.items():
        images_dir = output_root / "images" / split
        labels_dir = output_root / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(num_images):
            image = np.zeros((height, width, 3), dtype=np.uint8)
            image[:, :, 0] = 30 + (idx * 7) % 80
            image[:, :, 1] = 40 + (idx * 11) % 100
            image[:, :, 2] = 60 + (idx * 13) % 120

            lines: list[str] = []
            num_objects = 4 + (idx % 3)
            for obj_idx in range(num_objects):
                box_w = 8 + ((idx + obj_idx) % 4) * 4
                box_h = 10 + ((idx + 2 * obj_idx) % 4) * 4
                x1 = 12 + ((idx * 23 + obj_idx * 47) % max(1, width - box_w - 24))
                y1 = 10 + ((idx * 17 + obj_idx * 31) % max(1, height - box_h - 20))
                x2 = x1 + box_w
                y2 = y1 + box_h
                class_id = (idx + obj_idx) % len(VISDRONE_CLASS_NAMES)
                color = (40 + class_id * 17, 180 - class_id * 11, 80 + class_id * 13)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=-1)
                lines.append(_yolo_line(class_id, x1, y1, x2, y2, width=width, height=height))

            stem = f"{split}_{idx:04d}"
            cv2.imwrite(str(images_dir / f"{stem}.jpg"), image)
            (labels_dir / f"{stem}.txt").write_text("".join(lines), encoding="utf-8")

    data_yaml = {
        "path": output_root.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": {idx: name for idx, name in enumerate(VISDRONE_CLASS_NAMES)},
    }
    try:
        import yaml

        (output_root / "visdrone_fixture.yaml").write_text(
            yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    except Exception:
        pass

    return output_root


def create_predictions_from_gt(
    labels_dir: str | Path,
    output_dir: str | Path,
    confidence: float = 0.99,
    clean_output: bool = True,
) -> Path:
    """
    Convert YOLO GT labels into YOLO prediction txt files with confidence.

    This makes evaluator smoke tests deterministic and very fast.
    """
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    if clean_output and output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    for label_path in sorted(labels_dir.glob("*.txt")):
        pred_lines: list[str] = []
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            parts = raw_line.split()
            if len(parts) < 5:
                continue
            pred_lines.append(" ".join(parts[:5] + [f"{confidence:.6f}"]) + "\n")
        (output_dir / label_path.name).write_text("".join(pred_lines), encoding="utf-8")

    return output_dir
