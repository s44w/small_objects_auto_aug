from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import cv2

from src.data.yolo_label_reader import load_yolo_labels, yolo_bbox_to_xywh_px
from src.utils.io import dump_json


def _list_images(images_dir: Path, image_extensions: Iterable[str]) -> list[Path]:
    extensions = {ext.lower() for ext in image_extensions}
    return sorted(
        [
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        ]
    )


def _clip_coco_bbox(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    x = max(0.0, min(float(img_w), x))
    y = max(0.0, min(float(img_h), y))
    w = max(0.0, min(float(img_w) - x, w))
    h = max(0.0, min(float(img_h) - y, h))
    return x, y, w, h


def convert_yolo_gt_to_coco(
    images_dir: str | Path,
    labels_dir: str | Path,
    class_names: list[str],
    output_path: str | Path,
    image_extensions: Iterable[str] = (".jpg", ".jpeg", ".png", ".bmp"),
) -> dict:
    """
    Convert YOLO ground-truth annotations to COCO JSON.

    category_id in COCO starts from 1, so YOLO class_id is shifted by +1.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    images = []
    annotations = []
    categories = [{"id": idx + 1, "name": name} for idx, name in enumerate(class_names)]

    image_id_by_stem: dict[str, int] = {}
    ann_id = 1

    for image_id, image_path in enumerate(_list_images(images_dir, image_extensions), start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": int(w),
                "height": int(h),
            }
        )
        image_id_by_stem[image_path.stem] = image_id

        label_path = labels_dir / f"{image_path.stem}.txt"
        bboxes = load_yolo_labels(label_path)
        for bbox in bboxes:
            x, y, bw, bh = yolo_bbox_to_xywh_px(bbox=bbox, image_width=w, image_height=h)
            x, y, bw, bh = _clip_coco_bbox(x, y, bw, bh, img_w=w, img_h=h)
            if bw <= 0 or bh <= 0:
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(bbox.class_id) + 1,
                    "bbox": [float(x), float(y), float(bw), float(bh)],
                    "area": float(bw * bh),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    coco_payload = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    dump_json(coco_payload, output_path)
    return coco_payload


def convert_yolo_pred_txt_to_coco(
    pred_labels_dir: str | Path,
    images_dir: str | Path,
    output_path: str | Path,
    image_extensions: Iterable[str] = (".jpg", ".jpeg", ".png", ".bmp"),
) -> list[dict]:
    """
    Convert YOLO prediction txt files into COCO detections JSON.

    Expected line formats:
    - class x_center y_center w h
    - class x_center y_center w h score
    """
    pred_labels_dir = Path(pred_labels_dir)
    images_dir = Path(images_dir)

    image_id_by_stem: dict[str, tuple[int, int, int]] = {}
    for image_id, image_path in enumerate(_list_images(images_dir, image_extensions), start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        image_id_by_stem[image_path.stem] = (image_id, w, h)

    detections: list[dict] = []

    for txt_path in sorted(pred_labels_dir.glob("*.txt")):
        if txt_path.stem not in image_id_by_stem:
            continue
        image_id, img_w, img_h = image_id_by_stem[txt_path.stem]

        with txt_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                chunks = line.split()
                if len(chunks) < 5:
                    continue
                class_id = int(float(chunks[0]))
                x_center = float(chunks[1])
                y_center = float(chunks[2])
                width = float(chunks[3])
                height = float(chunks[4])
                score = float(chunks[5]) if len(chunks) > 5 else 1.0

                bw = width * img_w
                bh = height * img_h
                x = (x_center * img_w) - bw / 2.0
                y = (y_center * img_h) - bh / 2.0
                x, y, bw, bh = _clip_coco_bbox(x, y, bw, bh, img_w=img_w, img_h=img_h)
                if bw <= 0 or bh <= 0:
                    continue

                detections.append(
                    {
                        "image_id": image_id,
                        "category_id": class_id + 1,
                        "bbox": [float(x), float(y), float(bw), float(bh)],
                        "score": float(score),
                    }
                )

    dump_json(detections, output_path)
    return detections


def convert_prediction_json_to_coco(
    prediction_json_path: str | Path,
    output_path: str | Path,
) -> list[dict]:
    """
    Normalize a prediction json into COCO detections list.

    Input can be:
    - already COCO list[dict]
    - wrapper {"detections": list[dict]}
    """
    with Path(prediction_json_path).open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, dict) and "detections" in payload:
        detections = payload["detections"]
    elif isinstance(payload, list):
        detections = payload
    else:
        raise ValueError("Unsupported prediction JSON format.")

    dump_json(detections, output_path)
    return detections

