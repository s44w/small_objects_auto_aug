from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.data.coco_small_manager import (
    CocoSmallPrepareConfig,
    prepare_coco_small_by_mode,
    prepare_coco_small_yolo,
)


def _write_image(path: Path, width: int = 64, height: int = 48) -> None:
    # For conversion tests we only need files with valid names, not decodable pixels.
    # The converter uses metadata from COCO JSON for width/height.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xd8\xff\xdb\x00C\x00" + bytes([0] * 32) + b"\xff\xd9")


def _write_instances_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_prepare_coco_small_yolo_keeps_only_small_boxes() -> None:
    root = Path("artifacts") / "_test_tmp" / "coco_small_manager_case"
    raw_root = root / "raw"
    output_root = root / "yolo"
    shutil.rmtree(root, ignore_errors=True)

    _write_image(raw_root / "train2017" / "000000000001.jpg")
    _write_image(raw_root / "train2017" / "000000000002.jpg")
    _write_image(raw_root / "val2017" / "000000000003.jpg")

    categories = [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}]
    train_payload = {
        "images": [
            {"id": 1, "file_name": "000000000001.jpg", "width": 64, "height": 48},
            {"id": 2, "file_name": "000000000002.jpg", "width": 64, "height": 48},
        ],
        "categories": categories,
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 1, "bbox": [5, 5, 10, 8], "area": 80, "iscrowd": 0},
            {"id": 11, "image_id": 2, "category_id": 3, "bbox": [2, 2, 50, 40], "area": 2000, "iscrowd": 0},
        ],
    }
    val_payload = {
        "images": [{"id": 3, "file_name": "000000000003.jpg", "width": 64, "height": 48}],
        "categories": categories,
        "annotations": [
            {"id": 12, "image_id": 3, "category_id": 3, "bbox": [8, 8, 12, 8], "area": 96, "iscrowd": 0},
        ],
    }
    _write_instances_json(raw_root / "annotations" / "instances_train2017.json", train_payload)
    _write_instances_json(raw_root / "annotations" / "instances_val2017.json", val_payload)

    report = prepare_coco_small_yolo(
        raw_coco_root=raw_root,
        output_root=output_root,
        config=CocoSmallPrepareConfig(
            small_max_area=1024.0,
            keep_images_without_small=False,
            include_crowd=False,
            link_images=False,
        ),
    )

    assert report.is_valid is True
    assert report.class_names == ["person", "car"]
    assert report.splits["train"]["num_images"] == 1
    assert report.splits["val"]["num_images"] == 1
    label_train = (output_root / "labels" / "train" / "000000000001.txt").read_text(encoding="utf-8")
    assert label_train.startswith("0 ")
    label_val = (output_root / "labels" / "val" / "000000000003.txt").read_text(encoding="utf-8")
    assert label_val.startswith("1 ")
    assert (output_root / "coco_small.yaml").exists()


def test_prepare_coco_small_by_mode_manual_validates_existing_dataset() -> None:
    root = Path("artifacts") / "_test_tmp" / "coco_small_manual_case"
    shutil.rmtree(root, ignore_errors=True)
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    _write_image(root / "images" / "train" / "sample_train.jpg")
    _write_image(root / "images" / "val" / "sample_val.jpg")
    (root / "labels" / "train" / "sample_train.txt").write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")
    (root / "labels" / "val" / "sample_val.txt").write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")

    report = prepare_coco_small_by_mode(
        dataset_root=root,
        mode="manual",
        config=CocoSmallPrepareConfig(link_images=False),
    )
    assert report["is_valid"] is True
    assert report["summary"]["mode"] == "manual"
