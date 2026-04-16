from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.evaluation.coco_converter import convert_yolo_gt_to_coco, convert_yolo_pred_txt_to_coco


def test_yolo_to_coco_gt_and_predictions(tmp_path: Path):
    images_dir = tmp_path / "images" / "val"
    labels_dir = tmp_path / "labels" / "val"
    pred_dir = tmp_path / "preds"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    pred_dir.mkdir(parents=True)

    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image_path = images_dir / "img_001.jpg"
    cv2.imwrite(str(image_path), image)

    (labels_dir / "img_001.txt").write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
    (pred_dir / "img_001.txt").write_text("0 0.5 0.5 0.2 0.4 0.9\n", encoding="utf-8")

    gt_output = tmp_path / "coco_gt.json"
    dt_output = tmp_path / "coco_dt.json"

    gt_payload = convert_yolo_gt_to_coco(
        images_dir=images_dir,
        labels_dir=labels_dir,
        class_names=["class0"],
        output_path=gt_output,
    )
    dt_payload = convert_yolo_pred_txt_to_coco(
        pred_labels_dir=pred_dir,
        images_dir=images_dir,
        output_path=dt_output,
    )

    assert len(gt_payload["images"]) == 1
    assert len(gt_payload["annotations"]) == 1
    assert abs(gt_payload["annotations"][0]["area"] - 1600.0) < 1e-6
    assert gt_payload["annotations"][0]["category_id"] == 1

    assert len(dt_payload) == 1
    assert dt_payload[0]["image_id"] == 1
    assert dt_payload[0]["category_id"] == 1
    assert abs(dt_payload[0]["score"] - 0.9) < 1e-9
