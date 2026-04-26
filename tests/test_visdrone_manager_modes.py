from pathlib import Path
import shutil

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.data.visdrone_manager import prepare_dataset_by_mode


def _write_image(path: Path, width: int = 32, height: int = 24) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _write_dummy_yolo_pair(root: Path, split: str, stem: str = "000001") -> None:
    image_path = root / "images" / split / f"{stem}.jpg"
    label_path = root / "labels" / split / f"{stem}.txt"
    _write_image(image_path)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")


def _write_dummy_raw_split(raw_root: Path, split_dir_name: str, stem: str) -> None:
    split_root = raw_root / split_dir_name
    image_path = split_root / "images" / f"{stem}.jpg"
    ann_path = split_root / "annotations" / f"{stem}.txt"
    _write_image(image_path)
    ann_path.parent.mkdir(parents=True, exist_ok=True)
    # x,y,w,h,score,category,truncation,occlusion
    ann_path.write_text("2,2,10,8,1,1,0,0\n", encoding="utf-8")


def test_prepare_dataset_by_mode_manual_validates_existing_yolo() -> None:
    dataset_root = Path("artifacts") / "_test_tmp" / "visdrone_mode_manual_case" / "visdrone_yolo"
    shutil.rmtree(dataset_root.parent, ignore_errors=True)
    _write_dummy_yolo_pair(dataset_root, split="train", stem="train_001")
    _write_dummy_yolo_pair(dataset_root, split="val", stem="val_001")

    report = prepare_dataset_by_mode(dataset_root=dataset_root, mode="manual")
    assert report["is_valid"] is True
    assert report["summary"]["mode"] == "manual"
    assert report["summary"]["train_images"] == 1
    assert report["summary"]["val_images"] == 1


def test_prepare_dataset_by_mode_auto_converts_raw_to_yolo() -> None:
    tmp_path = Path("artifacts") / "_test_tmp" / "visdrone_mode_auto_case"
    shutil.rmtree(tmp_path, ignore_errors=True)
    raw_root = tmp_path / "visdrone_raw"
    dataset_root = tmp_path / "visdrone_yolo"

    _write_dummy_raw_split(raw_root, "VisDrone2019-DET-train", stem="train_001")
    _write_dummy_raw_split(raw_root, "VisDrone2019-DET-val", stem="val_001")

    report = prepare_dataset_by_mode(
        dataset_root=dataset_root,
        mode="auto",
        raw_visdrone_root=raw_root,
    )

    assert report["is_valid"] is True
    assert report["summary"]["mode"] == "auto"
    assert report["summary"]["train_images"] >= 1
    assert report["summary"]["val_images"] >= 1
    assert (dataset_root / "visdrone.yaml").exists()
