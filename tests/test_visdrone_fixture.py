from pathlib import Path
import shutil

from src.data.visdrone_fixture import (
    VISDRONE_CLASS_NAMES,
    create_predictions_from_gt,
    create_visdrone_tiny_fixture,
)
from src.data.visdrone_manager import validate_visdrone_yolo_structure


def test_create_visdrone_tiny_fixture_and_prediction_labels() -> None:
    tmp_root = Path("artifacts") / "_test_tmp" / "visdrone_fixture_case"
    shutil.rmtree(tmp_root, ignore_errors=True)

    dataset_root = create_visdrone_tiny_fixture(tmp_root / "dataset", train_images=3, val_images=2)
    report = validate_visdrone_yolo_structure(dataset_root, splits=("train", "val"))

    assert report["is_valid"] is True
    assert report["splits"]["train"]["num_images"] == 3
    assert report["splits"]["val"]["num_images"] == 2
    assert len(VISDRONE_CLASS_NAMES) == 10

    pred_dir = create_predictions_from_gt(dataset_root / "labels" / "val", tmp_root / "preds")
    pred_files = sorted(pred_dir.glob("*.txt"))
    assert len(pred_files) == 2
    first_line = pred_files[0].read_text(encoding="utf-8").splitlines()[0]
    assert len(first_line.split()) == 6
