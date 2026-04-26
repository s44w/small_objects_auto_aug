from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("albumentations")

from src.augmentation.albumentations_transforms import (
    BBoxAwareCropTransform,
    BBoxCopyPasteTransform,
    build_custom_transforms,
)


def _assert_bbox_in_bounds(box: list[float], width: int, height: int) -> None:
    x1, y1, x2, y2 = box
    assert 0.0 <= x1 <= float(width)
    assert 0.0 <= y1 <= float(height)
    assert 0.0 <= x2 <= float(width)
    assert 0.0 <= y2 <= float(height)
    assert x2 > x1
    assert y2 > y1


def test_bbox_aware_crop_transform_fallback_when_crop_returns_empty() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    sample = {
        "image": image,
        "bboxes": [[10.0, 10.0, 30.0, 30.0]],
        "class_labels": [2],
    }
    transform = BBoxAwareCropTransform(height=48, width=48, p=1.0, seed=42)

    # Force empty crop result to verify fallback to original sample.
    transform._crop = lambda **kwargs: {  # type: ignore[assignment]
        "image": kwargs["image"],
        "bboxes": [],
        "class_labels": [],
    }

    out = transform(sample)
    assert out["bboxes"] == sample["bboxes"]
    assert out["class_labels"] == sample["class_labels"]
    assert out["image"].shape == sample["image"].shape


def test_bbox_copy_paste_respects_preferred_classes_and_bbox_integrity() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    sample = {
        "image": image,
        "bboxes": [
            [5.0, 5.0, 15.0, 15.0],    # class 0
            [80.0, 80.0, 90.0, 90.0],  # class 1
        ],
        "class_labels": [0, 1],
    }

    transform = BBoxCopyPasteTransform(
        p=1.0,
        max_pastes=1,
        max_trials=100,
        prefer_small=True,
        preferred_classes={1},
        seed=42,
    )
    out = transform(sample)

    assert len(out["bboxes"]) == len(out["class_labels"])
    assert len(out["bboxes"]) >= len(sample["bboxes"])
    assert out["class_labels"][-1] == 1
    for box in out["bboxes"]:
        _assert_bbox_in_bounds(box, width=100, height=100)


def test_build_custom_transforms_passes_tail_class_ids() -> None:
    spec = [
        {
            "name": "BBoxCopyPaste",
            "p": 0.3,
            "params": {
                "max_pastes": 2,
                "tail_class_ids": [3, 7],
            },
        }
    ]
    transforms = build_custom_transforms(spec, seed=7)
    assert len(transforms) == 1
    transform = transforms[0]
    assert isinstance(transform, BBoxCopyPasteTransform)
    assert transform.preferred_classes == {3, 7}
