from __future__ import annotations

import random
import inspect
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.augmentation.object_bank import ObjectBank

try:
    import albumentations as A
except Exception:
    A = None


def _clip_bbox_xyxy(bbox: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = bbox
    x1 = float(max(0.0, min(float(width - 1), x1)))
    y1 = float(max(0.0, min(float(height - 1), y1)))
    x2 = float(max(0.0, min(float(width), x2)))
    y2 = float(max(0.0, min(float(height), y2)))
    return [x1, y1, x2, y2]


def _bbox_area(bbox: list[float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    union = _bbox_area(a) + _bbox_area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def _ioa_xyxy(a: list[float], b: list[float]) -> float:
    """
    Intersection over area of bbox a.
    Useful for copy-paste placement constraints.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    denom = _bbox_area(a)
    if denom <= 0:
        return 0.0
    return inter / denom


@dataclass
class BBoxAwareCropTransform:
    """
    Wrapper over Albumentations RandomSizedBBoxSafeCrop with explicit fallback.

    If crop removes all bboxes (which is undesirable for this MVP), we keep original sample.
    """

    height: int
    width: int
    min_visibility: float = 0.3
    min_area: float = 16.0
    p: float = 0.5
    seed: int = 42

    def __post_init__(self) -> None:
        if A is None:
            raise ImportError("albumentations is required for BBoxAwareCropTransform")
        self._rng = random.Random(self.seed)
        bbox_params_kwargs = {
            "format": "pascal_voc",
            "label_fields": ["class_labels"],
            "min_visibility": self.min_visibility,
            "min_area": self.min_area,
        }
        if "clip" in inspect.signature(A.BboxParams).parameters:
            bbox_params_kwargs["clip"] = True
        self._crop = A.Compose(
            [A.RandomSizedBBoxSafeCrop(height=self.height, width=self.width, p=1.0)],
            bbox_params=A.BboxParams(**bbox_params_kwargs),
        )

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if self._rng.random() > self.p:
            return sample
        if len(sample.get("bboxes", [])) == 0:
            return sample

        transformed = self._crop(
            image=sample["image"],
            bboxes=sample["bboxes"],
            class_labels=sample["class_labels"],
        )

        # Fallback to source sample when crop became empty.
        if len(transformed["bboxes"]) == 0:
            return sample

        return sanitize_bboxes(
            {
            "image": transformed["image"],
            "bboxes": [list(map(float, bbox)) for bbox in transformed["bboxes"]],
            "class_labels": [int(cls_id) for cls_id in transformed["class_labels"]],
            },
            min_area=self.min_area,
        )


@dataclass
class BBoxCopyPasteTransform:
    """
    BBox-only copy-paste augmentation for detect task (without masks).

    Sample format:
      {
        "image": np.ndarray(H,W,3),
        "bboxes": list[[x1,y1,x2,y2]],  # absolute pixels
        "class_labels": list[int]
      }
    """

    p: float = 0.2
    ioa_threshold: float = 0.3
    iou_threshold: float = 0.3
    max_pastes: int = 3
    max_trials: int = 20
    prefer_small: bool = False
    preferred_classes: set[int] | None = None
    small_max_area: float = 32.0**2
    seed: int = 42
    object_bank: ObjectBank | None = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        if self.preferred_classes is not None:
            self.preferred_classes = {int(class_id) for class_id in self.preferred_classes}

    def _sample_donor_from_sample(
        self,
        image: np.ndarray,
        bboxes: list[list[float]],
        class_labels: list[int],
    ) -> tuple[np.ndarray, int] | None:
        if not bboxes:
            return None

        indices = list(range(len(bboxes)))
        if self.preferred_classes:
            class_filtered = [idx for idx, class_id in enumerate(class_labels) if class_id in self.preferred_classes]
            if class_filtered:
                indices = class_filtered

        if self.prefer_small:
            small_indices = [idx for idx, box in enumerate(bboxes) if _bbox_area(box) <= self.small_max_area]
            if small_indices:
                indices = [idx for idx in small_indices if idx in indices] or small_indices

        donor_idx = self._rng.choice(indices)
        x1, y1, x2, y2 = [int(round(v)) for v in bboxes[donor_idx]]
        x1 = max(0, min(image.shape[1] - 1, x1))
        y1 = max(0, min(image.shape[0] - 1, y1))
        x2 = max(0, min(image.shape[1], x2))
        y2 = max(0, min(image.shape[0], y2))
        if x2 <= x1 or y2 <= y1:
            return None

        patch = image[y1:y2, x1:x2].copy()
        if patch.size == 0:
            return None
        donor_class = int(class_labels[donor_idx])
        return patch, donor_class

    def _sample_donor(
        self,
        image: np.ndarray,
        bboxes: list[list[float]],
        class_labels: list[int],
    ) -> tuple[np.ndarray, int] | None:
        # Prefer object bank when available and non-empty.
        if self.object_bank is not None and self.object_bank.size > 0:
            entry = self.object_bank.sample_entry(
                preferred_classes=self.preferred_classes,
                prefer_small=self.prefer_small,
            )
            if entry is not None:
                patch = self.object_bank.extract_patch(entry)
                if patch is not None:
                    return patch, int(entry.class_id)

        return self._sample_donor_from_sample(image=image, bboxes=bboxes, class_labels=class_labels)

    def _valid_placement(
        self,
        new_bbox: list[float],
        existing_bboxes: list[list[float]],
    ) -> bool:
        for box in existing_bboxes:
            if _iou_xyxy(new_bbox, box) > self.iou_threshold:
                return False
            if _ioa_xyxy(new_bbox, box) > self.ioa_threshold:
                return False
            if _ioa_xyxy(box, new_bbox) > self.ioa_threshold:
                return False
        return True

    def _place_patch(
        self,
        image: np.ndarray,
        patch: np.ndarray,
        existing_bboxes: list[list[float]],
    ) -> list[float] | None:
        h, w = image.shape[:2]
        ph, pw = patch.shape[:2]
        if ph <= 0 or pw <= 0 or ph >= h or pw >= w:
            return None

        for _ in range(self.max_trials):
            x1 = self._rng.randint(0, max(0, w - pw))
            y1 = self._rng.randint(0, max(0, h - ph))
            new_bbox = [float(x1), float(y1), float(x1 + pw), float(y1 + ph)]
            if self._valid_placement(new_bbox, existing_bboxes):
                image[y1 : y1 + ph, x1 : x1 + pw] = patch
                return new_bbox
        return None

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if self._rng.random() > self.p:
            return sample

        image = sample["image"].copy()
        bboxes = [list(map(float, box)) for box in sample.get("bboxes", [])]
        class_labels = [int(label) for label in sample.get("class_labels", [])]

        for _ in range(self.max_pastes):
            donor = self._sample_donor(image=image, bboxes=bboxes, class_labels=class_labels)
            if donor is None:
                break
            patch, donor_class = donor
            placed_bbox = self._place_patch(image=image, patch=patch, existing_bboxes=bboxes)
            if placed_bbox is None:
                continue
            bboxes.append(placed_bbox)
            class_labels.append(donor_class)

        return {"image": image, "bboxes": bboxes, "class_labels": class_labels}


def sanitize_bboxes(
    sample: dict[str, Any],
    min_area: float = 1.0,
) -> dict[str, Any]:
    """
    Clip bboxes to image bounds and remove invalid boxes.
    """
    image = sample["image"]
    h, w = image.shape[:2]
    clean_boxes: list[list[float]] = []
    clean_labels: list[int] = []

    for box, label in zip(sample.get("bboxes", []), sample.get("class_labels", [])):
        clipped = _clip_bbox_xyxy(list(map(float, box)), width=w, height=h)
        if _bbox_area(clipped) < min_area:
            continue
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            continue
        clean_boxes.append(clipped)
        clean_labels.append(int(label))

    return {"image": image, "bboxes": clean_boxes, "class_labels": clean_labels}


def build_custom_transforms(
    albumentations_spec: list[dict[str, Any]],
    object_bank: ObjectBank | None = None,
    seed: int = 42,
) -> list[Any]:
    """
    Build runtime transform callables from serializable policy spec.
    """
    transforms: list[Any] = []
    for index, item in enumerate(albumentations_spec):
        name = item["name"]
        p = float(item.get("p", 1.0))
        params = dict(item.get("params", {}))

        if name == "BBoxAwareCrop":
            transforms.append(
                BBoxAwareCropTransform(
                    height=int(params.get("height", 640)),
                    width=int(params.get("width", 640)),
                    min_visibility=float(params.get("min_visibility", 0.3)),
                    min_area=float(params.get("min_area", 16.0)),
                    p=p,
                    seed=seed + index,
                )
            )
            continue

        if name == "BBoxCopyPaste":
            transforms.append(
                BBoxCopyPasteTransform(
                    p=p,
                    ioa_threshold=float(params.get("ioa_threshold", 0.3)),
                    iou_threshold=float(params.get("iou_threshold", 0.3)),
                    max_pastes=int(params.get("max_pastes", 3)),
                    prefer_small=bool(params.get("prefer_small", False)),
                    preferred_classes=set(params.get("tail_class_ids", [])) or None,
                    seed=seed + index,
                    object_bank=object_bank,
                )
            )
            continue

        raise ValueError(f"Unsupported custom transform name: '{name}'")
    return transforms


def apply_custom_transforms(
    sample: dict[str, Any],
    transforms: list[Any],
    min_area: float = 1.0,
) -> dict[str, Any]:
    """
    Apply runtime custom transforms sequentially.

    This helper is framework-agnostic and useful for notebook debugging and tests.
    """
    out = {
        "image": sample["image"],
        "bboxes": [list(map(float, box)) for box in sample.get("bboxes", [])],
        "class_labels": [int(label) for label in sample.get("class_labels", [])],
    }
    for transform in transforms:
        out = transform(out)
        out = sanitize_bboxes(out, min_area=min_area)
    return out
