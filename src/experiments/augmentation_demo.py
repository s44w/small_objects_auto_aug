from __future__ import annotations

import inspect
import random
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from src.augmentation.albumentations_transforms import apply_custom_transforms
from src.augmentation.object_bank import ObjectBank
from src.data.yolo_label_reader import load_yolo_labels, yolo_bbox_area_px, yolo_bbox_to_xyxy_px

try:
    import albumentations as A
except Exception:
    A = None


Sample = dict[str, Any]


def _clone_sample(sample: Sample) -> Sample:
    return {
        "image": sample["image"].copy(),
        "bboxes": [list(map(float, box)) for box in sample.get("bboxes", [])],
        "class_labels": [int(label) for label in sample.get("class_labels", [])],
        "image_path": sample.get("image_path"),
    }


def _list_image_paths(
    dataset_root: str | Path,
    split: str = "train",
    image_extensions: Iterable[str] = (".jpg", ".jpeg", ".png", ".bmp"),
) -> list[Path]:
    root = Path(dataset_root) / "images" / split
    ext_set = {ext.lower() for ext in image_extensions}
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in ext_set
    )


def load_yolo_sample(image_path: str | Path, labels_dir: str | Path) -> Sample:
    image_path = Path(image_path)
    labels_dir = Path(labels_dir)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path.as_posix()}")

    height, width = image.shape[:2]
    labels = load_yolo_labels(labels_dir / f"{image_path.stem}.txt")
    bboxes = [list(yolo_bbox_to_xyxy_px(bbox, width, height)) for bbox in labels]
    class_labels = [bbox.class_id for bbox in labels]
    return {
        "image": image,
        "bboxes": bboxes,
        "class_labels": class_labels,
        "image_path": image_path.as_posix(),
    }


def load_split_samples(
    dataset_root: str | Path,
    split: str = "train",
    image_extensions: Iterable[str] = (".jpg", ".jpeg", ".png", ".bmp"),
    limit: int | None = None,
) -> list[Sample]:
    dataset_root = Path(dataset_root)
    labels_dir = dataset_root / "labels" / split
    image_paths = _list_image_paths(
        dataset_root=dataset_root,
        split=split,
        image_extensions=image_extensions,
    )
    if limit is not None:
        image_paths = image_paths[:limit]
    return [load_yolo_sample(path, labels_dir=labels_dir) for path in image_paths]


def pick_demo_image_paths(
    dataset_root: str | Path,
    split: str = "train",
    limit: int = 3,
    image_extensions: Iterable[str] = (".jpg", ".jpeg", ".png", ".bmp"),
) -> list[Path]:
    dataset_root = Path(dataset_root)
    labels_dir = dataset_root / "labels" / split
    candidates: list[tuple[int, Path]] = []
    for image_path in _list_image_paths(dataset_root, split=split, image_extensions=image_extensions):
        label_count = len(load_yolo_labels(labels_dir / f"{image_path.stem}.txt"))
        if label_count <= 0:
            continue
        candidates.append((label_count, image_path))

    if not candidates:
        return []

    candidates.sort(key=lambda item: (item[0], item[1].name))
    if len(candidates) <= limit:
        return [path for _, path in candidates]

    indices = np.linspace(0, len(candidates) - 1, num=limit, dtype=int)
    return [candidates[idx][1] for idx in indices]


def sample_object_summary(
    sample: Sample,
    small_max_area: float,
    tiny_max_area: float,
) -> dict[str, float]:
    image = sample["image"]
    height, width = image.shape[:2]
    total = len(sample.get("bboxes", []))
    small = 0
    tiny = 0
    for bbox, class_id in zip(sample.get("bboxes", []), sample.get("class_labels", [])):
        x1, y1, x2, y2 = bbox
        yolo_like = type("TmpBBox", (), {})()
        yolo_like.class_id = int(class_id)
        yolo_like.x_center = ((x1 + x2) / 2.0) / width
        yolo_like.y_center = ((y1 + y2) / 2.0) / height
        yolo_like.width = max(0.0, x2 - x1) / width
        yolo_like.height = max(0.0, y2 - y1) / height
        area = yolo_bbox_area_px(yolo_like, width, height)
        if area <= float(small_max_area):
            small += 1
        if area <= float(tiny_max_area):
            tiny += 1
    return {
        "num_objects": float(total),
        "small_objects": float(small),
        "tiny_objects": float(tiny),
    }


def _resize_sample(sample: Sample, width: int, height: int) -> Sample:
    image = sample["image"]
    src_h, src_w = image.shape[:2]
    scale_x = float(width) / max(1.0, float(src_w))
    scale_y = float(height) / max(1.0, float(src_h))
    resized = cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)
    bboxes = []
    for x1, y1, x2, y2 in sample.get("bboxes", []):
        bboxes.append([x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y])
    return {
        "image": resized,
        "bboxes": bboxes,
        "class_labels": list(sample.get("class_labels", [])),
        "image_path": sample.get("image_path"),
    }


def build_mosaic_sample(anchor_sample: Sample, donor_samples: list[Sample]) -> Sample:
    donors = donor_samples[:3]
    if len(donors) < 3:
        return _clone_sample(anchor_sample)

    tile_h, tile_w = anchor_sample["image"].shape[:2]
    tiles = [_resize_sample(anchor_sample, tile_w, tile_h)] + [
        _resize_sample(sample, tile_w, tile_h) for sample in donors
    ]

    canvas = np.zeros((tile_h * 2, tile_w * 2, 3), dtype=np.uint8)
    placements = [
        (0, 0),
        (0, tile_w),
        (tile_h, 0),
        (tile_h, tile_w),
    ]

    mosaic_boxes: list[list[float]] = []
    mosaic_labels: list[int] = []
    for tile, (offset_y, offset_x) in zip(tiles, placements):
        canvas[offset_y : offset_y + tile_h, offset_x : offset_x + tile_w] = tile["image"]
        for bbox, class_id in zip(tile.get("bboxes", []), tile.get("class_labels", [])):
            x1, y1, x2, y2 = bbox
            mosaic_boxes.append(
                [
                    float(x1 + offset_x),
                    float(y1 + offset_y),
                    float(x2 + offset_x),
                    float(y2 + offset_y),
                ]
            )
            mosaic_labels.append(int(class_id))

    return {
        "image": canvas,
        "bboxes": mosaic_boxes,
        "class_labels": mosaic_labels,
        "image_path": anchor_sample.get("image_path"),
    }


def _bbox_params(min_area: float = 1.0) -> Any:
    if A is None:
        raise ImportError("albumentations is required for augmentation demo helpers")
    kwargs = {
        "format": "pascal_voc",
        "label_fields": ["class_labels"],
        "min_visibility": 0.0,
        "min_area": float(min_area),
    }
    if "clip" in inspect.signature(A.BboxParams).parameters:
        kwargs["clip"] = True
    return A.BboxParams(**kwargs)


def _scalar_demo_values(args: dict[str, float], seed: int = 42) -> dict[str, float]:
    rng = random.Random(seed)

    def signed_fraction(min_fraction: float, max_fraction: float) -> float:
        magnitude = min_fraction + (max_fraction - min_fraction) * rng.random()
        sign = -1.0 if rng.random() < 0.5 else 1.0
        return sign * magnitude

    degrees = float(args.get("degrees", 0.0))
    translate = float(args.get("translate", 0.0))
    scale = float(args.get("scale", 0.0))
    hsv_h = float(args.get("hsv_h", 0.0))
    hsv_s = float(args.get("hsv_s", 0.0))
    hsv_v = float(args.get("hsv_v", 0.0))

    scale_delta = signed_fraction(0.22, 0.48) * scale if scale > 0 else 0.0
    scale_factor = max(0.65, 1.0 + scale_delta)

    return {
        "rotate": degrees * signed_fraction(0.35, 0.75) if degrees > 0 else 0.0,
        "translate_x": translate * signed_fraction(0.30, 0.70) if translate > 0 else 0.0,
        "translate_y": translate * signed_fraction(0.20, 0.60) if translate > 0 else 0.0,
        "scale_factor": scale_factor,
        "hue_shift": hsv_h * 180.0 * signed_fraction(0.30, 0.80) if hsv_h > 0 else 0.0,
        "sat_mult": 1.0 + hsv_s * (0.14 + 0.16 * rng.random()) if hsv_s > 0 else 1.0,
        "val_mult": 1.0 + hsv_v * (0.10 + 0.14 * rng.random()) if hsv_v > 0 else 1.0,
    }


def apply_demo_affine_and_flips(
    sample: Sample,
    args: dict[str, float],
    seed: int = 42,
) -> Sample:
    if A is None:
        raise ImportError("albumentations is required for augmentation demo helpers")

    values = _scalar_demo_values(args=args, seed=seed)
    transforms: list[Any] = []
    has_affine = any(
        abs(values[key]) > 1e-9
        for key in ("rotate", "translate_x", "translate_y")
    ) or abs(values["scale_factor"] - 1.0) > 1e-9
    if has_affine:
        transforms.append(
            A.Affine(
                scale=float(values["scale_factor"]),
                translate_percent={
                    "x": float(values["translate_x"]),
                    "y": float(values["translate_y"]),
                },
                rotate=float(values["rotate"]),
                fit_output=False,
                p=1.0,
            )
        )
    if float(args.get("fliplr", 0.0)) > 0:
        transforms.append(A.HorizontalFlip(p=1.0))
    if float(args.get("flipud", 0.0)) > 0:
        transforms.append(A.VerticalFlip(p=1.0))

    if not transforms:
        return _clone_sample(sample)

    compose = A.Compose(transforms, bbox_params=_bbox_params(min_area=1.0))
    out = compose(
        image=sample["image"],
        bboxes=sample.get("bboxes", []),
        class_labels=sample.get("class_labels", []),
    )
    return {
        "image": out["image"],
        "bboxes": [list(map(float, box)) for box in out["bboxes"]],
        "class_labels": [int(label) for label in out["class_labels"]],
        "image_path": sample.get("image_path"),
    }


def apply_demo_hsv(sample: Sample, args: dict[str, float], seed: int = 42) -> Sample:
    out = _clone_sample(sample)
    values = _scalar_demo_values(args=args, seed=seed)
    image = out["image"].copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + values["hue_shift"]) % 180.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * values["sat_mult"], 0.0, 255.0)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * values["val_mult"], 0.0, 255.0)
    out["image"] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def apply_demo_scalar_policy(
    sample: Sample,
    scalar_args: dict[str, float],
    donor_pool: list[Sample] | None = None,
    seed: int = 42,
) -> Sample:
    out = _clone_sample(sample)
    if float(scalar_args.get("mosaic", 0.0)) > 0 and donor_pool:
        rng = random.Random(seed)
        donor_candidates = [
            donor
            for donor in donor_pool
            if donor.get("image_path") != sample.get("image_path")
        ]
        if len(donor_candidates) >= 3:
            donors = rng.sample(donor_candidates, 3)
            out = build_mosaic_sample(anchor_sample=out, donor_samples=donors)

    out = apply_demo_affine_and_flips(out, args=scalar_args, seed=seed)
    out = apply_demo_hsv(out, args=scalar_args, seed=seed + 17)
    return out


def apply_demo_mode(
    sample: Sample,
    scalar_args: dict[str, float],
    donor_pool: list[Sample] | None = None,
    custom_transforms: list[Any] | None = None,
    seed: int = 42,
) -> Sample:
    out = apply_demo_scalar_policy(
        sample=sample,
        scalar_args=scalar_args,
        donor_pool=donor_pool,
        seed=seed,
    )
    if custom_transforms:
        out = apply_custom_transforms(out, custom_transforms, min_area=1.0)
        out["image_path"] = sample.get("image_path")
    return out


def build_object_bank_from_dataset(
    dataset_root: str | Path,
    split: str = "train",
    small_max_area: float = 32.0**2,
    tiny_max_area: float = 16.0**2,
    max_items_per_class: int = 2000,
    seed: int = 42,
) -> ObjectBank:
    dataset_root = Path(dataset_root)
    bank = ObjectBank(
        small_max_area=float(small_max_area),
        tiny_max_area=float(tiny_max_area),
        max_items_per_class=int(max_items_per_class),
        seed=seed,
    )
    bank.build_from_dataset(
        images_dir=dataset_root / "images" / split,
        labels_dir=dataset_root / "labels" / split,
    )
    return bank


def mode_summary_rows(
    baseline_args: dict[str, float],
    manual_args: dict[str, float],
    adaptive_args: dict[str, float],
    adaptive_spec: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    adaptive_spec = adaptive_spec or []

    def active_components(mode_name: str, args: dict[str, float], custom_names: list[str] | None = None) -> str:
        items: list[str] = []
        if float(args.get("mosaic", 0.0)) > 0:
            items.append(f"mosaic={args['mosaic']}")
        if any(float(args.get(key, 0.0)) > 0 for key in ("degrees", "translate", "scale")):
            items.append(
                "geometric("
                f"deg={args.get('degrees', 0.0)}, "
                f"translate={args.get('translate', 0.0)}, "
                f"scale={args.get('scale', 0.0)})"
            )
        if any(float(args.get(key, 0.0)) > 0 for key in ("hsv_h", "hsv_s", "hsv_v")):
            items.append(
                "hsv("
                f"h={args.get('hsv_h', 0.0)}, "
                f"s={args.get('hsv_s', 0.0)}, "
                f"v={args.get('hsv_v', 0.0)})"
            )
        if float(args.get("fliplr", 0.0)) > 0:
            items.append(f"fliplr={args['fliplr']}")
        if float(args.get("flipud", 0.0)) > 0:
            items.append(f"flipud={args['flipud']}")
        if float(args.get("mixup", 0.0)) > 0:
            items.append(f"mixup={args['mixup']}")
        if float(args.get("cutmix", 0.0)) > 0:
            items.append(f"cutmix={args['cutmix']}")
        if custom_names:
            items.extend(custom_names)
        return ", ".join(items) if items else "none"

    return [
        {
            "mode": "baseline",
            "active_augmentations": active_components("baseline", baseline_args),
        },
        {
            "mode": "manual",
            "active_augmentations": active_components("manual", manual_args),
        },
        {
            "mode": "adaptive",
            "active_augmentations": active_components(
                "adaptive",
                adaptive_args,
                custom_names=[item["name"] for item in adaptive_spec],
            ),
        },
    ]


def class_color(class_id: int) -> tuple[int, int, int]:
    rng = random.Random(int(class_id) + 1337)
    return (
        int(64 + rng.randint(0, 191)),
        int(64 + rng.randint(0, 191)),
        int(64 + rng.randint(0, 191)),
    )


def draw_sample(
    sample: Sample,
    class_names: list[str] | None = None,
    line_width: int = 2,
) -> np.ndarray:
    image = sample["image"].copy()
    for bbox, class_id in zip(sample.get("bboxes", []), sample.get("class_labels", [])):
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        color = class_color(int(class_id))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_width)
        if class_names and 0 <= int(class_id) < len(class_names):
            label = class_names[int(class_id)]
        else:
            label = str(int(class_id))
        cv2.putText(
            image,
            label,
            (max(0, x1), max(18, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
