from __future__ import annotations

import json
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.data.visdrone_manager import DEFAULT_IMAGE_SUFFIXES, validate_yolo_split
from src.utils.io import dump_json, dump_yaml


COCO_ARCHIVES_2017: dict[str, str] = {
    "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


@dataclass
class CocoSmallPrepareConfig:
    """
    Conversion options for COCO -> YOLO (small objects only).
    """

    small_max_area: float = 32.0**2
    keep_images_without_small: bool = False
    include_crowd: bool = False
    splits: tuple[str, ...] = ("train", "val")
    link_images: bool = True
    image_extensions: tuple[str, ...] = DEFAULT_IMAGE_SUFFIXES
    selected_category_ids: list[int] | None = None


@dataclass
class CocoSmallPrepareReport:
    dataset_root: str
    is_valid: bool
    splits: dict[str, dict[str, Any]] = field(default_factory=dict)
    class_names: list[str] = field(default_factory=list)
    small_max_area: float = 32.0**2


def _download_file(url: str, target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and target_path.stat().st_size > 0:
        return target_path
    urllib.request.urlretrieve(url, filename=str(target_path))
    return target_path


def download_coco_2017_archives(download_root: str | Path) -> dict[str, Path]:
    """
    Download COCO 2017 train/val images and annotation archives.
    """
    download_root = Path(download_root)
    paths: dict[str, Path] = {}
    for archive_name, url in COCO_ARCHIVES_2017.items():
        path = _download_file(url=url, target_path=download_root / archive_name)
        paths[archive_name] = path
    return paths


def extract_coco_2017_archives(
    archives_root: str | Path,
    output_root: str | Path,
    overwrite: bool = False,
) -> dict[str, Path]:
    """
    Extract COCO archives into raw structure:
      <output_root>/train2017
      <output_root>/val2017
      <output_root>/annotations
    """
    archives_root = Path(archives_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    extracted: dict[str, Path] = {}
    for archive_name in COCO_ARCHIVES_2017:
        zip_path = archives_root / archive_name
        if not zip_path.exists():
            raise FileNotFoundError(f"Missing COCO archive: {zip_path.as_posix()}")

        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.namelist()
            if not members:
                continue
            top_level = members[0].split("/")[0]
            target = output_root / top_level
            if target.exists() and not overwrite:
                extracted[top_level] = target
                continue
            zf.extractall(path=output_root)
            extracted[top_level] = target

    return extracted


def _load_coco_instances(instances_path: Path) -> dict:
    with instances_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _symlink_or_copy_image(source_path: Path, target_path: Path, use_symlink: bool) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        return
    if use_symlink:
        try:
            target_path.symlink_to(source_path)
            return
        except OSError:
            pass
    shutil.copy2(source_path, target_path)


def _clip_bbox_xywh(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    x = max(0.0, min(float(img_w), x))
    y = max(0.0, min(float(img_h), y))
    w = max(0.0, min(float(img_w) - x, w))
    h = max(0.0, min(float(img_h) - y, h))
    return x, y, w, h


def _build_category_mapping(coco_payload: dict, selected_category_ids: list[int] | None) -> tuple[dict[int, int], list[str]]:
    categories = sorted(coco_payload.get("categories", []), key=lambda item: int(item["id"]))
    if selected_category_ids is not None:
        allowed = {int(x) for x in selected_category_ids}
        categories = [item for item in categories if int(item["id"]) in allowed]

    category_to_yolo: dict[int, int] = {}
    class_names: list[str] = []
    for yolo_id, item in enumerate(categories):
        coco_id = int(item["id"])
        category_to_yolo[coco_id] = yolo_id
        class_names.append(str(item["name"]))
    return category_to_yolo, class_names


def _convert_split_to_small_yolo(
    raw_root: Path,
    output_root: Path,
    split: str,
    config: CocoSmallPrepareConfig,
    category_to_yolo: dict[int, int],
) -> dict[str, int]:
    split_to_folder = {"train": "train2017", "val": "val2017", "test": "test2017"}
    folder_name = split_to_folder.get(split, f"{split}2017")
    images_src = raw_root / folder_name
    ann_path = raw_root / "annotations" / f"instances_{folder_name}.json"
    if not ann_path.exists():
        # test split in COCO may not have instance annotations.
        return {
            "num_images": 0,
            "num_label_files": 0,
            "num_objects": 0,
            "num_small_objects": 0,
        }
    if not images_src.exists():
        raise FileNotFoundError(f"Missing COCO images dir: {images_src.as_posix()}")

    coco_payload = _load_coco_instances(ann_path)
    images = {int(item["id"]): item for item in coco_payload.get("images", [])}

    grouped_annotations: dict[int, list[dict]] = {}
    total_objects = 0
    total_small_objects = 0
    for ann in coco_payload.get("annotations", []):
        category_id = int(ann.get("category_id", -1))
        if category_id not in category_to_yolo:
            continue
        if not config.include_crowd and int(ann.get("iscrowd", 0)) == 1:
            continue

        bbox = ann.get("bbox", [])
        if len(bbox) != 4:
            continue
        x, y, w, h = map(float, bbox)
        area = float(ann.get("area", w * h))
        if area > float(config.small_max_area):
            total_objects += 1
            continue

        image_id = int(ann["image_id"])
        grouped_annotations.setdefault(image_id, []).append(ann)
        total_small_objects += 1
        total_objects += 1

    images_dst = output_root / "images" / split
    labels_dst = output_root / "labels" / split
    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    written_images = 0
    written_labels = 0

    for image_id, image_info in images.items():
        anns = grouped_annotations.get(image_id, [])
        if not anns and not config.keep_images_without_small:
            continue

        file_name = str(image_info["file_name"])
        src_image_path = images_src / file_name
        if not src_image_path.exists():
            continue

        target_image_path = images_dst / file_name
        _symlink_or_copy_image(
            source_path=src_image_path,
            target_path=target_image_path,
            use_symlink=config.link_images,
        )
        written_images += 1

        width = int(image_info["width"])
        height = int(image_info["height"])
        label_lines: list[str] = []
        for ann in anns:
            x, y, w, h = map(float, ann["bbox"])
            x, y, w, h = _clip_bbox_xywh(x, y, w, h, img_w=width, img_h=height)
            if w <= 1e-6 or h <= 1e-6:
                continue
            x_center = (x + w / 2.0) / max(1.0, float(width))
            y_center = (y + h / 2.0) / max(1.0, float(height))
            w_norm = w / max(1.0, float(width))
            h_norm = h / max(1.0, float(height))
            class_id = category_to_yolo[int(ann["category_id"])]
            label_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"
            )

        label_path = labels_dst / f"{Path(file_name).stem}.txt"
        label_path.write_text("".join(label_lines), encoding="utf-8")
        written_labels += 1

    return {
        "num_images": written_images,
        "num_label_files": written_labels,
        "num_objects": total_objects,
        "num_small_objects": total_small_objects,
    }


def _write_coco_small_yaml(output_root: Path, class_names: list[str]) -> Path:
    payload = {
        "path": output_root.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {idx: name for idx, name in enumerate(class_names)},
    }
    yaml_path = output_root / "coco_small.yaml"
    dump_yaml(payload, yaml_path)
    return yaml_path


def prepare_coco_small_yolo(
    raw_coco_root: str | Path,
    output_root: str | Path,
    config: CocoSmallPrepareConfig | None = None,
) -> CocoSmallPrepareReport:
    """
    Convert COCO raw dataset into YOLO structure using only small objects.
    """
    if config is None:
        config = CocoSmallPrepareConfig()

    raw_coco_root = Path(raw_coco_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    instances_train = raw_coco_root / "annotations" / "instances_train2017.json"
    if not instances_train.exists():
        raise FileNotFoundError(
            f"Missing COCO annotations file: {instances_train.as_posix()}"
        )
    train_payload = _load_coco_instances(instances_train)
    category_to_yolo, class_names = _build_category_mapping(
        coco_payload=train_payload,
        selected_category_ids=config.selected_category_ids,
    )
    if not class_names:
        raise ValueError("No categories selected for COCO small conversion.")

    split_reports: dict[str, dict[str, Any]] = {}
    for split in config.splits:
        split_reports[split] = _convert_split_to_small_yolo(
            raw_root=raw_coco_root,
            output_root=output_root,
            split=split,
            config=config,
            category_to_yolo=category_to_yolo,
        )

    yaml_path = _write_coco_small_yaml(output_root=output_root, class_names=class_names)

    validation: dict[str, Any] = {"splits": {}, "is_valid": True}
    for split in config.splits:
        split_result = validate_yolo_split(
            images_dir=output_root / "images" / split,
            labels_dir=output_root / "labels" / split,
            split=split,
            image_extensions=config.image_extensions,
        )
        payload = {
            **split_result.__dict__,
            **split_reports.get(split, {}),
        }
        validation["splits"][split] = payload
        if split_result.num_missing_labels > 0 or split_result.num_orphan_labels > 0:
            validation["is_valid"] = False

    report = CocoSmallPrepareReport(
        dataset_root=output_root.as_posix(),
        is_valid=bool(validation["is_valid"]),
        splits=validation["splits"],
        class_names=class_names,
        small_max_area=float(config.small_max_area),
    )
    dump_json(
        {
            "dataset_root": report.dataset_root,
            "is_valid": report.is_valid,
            "small_max_area": report.small_max_area,
            "class_names": report.class_names,
            "splits": report.splits,
            "data_yaml": yaml_path.as_posix(),
        },
        output_root / "coco_small_prepare_report.json",
    )
    return report


def prepare_coco_small_by_mode(
    dataset_root: str | Path,
    mode: str = "manual",
    raw_coco_root: str | Path | None = None,
    config: CocoSmallPrepareConfig | None = None,
) -> dict:
    """
    Dataset-mode entrypoint for COCO-small flow.

    - mode=manual: validate existing YOLO dataset.
    - mode=auto: convert raw COCO to YOLO small dataset and validate.
    """
    if config is None:
        config = CocoSmallPrepareConfig()

    mode_norm = str(mode).strip().lower()
    dataset_root = Path(dataset_root)
    if mode_norm not in {"manual", "auto"}:
        raise ValueError(f"Unsupported dataset mode: '{mode}'. Expected 'manual' or 'auto'.")

    if mode_norm == "auto":
        if raw_coco_root is None:
            raise ValueError("raw_coco_root must be provided when dataset.mode='auto'.")
        converted = prepare_coco_small_yolo(
            raw_coco_root=raw_coco_root,
            output_root=dataset_root,
            config=config,
        )
        return {
            "dataset_root": converted.dataset_root,
            "is_valid": converted.is_valid,
            "summary": {
                "mode": mode_norm,
                "small_max_area": converted.small_max_area,
                "num_classes": len(converted.class_names),
            },
            "splits": converted.splits,
            "class_names": converted.class_names,
        }

    splits_report: dict[str, Any] = {}
    is_valid = True
    for split in config.splits:
        split_result = validate_yolo_split(
            images_dir=dataset_root / "images" / split,
            labels_dir=dataset_root / "labels" / split,
            split=split,
            image_extensions=config.image_extensions,
        )
        splits_report[split] = split_result.__dict__
        if split_result.num_missing_labels > 0 or split_result.num_orphan_labels > 0:
            is_valid = False

    return {
        "dataset_root": dataset_root.as_posix(),
        "is_valid": is_valid,
        "summary": {
            "mode": mode_norm,
            "small_max_area": float(config.small_max_area),
        },
        "splits": splits_report,
    }
