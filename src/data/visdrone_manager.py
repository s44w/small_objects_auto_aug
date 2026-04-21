from __future__ import annotations

import inspect
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.utils.io import dump_json


DEFAULT_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class SplitValidationResult:
    """Validation report for a single split (train/val/test)."""

    split: str
    num_images: int = 0
    num_label_files: int = 0
    num_images_with_labels: int = 0
    num_empty_label_files: int = 0
    num_unreadable_label_files: int = 0
    num_missing_labels: int = 0
    num_orphan_labels: int = 0
    missing_labels: list[str] = field(default_factory=list)
    orphan_labels: list[str] = field(default_factory=list)
    unreadable_labels: list[str] = field(default_factory=list)


def _find_images(images_dir: Path, image_extensions: tuple[str, ...]) -> list[Path]:
    return sorted(
        [
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in image_extensions
        ]
    )


def validate_yolo_split(
    images_dir: str | Path,
    labels_dir: str | Path,
    split: str,
    image_extensions: tuple[str, ...] = DEFAULT_IMAGE_SUFFIXES,
) -> SplitValidationResult:
    """
    Validate one split of a YOLO dataset.

    Expected structure:
    - images/<split>/*.jpg
    - labels/<split>/*.txt
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    result = SplitValidationResult(split=split)

    if not images_dir.exists():
        result.missing_labels.append(f"Missing directory: {images_dir.as_posix()}")
        return result
    if not labels_dir.exists():
        result.orphan_labels.append(f"Missing directory: {labels_dir.as_posix()}")
        return result

    images = _find_images(images_dir, image_extensions=image_extensions)
    labels = sorted(labels_dir.glob("*.txt"))

    result.num_images = len(images)
    result.num_label_files = len(labels)

    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}

    missing = sorted(image_stems - label_stems)
    orphan = sorted(label_stems - image_stems)
    matched = sorted(image_stems & label_stems)

    result.num_missing_labels = len(missing)
    result.num_orphan_labels = len(orphan)
    result.num_images_with_labels = len(matched)
    result.missing_labels = [f"{stem}.txt" for stem in missing]
    result.orphan_labels = [f"{stem}.txt" for stem in orphan]

    for label_path in labels:
        try:
            if label_path.stat().st_size == 0:
                result.num_empty_label_files += 1
                continue
            with label_path.open("r", encoding="utf-8") as file:
                if not any(line.strip() for line in file):
                    result.num_empty_label_files += 1
        except OSError as exc:
            # Drive-mounted filesystems in Colab can intermittently abort I/O.
            # We keep validation resilient and report unreadable files explicitly.
            result.num_unreadable_label_files += 1
            result.unreadable_labels.append(f"{label_path.name}: {exc}")

    return result


def validate_visdrone_yolo_structure(
    dataset_root: str | Path,
    splits: tuple[str, ...] = ("train", "val"),
    image_extensions: tuple[str, ...] = DEFAULT_IMAGE_SUFFIXES,
) -> dict:
    """Validate YOLO-like VisDrone dataset directory and return detailed report."""
    dataset_root = Path(dataset_root)
    report = {
        "dataset_root": dataset_root.as_posix(),
        "splits": {},
        "is_valid": True,
    }

    for split in splits:
        split_result = validate_yolo_split(
            images_dir=dataset_root / "images" / split,
            labels_dir=dataset_root / "labels" / split,
            split=split,
            image_extensions=image_extensions,
        )
        report["splits"][split] = asdict(split_result)

        if split_result.num_missing_labels > 0 or split_result.num_orphan_labels > 0:
            report["is_valid"] = False

    return report


def save_validation_report(report: dict, output_path: str | Path) -> None:
    """Persist validation report as JSON."""
    dump_json(report, output_path)


def _resolve_visdrone_split_dir(raw_root: Path, split: str) -> Path | None:
    """
    Resolve source directory for a split in raw VisDrone layout.

    Supported examples:
    - <raw_root>/VisDrone2019-DET-train
    - <raw_root>/VisDrone2019-DET-val
    - <raw_root>/VisDrone2019-DET-test-dev
    - nested locations with those names
    """
    names_by_split = {
        "train": ["VisDrone2019-DET-train", "train"],
        "val": ["VisDrone2019-DET-val", "val"],
        "test": ["VisDrone2019-DET-test-dev", "VisDrone2019-DET-test-challenge", "test"],
    }
    names = names_by_split.get(split, [split])

    for name in names:
        candidate = raw_root / name
        if candidate.exists() and candidate.is_dir():
            return candidate

    for name in names:
        matches = list(raw_root.rglob(name))
        for match in matches:
            if match.exists() and match.is_dir():
                return match
    return None


def _image_size(image_path: Path) -> tuple[int, int]:
    """Return image size as (width, height)."""
    try:
        from PIL import Image
    except Exception:
        Image = None

    if Image is not None:
        with Image.open(image_path) as im:
            return int(im.width), int(im.height)

    # Fallback: OpenCV
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image for size: {image_path.as_posix()}")
    height, width = image.shape[:2]
    return int(width), int(height)


def _convert_annotation_row_to_yolo(
    row: list[str],
    width: int,
    height: int,
) -> str | None:
    """
    Convert one VisDrone annotation row to YOLO string.

    VisDrone row format (DET):
      bbox_left,bbox_top,bbox_width,bbox_height,score,category,truncation,occlusion
    """
    if len(row) < 6:
        return None

    # Skip ignored regions.
    if row[4].strip() == "0":
        return None

    x, y, w, h = map(float, row[:4])
    class_id = int(float(row[5])) - 1  # VisDrone class ids are 1..10 for valid objects
    if class_id < 0 or class_id > 9:
        return None

    # Clip raw boxes to image bounds before normalization.
    x1 = max(0.0, min(float(width), x))
    y1 = max(0.0, min(float(height), y))
    x2 = max(0.0, min(float(width), x + w))
    y2 = max(0.0, min(float(height), y + h))
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1e-6 or bh <= 1e-6:
        return None

    dw = 1.0 / max(1.0, float(width))
    dh = 1.0 / max(1.0, float(height))
    x_center = (x1 + bw / 2.0) * dw
    y_center = (y1 + bh / 2.0) * dh
    w_norm = bw * dw
    h_norm = bh * dh

    # Final safety clamp to valid normalized range.
    x_center = min(max(x_center, 0.0), 1.0)
    y_center = min(max(y_center, 0.0), 1.0)
    w_norm = min(max(w_norm, 0.0), 1.0)
    h_norm = min(max(h_norm, 0.0), 1.0)
    if w_norm <= 0.0 or h_norm <= 0.0:
        return None

    return f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"


def _convert_visdrone_split_to_yolo(
    source_dir: Path,
    output_root: Path,
    split: str,
) -> tuple[int, int]:
    """
    Convert one split from raw VisDrone format to YOLO format.

    Returns:
        (num_images_copied, num_labels_written)
    """
    source_images_dir = source_dir / "images"
    source_annotations_dir = source_dir / "annotations"
    if not source_images_dir.exists():
        raise FileNotFoundError(f"Missing source images dir: {source_images_dir.as_posix()}")

    images_dir = output_root / "images" / split
    labels_dir = output_root / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    copied_images = 0
    for image_path in source_images_dir.glob("*.jpg"):
        target_path = images_dir / image_path.name
        if not target_path.exists():
            shutil.copy2(image_path, target_path)
            copied_images += 1

    written_labels = 0
    if source_annotations_dir.exists():
        for ann_path in sorted(source_annotations_dir.glob("*.txt")):
            image_path = images_dir / ann_path.with_suffix(".jpg").name
            if not image_path.exists():
                # Fallback for png extension
                image_path = images_dir / ann_path.with_suffix(".png").name
            if not image_path.exists():
                continue

            width, height = _image_size(image_path)
            lines: list[str] = []
            with ann_path.open("r", encoding="utf-8") as file:
                for raw_line in file.read().strip().splitlines():
                    if not raw_line.strip():
                        continue
                    yolo_line = _convert_annotation_row_to_yolo(
                        row=[x.strip() for x in raw_line.split(",")],
                        width=width,
                        height=height,
                    )
                    if yolo_line is not None:
                        lines.append(yolo_line)
            (labels_dir / ann_path.name).write_text("".join(lines), encoding="utf-8")
            written_labels += 1

    return copied_images, written_labels


def _write_visdrone_data_yaml(output_root: Path) -> None:
    """Write a minimal VisDrone-compatible data.yaml for training convenience."""
    names = {
        0: "pedestrian",
        1: "people",
        2: "bicycle",
        3: "car",
        4: "van",
        5: "truck",
        6: "tricycle",
        7: "awning-tricycle",
        8: "bus",
        9: "motor",
    }
    yaml_content = {
        "path": output_root.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }
    try:
        import yaml
    except Exception:
        return
    (output_root / "visdrone.yaml").write_text(
        yaml.safe_dump(yaml_content, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _convert_visdrone_without_ultralytics(raw_root: Path, output_root: Path) -> None:
    """Internal robust converter used when Ultralytics converter function is unavailable."""
    output_root.mkdir(parents=True, exist_ok=True)
    total_images = 0
    total_labels = 0

    for split in ("train", "val", "test"):
        source_dir = _resolve_visdrone_split_dir(raw_root, split=split)
        if source_dir is None:
            continue
        copied_images, written_labels = _convert_visdrone_split_to_yolo(
            source_dir=source_dir,
            output_root=output_root,
            split=split,
        )
        total_images += copied_images
        total_labels += written_labels

    if not (output_root / "images" / "train").exists() or not (output_root / "images" / "val").exists():
        raise FileNotFoundError(
            "Failed to build train/val splits from raw VisDrone. "
            f"Checked raw root: {raw_root.as_posix()}"
        )

    _write_visdrone_data_yaml(output_root)
    print(
        f"[prepare_visdrone_auto] Fallback conversion complete: images_copied={total_images}, "
        f"labels_written={total_labels}, output={output_root.as_posix()}"
    )


def prepare_visdrone_auto(raw_visdrone_root: str | Path, output_root: str | Path) -> None:
    """
    Try to trigger Ultralytics VisDrone conversion in "auto" mode.

    Notes:
    - The exact converter signature can differ by Ultralytics version.
    - We inspect available arguments and call converter in a version-safe way.
    """
    raw_visdrone_root = Path(raw_visdrone_root)
    output_root = Path(output_root)

    try:
        from ultralytics.data.converter import convert_visdrone
    except Exception:
        convert_visdrone = None

    if convert_visdrone is None:
        _convert_visdrone_without_ultralytics(raw_root=raw_visdrone_root, output_root=output_root)
        return

    try:
        signature = inspect.signature(convert_visdrone)
        kwargs = {}
        for candidate in ("path", "root", "root_dir", "dataset_dir", "dir"):
            if candidate in signature.parameters:
                kwargs[candidate] = str(raw_visdrone_root)
                break

        for candidate in ("save_dir", "output_dir", "output"):
            if candidate in signature.parameters:
                kwargs[candidate] = str(output_root)
                break

        if not kwargs:
            # Fallback when converter expects positional args only.
            convert_visdrone(str(raw_visdrone_root))
            return

        convert_visdrone(**kwargs)
    except Exception:
        # Last-resort stable path: local converter logic.
        _convert_visdrone_without_ultralytics(raw_root=raw_visdrone_root, output_root=output_root)
