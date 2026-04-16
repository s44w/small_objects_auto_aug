from __future__ import annotations

import inspect
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
    num_missing_labels: int = 0
    num_orphan_labels: int = 0
    missing_labels: list[str] = field(default_factory=list)
    orphan_labels: list[str] = field(default_factory=list)


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
        if label_path.stat().st_size == 0:
            result.num_empty_label_files += 1
            continue
        with label_path.open("r", encoding="utf-8") as file:
            if not any(line.strip() for line in file):
                result.num_empty_label_files += 1

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
    except Exception as exc:
        raise RuntimeError(
            "Ultralytics converter is unavailable. Install ultralytics and retry."
        ) from exc

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

