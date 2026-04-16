from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class SubsetBuildReport:
    """Summary of generated subset sizes."""

    train_images: int
    val_images: int
    output_root: str


def _list_images(images_dir: Path, image_extensions: Iterable[str]) -> list[Path]:
    ext_set = {ext.lower() for ext in image_extensions}
    return sorted(
        [
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in ext_set
        ]
    )


def _copy_split_subset(
    dataset_root: Path,
    output_root: Path,
    split: str,
    max_images: int,
    seed: int,
    image_extensions: Iterable[str],
) -> int:
    """
    Copy random subset for one split while preserving YOLO structure.

    Images are copied to:
      output_root/images/<split>/
    Labels are copied to:
      output_root/labels/<split>/
    """
    images_src = dataset_root / "images" / split
    labels_src = dataset_root / "labels" / split
    images_dst = output_root / "images" / split
    labels_dst = output_root / "labels" / split

    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    image_paths = _list_images(images_src, image_extensions=image_extensions)
    if not image_paths:
        return 0

    rng = random.Random(seed + hash(split) % 10_000)
    sample_size = min(max_images, len(image_paths))
    selected = rng.sample(image_paths, k=sample_size)

    copied = 0
    for image_path in selected:
        label_path = labels_src / f"{image_path.stem}.txt"
        if not label_path.exists():
            # For MVP testing we skip unmatched image files to keep pairs strict.
            continue
        shutil.copy2(image_path, images_dst / image_path.name)
        shutil.copy2(label_path, labels_dst / label_path.name)
        copied += 1

    return copied


def build_yolo_subset(
    dataset_root: str | Path,
    output_root: str | Path,
    train_images: int = 120,
    val_images: int = 40,
    seed: int = 42,
    image_extensions: Iterable[str] = (".jpg", ".jpeg", ".png", ".bmp"),
    clean_output: bool = True,
) -> SubsetBuildReport:
    """
    Build a lightweight local subset for fast CPU component testing.

    The defaults are tuned for end-to-end smoke tests under ~30 minutes on CPU.
    """
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)

    if clean_output and output_root.exists():
        shutil.rmtree(output_root, ignore_errors=True)
    output_root.mkdir(parents=True, exist_ok=True)

    train_count = _copy_split_subset(
        dataset_root=dataset_root,
        output_root=output_root,
        split="train",
        max_images=train_images,
        seed=seed,
        image_extensions=image_extensions,
    )
    val_count = _copy_split_subset(
        dataset_root=dataset_root,
        output_root=output_root,
        split="val",
        max_images=val_images,
        seed=seed,
        image_extensions=image_extensions,
    )

    return SubsetBuildReport(
        train_images=train_count,
        val_images=val_count,
        output_root=output_root.as_posix(),
    )

