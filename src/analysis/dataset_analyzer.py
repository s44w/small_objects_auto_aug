from __future__ import annotations

import argparse
import datetime as dt
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from src.analysis.stats_schema import (
    MEDIUM_MAX_AREA_DEFAULT,
    SMALL_MAX_AREA_DEFAULT,
    TINY_MAX_AREA_DEFAULT,
    area_bucket,
    is_tiny,
    validate_stats_payload,
)
from src.data.yolo_label_reader import load_yolo_labels, yolo_bbox_area_px
from src.utils.io import dump_json, dump_rows_to_csv, ensure_dir, flatten_dict
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)


@dataclass
class DatasetAnalyzerConfig:
    """Configurable thresholds for the dataset analyzer."""

    small_max_area: float = SMALL_MAX_AREA_DEFAULT
    medium_max_area: float = MEDIUM_MAX_AREA_DEFAULT
    tiny_max_area: float = TINY_MAX_AREA_DEFAULT
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")
    generate_plots: bool = True


def _safe_describe(values: list[float]) -> dict[str, float]:
    """Return robust descriptive statistics even when input is empty."""
    if not values:
        zeros = {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
        return zeros

    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def _imbalance_ratio(counts: Counter[int]) -> float:
    """Compute class imbalance as max_count / min_count over non-empty classes."""
    non_zero = [value for value in counts.values() if value > 0]
    if not non_zero:
        return 1.0
    return float(max(non_zero)) / float(max(1, min(non_zero)))


def _image_paths(images_dir: Path, extensions: Iterable[str]) -> list[Path]:
    ext_set = {ext.lower() for ext in extensions}
    return sorted(
        [
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in ext_set
        ]
    )


def _compute_illumination_features(image_bgr: np.ndarray) -> tuple[float, float, float]:
    """
    Compute image-level illumination features from HSV V channel.

    - v_mean: average brightness
    - v_std: brightness variability
    - contrast: standard deviation over grayscale intensity
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2].astype(np.float32)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(v_channel.mean()), float(v_channel.std()), float(gray.std())


def _analyze_split(
    images_dir: Path,
    labels_dir: Path,
    split: str,
    config: DatasetAnalyzerConfig,
) -> dict:
    image_paths = _image_paths(images_dir, extensions=config.image_extensions)
    label_paths = sorted(labels_dir.glob("*.txt")) if labels_dir.exists() else []

    areas_px2: list[float] = []
    widths: list[float] = []
    heights: list[float] = []
    aspect_ratios: list[float] = []

    v_means: list[float] = []
    v_stds: list[float] = []
    contrasts: list[float] = []

    objects_per_image: list[float] = []
    objects_per_mpix: list[float] = []

    class_counts: Counter[int] = Counter()
    small_class_counts: Counter[int] = Counter()

    tiny_count = 0
    small_count = 0
    medium_count = 0
    large_count = 0
    empty_labels_count = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            LOGGER.warning("Failed to load image: %s", image_path.as_posix())
            continue

        height, width = image.shape[:2]
        widths.append(float(width))
        heights.append(float(height))
        aspect_ratios.append(float(width) / float(max(1, height)))

        v_mean, v_std, contrast = _compute_illumination_features(image)
        v_means.append(v_mean)
        v_stds.append(v_std)
        contrasts.append(contrast)

        label_path = labels_dir / f"{image_path.stem}.txt"
        boxes = load_yolo_labels(label_path)

        if len(boxes) == 0:
            empty_labels_count += 1

        object_count = 0
        mpix = (float(width) * float(height)) / 1_000_000.0

        for bbox in boxes:
            area = yolo_bbox_area_px(bbox=bbox, image_width=width, image_height=height)
            if area <= 0:
                continue

            object_count += 1
            areas_px2.append(area)
            class_counts[bbox.class_id] += 1

            bucket = area_bucket(
                area_px2=area,
                small_max_area=config.small_max_area,
                medium_max_area=config.medium_max_area,
            )
            if bucket == "small":
                small_count += 1
                small_class_counts[bbox.class_id] += 1
            elif bucket == "medium":
                medium_count += 1
            else:
                large_count += 1

            if is_tiny(area, tiny_max_area=config.tiny_max_area):
                tiny_count += 1

        objects_per_image.append(float(object_count))
        objects_per_mpix.append(float(object_count) / max(mpix, 1e-6))

    num_objects = int(sum(class_counts.values()))
    small_ratio = float(small_count) / max(1, num_objects)
    medium_ratio = float(medium_count) / max(1, num_objects)
    large_ratio = float(large_count) / max(1, num_objects)
    tiny_ratio = float(tiny_count) / max(1, num_objects)

    split_stats = {
        "num_images": len(image_paths),
        "num_label_files": len(label_paths),
        "num_objects": num_objects,
        "empty_labels_count": empty_labels_count,
        "area_px2": _safe_describe(areas_px2),
        "ratios": {
            "small_ratio": small_ratio,
            "medium_ratio": medium_ratio,
            "large_ratio": large_ratio,
            "tiny_ratio": tiny_ratio,
        },
        "density": {
            "objects_per_image": _safe_describe(objects_per_image),
            "objects_per_mpix": _safe_describe(objects_per_mpix),
        },
        "class_distribution": {
            "counts": {str(key): int(value) for key, value in sorted(class_counts.items())},
            "small_counts": {str(key): int(value) for key, value in sorted(small_class_counts.items())},
            "imbalance_ratio": _imbalance_ratio(class_counts),
            "imbalance_ratio_small": _imbalance_ratio(small_class_counts),
        },
        "image_size": {
            "width": _safe_describe(widths),
            "height": _safe_describe(heights),
            "aspect_ratio": _safe_describe(aspect_ratios),
        },
        "illumination": {
            "v_mean": _safe_describe(v_means),
            "v_std": _safe_describe(v_stds),
            "contrast": _safe_describe(contrasts),
        },
    }

    LOGGER.info(
        "Split '%s': images=%d, objects=%d, small_ratio=%.4f, tiny_ratio=%.4f",
        split,
        split_stats["num_images"],
        split_stats["num_objects"],
        split_stats["ratios"]["small_ratio"],
        split_stats["ratios"]["tiny_ratio"],
    )
    return split_stats


def _save_flat_csv(stats: dict, output_csv_path: Path) -> None:
    rows = []
    for split_name, split_stats in stats["splits"].items():
        row = {
            "split": split_name,
            "num_images": split_stats["num_images"],
            "num_label_files": split_stats["num_label_files"],
            "num_objects": split_stats["num_objects"],
            "empty_labels_count": split_stats["empty_labels_count"],
        }
        for key, value in flatten_dict(split_stats).items():
            if key in row:
                continue
            row[key] = value
        rows.append(row)
    dump_rows_to_csv(rows, output_csv_path)


def _save_plots(stats: dict, output_dir: Path) -> None:
    """
    Save basic visual diagnostics.

    We intentionally keep this optional and guarded by try/except, because plots are
    useful in Colab but should not break the whole pipeline when matplotlib is absent.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        LOGGER.warning("matplotlib is not available, skipping plots")
        return

    plots_dir = ensure_dir(output_dir / "plots")

    for split_name, split_stats in stats["splits"].items():
        area_stats = split_stats["area_px2"]
        density_stats = split_stats["density"]["objects_per_image"]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].bar(
            ["p10", "p50", "p90", "p99"],
            [
                area_stats["p10"],
                area_stats["p50"],
                area_stats["p90"],
                area_stats["p99"],
            ],
        )
        axes[0].set_title(f"{split_name}: area px^2 quantiles")
        axes[0].set_ylabel("Area px^2")

        axes[1].bar(
            ["min", "mean", "max"],
            [
                density_stats["min"],
                density_stats["mean"],
                density_stats["max"],
            ],
        )
        axes[1].set_title(f"{split_name}: objects per image")
        axes[1].set_ylabel("Count")

        plt.tight_layout()
        fig.savefig(plots_dir / f"{split_name}_quick_stats.png", dpi=150)
        plt.close(fig)


def analyze_dataset(
    dataset_root: str | Path,
    output_dir: str | Path = "artifacts/stats",
    splits: tuple[str, ...] = ("train", "val"),
    config: DatasetAnalyzerConfig | None = None,
) -> dict:
    """
    Analyze YOLO dataset and save:
    - dataset_stats.json
    - dataset_stats.csv
    - plots/*.png (optional)
    """
    if config is None:
        config = DatasetAnalyzerConfig()

    dataset_root = Path(dataset_root)
    output_dir = ensure_dir(output_dir)

    stats = {
        "schema_version": "1.0.0",
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "thresholds": {
            "small_max_area": float(config.small_max_area),
            "medium_max_area": float(config.medium_max_area),
            "tiny_max_area": float(config.tiny_max_area),
        },
        "splits": {},
    }

    for split in splits:
        images_dir = dataset_root / "images" / split
        labels_dir = dataset_root / "labels" / split
        if not images_dir.exists():
            raise FileNotFoundError(f"Missing split images directory: {images_dir.as_posix()}")
        if not labels_dir.exists():
            raise FileNotFoundError(f"Missing split labels directory: {labels_dir.as_posix()}")

        stats["splits"][split] = _analyze_split(
            images_dir=images_dir,
            labels_dir=labels_dir,
            split=split,
            config=config,
        )

    validate_stats_payload(stats)

    stats_json_path = output_dir / "dataset_stats.json"
    stats_csv_path = output_dir / "dataset_stats.csv"

    dump_json(stats, stats_json_path)
    _save_flat_csv(stats, stats_csv_path)
    if config.generate_plots:
        _save_plots(stats, output_dir=output_dir)

    LOGGER.info("Saved dataset stats JSON: %s", stats_json_path.as_posix())
    LOGGER.info("Saved dataset stats CSV: %s", stats_csv_path.as_posix())
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze YOLO dataset for MVP stats.")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to YOLO dataset root")
    parser.add_argument("--output-dir", type=str, default="artifacts/stats", help="Where to save stats")
    parser.add_argument("--small-max-area", type=float, default=SMALL_MAX_AREA_DEFAULT)
    parser.add_argument("--medium-max-area", type=float, default=MEDIUM_MAX_AREA_DEFAULT)
    parser.add_argument("--tiny-max-area", type=float, default=TINY_MAX_AREA_DEFAULT)
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = DatasetAnalyzerConfig(
        small_max_area=args.small_max_area,
        medium_max_area=args.medium_max_area,
        tiny_max_area=args.tiny_max_area,
        generate_plots=not args.no_plots,
    )
    analyze_dataset(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        config=config,
    )


if __name__ == "__main__":
    main()
