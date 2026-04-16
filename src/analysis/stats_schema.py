from __future__ import annotations

from numbers import Number


SMALL_MAX_AREA_DEFAULT = 32.0**2
MEDIUM_MAX_AREA_DEFAULT = 96.0**2
TINY_MAX_AREA_DEFAULT = 16.0**2


class StatsSchemaError(ValueError):
    """Raised when dataset_stats.json does not match expected MVP schema."""


def area_bucket(
    area_px2: float,
    small_max_area: float = SMALL_MAX_AREA_DEFAULT,
    medium_max_area: float = MEDIUM_MAX_AREA_DEFAULT,
) -> str:
    """Map area in px^2 to COCO-like bins."""
    if area_px2 <= small_max_area:
        return "small"
    if area_px2 <= medium_max_area:
        return "medium"
    return "large"


def is_tiny(area_px2: float, tiny_max_area: float = TINY_MAX_AREA_DEFAULT) -> bool:
    """Return whether area belongs to tiny subset used for internal reports."""
    return area_px2 <= tiny_max_area


def _require_keys(payload: dict, keys: set[str], section: str) -> None:
    missing = keys - set(payload.keys())
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise StatsSchemaError(f"Missing keys in '{section}': {missing_list}")


def _assert_numeric(payload: dict, keys: set[str], section: str) -> None:
    for key in keys:
        value = payload.get(key)
        if not isinstance(value, Number):
            raise StatsSchemaError(f"Expected numeric value for '{section}.{key}', got {type(value)}")


def validate_stats_payload(stats: dict) -> None:
    """
    Validate the minimal strict schema required by MVP.

    This is intentionally explicit to catch silent regressions in generated stats.
    """
    top_level_required = {"schema_version", "generated_at", "thresholds", "splits"}
    _require_keys(stats, top_level_required, section="root")

    thresholds_required = {"small_max_area", "medium_max_area", "tiny_max_area"}
    thresholds = stats["thresholds"]
    if not isinstance(thresholds, dict):
        raise StatsSchemaError("'thresholds' must be a dictionary")
    _require_keys(thresholds, thresholds_required, section="thresholds")
    _assert_numeric(thresholds, thresholds_required, section="thresholds")

    splits = stats["splits"]
    if not isinstance(splits, dict) or not splits:
        raise StatsSchemaError("'splits' must be a non-empty dictionary")

    split_required = {
        "num_images",
        "num_label_files",
        "num_objects",
        "empty_labels_count",
        "area_px2",
        "ratios",
        "density",
        "class_distribution",
        "image_size",
        "illumination",
    }

    describe_required = {
        "min",
        "max",
        "mean",
        "median",
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
        "p99",
    }

    ratio_required = {"small_ratio", "medium_ratio", "large_ratio", "tiny_ratio"}
    class_dist_required = {"counts", "small_counts", "imbalance_ratio", "imbalance_ratio_small"}

    for split_name, split_stats in splits.items():
        if not isinstance(split_stats, dict):
            raise StatsSchemaError(f"Split '{split_name}' must be a dictionary")
        _require_keys(split_stats, split_required, section=f"splits.{split_name}")

        _assert_numeric(
            split_stats,
            {"num_images", "num_label_files", "num_objects", "empty_labels_count"},
            section=f"splits.{split_name}",
        )

        for section_name in ("area_px2",):
            section_payload = split_stats[section_name]
            if not isinstance(section_payload, dict):
                raise StatsSchemaError(f"'{section_name}' in split '{split_name}' must be dict")
            _require_keys(section_payload, describe_required, section=f"splits.{split_name}.{section_name}")
            _assert_numeric(section_payload, describe_required, section=f"splits.{split_name}.{section_name}")

        ratios = split_stats["ratios"]
        if not isinstance(ratios, dict):
            raise StatsSchemaError(f"'ratios' in split '{split_name}' must be dict")
        _require_keys(ratios, ratio_required, section=f"splits.{split_name}.ratios")
        _assert_numeric(ratios, ratio_required, section=f"splits.{split_name}.ratios")

        density = split_stats["density"]
        if not isinstance(density, dict):
            raise StatsSchemaError(f"'density' in split '{split_name}' must be dict")
        for key in ("objects_per_image", "objects_per_mpix"):
            payload = density.get(key)
            if not isinstance(payload, dict):
                raise StatsSchemaError(f"'{key}' in split '{split_name}.density' must be dict")
            _require_keys(payload, describe_required, section=f"splits.{split_name}.density.{key}")
            _assert_numeric(payload, describe_required, section=f"splits.{split_name}.density.{key}")

        class_distribution = split_stats["class_distribution"]
        if not isinstance(class_distribution, dict):
            raise StatsSchemaError(f"'class_distribution' in split '{split_name}' must be dict")
        _require_keys(class_distribution, class_dist_required, section=f"splits.{split_name}.class_distribution")

        if not isinstance(class_distribution["counts"], dict):
            raise StatsSchemaError(f"'counts' in split '{split_name}.class_distribution' must be dict")
        if not isinstance(class_distribution["small_counts"], dict):
            raise StatsSchemaError(f"'small_counts' in split '{split_name}.class_distribution' must be dict")
        _assert_numeric(
            class_distribution,
            {"imbalance_ratio", "imbalance_ratio_small"},
            section=f"splits.{split_name}.class_distribution",
        )

        image_size = split_stats["image_size"]
        if not isinstance(image_size, dict):
            raise StatsSchemaError(f"'image_size' in split '{split_name}' must be dict")
        for key in ("width", "height", "aspect_ratio"):
            payload = image_size.get(key)
            if not isinstance(payload, dict):
                raise StatsSchemaError(f"'{key}' in split '{split_name}.image_size' must be dict")
            _require_keys(payload, describe_required, section=f"splits.{split_name}.image_size.{key}")
            _assert_numeric(payload, describe_required, section=f"splits.{split_name}.image_size.{key}")

        illumination = split_stats["illumination"]
        if not isinstance(illumination, dict):
            raise StatsSchemaError(f"'illumination' in split '{split_name}' must be dict")
        for key in ("v_mean", "v_std", "contrast"):
            payload = illumination.get(key)
            if not isinstance(payload, dict):
                raise StatsSchemaError(f"'{key}' in split '{split_name}.illumination' must be dict")
            _require_keys(payload, describe_required, section=f"splits.{split_name}.illumination.{key}")
            _assert_numeric(payload, describe_required, section=f"splits.{split_name}.illumination.{key}")

