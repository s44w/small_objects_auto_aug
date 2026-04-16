from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.policy.policy_schema import filter_ultralytics_detect_args, validate_policy_dict
from src.utils.io import dump_json, dump_yaml, load_json, load_yaml
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)


@dataclass
class RuleEngineConfig:
    """Thresholds and switches for rule-based adaptive policy generation."""

    small_ratio_threshold: float = 0.5
    dense_objects_per_image_threshold: float = 15.0
    dense_objects_per_mpix_threshold: float = 30.0
    illum_v_std_threshold: float = 35.0
    imbalance_ratio_threshold: float = 10.0
    small_imbalance_ratio_threshold: float = 6.0
    allow_flipud: bool = False
    enable_mixup_cutmix: bool = False
    low_variability_vstd_threshold: float = 14.0
    bbox_crop_min_visibility: float = 0.30
    bbox_crop_min_area: float = 16.0
    bbox_copy_paste_ioa_threshold: float = 0.30
    bbox_copy_paste_max_pastes: int = 3

    @classmethod
    def from_project_config(cls, project_config: dict | None) -> "RuleEngineConfig":
        if not project_config:
            return cls()
        policy_cfg = project_config.get("policy", {})
        return cls(
            small_ratio_threshold=float(policy_cfg.get("small_ratio_threshold", 0.5)),
            dense_objects_per_image_threshold=float(
                policy_cfg.get("dense_objects_per_image_threshold", 15.0)
            ),
            dense_objects_per_mpix_threshold=float(
                policy_cfg.get("dense_objects_per_mpix_threshold", 30.0)
            ),
            illum_v_std_threshold=float(policy_cfg.get("illum_v_std_threshold", 35.0)),
            imbalance_ratio_threshold=float(policy_cfg.get("imbalance_ratio_threshold", 10.0)),
            small_imbalance_ratio_threshold=float(
                policy_cfg.get("small_imbalance_ratio_threshold", 6.0)
            ),
            allow_flipud=bool(policy_cfg.get("allow_flipud", False)),
            enable_mixup_cutmix=bool(policy_cfg.get("enable_mixup_cutmix", False)),
            low_variability_vstd_threshold=float(
                policy_cfg.get("low_variability_vstd_threshold", 14.0)
            ),
            bbox_crop_min_visibility=float(policy_cfg.get("bbox_crop_min_visibility", 0.3)),
            bbox_crop_min_area=float(policy_cfg.get("bbox_crop_min_area", 16.0)),
            bbox_copy_paste_ioa_threshold=float(
                policy_cfg.get("bbox_copy_paste_ioa_threshold", 0.3)
            ),
            bbox_copy_paste_max_pastes=int(policy_cfg.get("bbox_copy_paste_max_pastes", 3)),
        )


def _extract_train_features(stats: dict) -> dict[str, float]:
    """Extract only the features needed by MVP rule set."""
    if "train" not in stats["splits"]:
        raise KeyError("dataset_stats.json must contain split 'train'")

    train_stats = stats["splits"]["train"]
    features = {
        "small_ratio": float(train_stats["ratios"]["small_ratio"]),
        "tiny_ratio": float(train_stats["ratios"]["tiny_ratio"]),
        "objects_per_image_mean": float(train_stats["density"]["objects_per_image"]["mean"]),
        "objects_per_mpix_mean": float(train_stats["density"]["objects_per_mpix"]["mean"]),
        "illum_v_std_mean": float(train_stats["illumination"]["v_std"]["mean"]),
        "imbalance_ratio": float(train_stats["class_distribution"]["imbalance_ratio"]),
        "imbalance_ratio_small": float(train_stats["class_distribution"]["imbalance_ratio_small"]),
    }
    return features


def _compute_flags(features: dict[str, float], cfg: RuleEngineConfig) -> dict[str, bool]:
    """Convert numeric features to categorical flags used by interpretable rules."""
    return {
        "is_small_heavy": features["small_ratio"] >= cfg.small_ratio_threshold,
        "is_dense": (
            features["objects_per_image_mean"] >= cfg.dense_objects_per_image_threshold
            or features["objects_per_mpix_mean"] >= cfg.dense_objects_per_mpix_threshold
        ),
        "is_illum_var_high": features["illum_v_std_mean"] >= cfg.illum_v_std_threshold,
        "is_imbalanced": features["imbalance_ratio"] >= cfg.imbalance_ratio_threshold,
        "is_small_imbalanced": (
            features["imbalance_ratio_small"] >= cfg.small_imbalance_ratio_threshold
        ),
        "is_low_variability": features["illum_v_std_mean"] <= cfg.low_variability_vstd_threshold,
    }


def _record_change(
    report_items: list[dict[str, Any]],
    rule_name: str,
    conditions: dict[str, Any],
    key: str,
    before: Any,
    after: Any,
) -> None:
    if before == after:
        return
    report_items.append(
        {
            "rule_name": rule_name,
            "conditions": conditions,
            "parameter": key,
            "before": before,
            "after": after,
        }
    )


def generate_policy_from_stats(
    stats: dict,
    cfg: RuleEngineConfig | None = None,
) -> tuple[dict, dict]:
    """
    Build adaptive policy dict and explainable decision report from dataset stats.
    """
    if cfg is None:
        cfg = RuleEngineConfig()

    features = _extract_train_features(stats)
    flags = _compute_flags(features, cfg)
    changes: list[dict[str, Any]] = []

    ultralytics_args = {
        "mosaic": 0.3,
        "close_mosaic": 10,
        "hsv_h": 0.015,
        "hsv_s": 0.50,
        "hsv_v": 0.40,
        "degrees": 10.0,
        "translate": 0.10,
        "scale": 0.50,
        "perspective": 0.0,
        "fliplr": 0.5,
        "flipud": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
    }

    # R_mosaic
    mosaic_before = ultralytics_args["mosaic"]
    ultralytics_args["mosaic"] = 0.7 if flags["is_dense"] else 0.3
    _record_change(
        report_items=changes,
        rule_name="R_mosaic",
        conditions={"is_dense": flags["is_dense"]},
        key="mosaic",
        before=mosaic_before,
        after=ultralytics_args["mosaic"],
    )

    # R_geom_small_safe
    if flags["is_small_heavy"]:
        for key, new_value in {"degrees": 5.0, "translate": 0.05, "scale": 0.30, "perspective": 0.0}.items():
            before = ultralytics_args[key]
            ultralytics_args[key] = min(before, new_value) if key != "perspective" else new_value
            _record_change(
                report_items=changes,
                rule_name="R_geom_small_safe",
                conditions={"is_small_heavy": True},
                key=key,
                before=before,
                after=ultralytics_args[key],
            )

    # R_flip
    before_flipud = ultralytics_args["flipud"]
    ultralytics_args["flipud"] = 0.1 if cfg.allow_flipud else 0.0
    _record_change(
        report_items=changes,
        rule_name="R_flip",
        conditions={"allow_flipud": cfg.allow_flipud},
        key="flipud",
        before=before_flipud,
        after=ultralytics_args["flipud"],
    )

    # R_photo
    if flags["is_illum_var_high"]:
        for key, delta in {"hsv_s": 0.10, "hsv_v": 0.10}.items():
            before = ultralytics_args[key]
            ultralytics_args[key] = min(1.0, before + delta)
            _record_change(
                report_items=changes,
                rule_name="R_photo",
                conditions={"is_illum_var_high": True},
                key=key,
                before=before,
                after=ultralytics_args[key],
            )

    # R_mixup_cutmix
    if cfg.enable_mixup_cutmix and flags["is_low_variability"] and not flags["is_small_heavy"]:
        for key in ("mixup", "cutmix"):
            before = ultralytics_args[key]
            ultralytics_args[key] = 0.10
            _record_change(
                report_items=changes,
                rule_name="R_mixup_cutmix",
                conditions={
                    "enable_mixup_cutmix": cfg.enable_mixup_cutmix,
                    "is_low_variability": flags["is_low_variability"],
                    "is_small_heavy": flags["is_small_heavy"],
                },
                key=key,
                before=before,
                after=ultralytics_args[key],
            )

    # Albumentations spec for Python API.
    albumentations_spec = [
        {
            "name": "BBoxAwareCrop",
            "p": 0.60 if flags["is_small_heavy"] else 0.30,
            "params": {
                "height": 640,
                "width": 640,
                "min_visibility": cfg.bbox_crop_min_visibility,
                "min_area": cfg.bbox_crop_min_area,
            },
        },
        {
            "name": "BBoxCopyPaste",
            "p": 0.30 if flags["is_small_imbalanced"] else 0.10,
            "params": {
                "ioa_threshold": cfg.bbox_copy_paste_ioa_threshold,
                "max_pastes": cfg.bbox_copy_paste_max_pastes,
                "prefer_small": flags["is_small_imbalanced"],
            },
        },
    ]

    policy = {
        "policy_name": "adaptive_policy_mvp",
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "ultralytics_args": filter_ultralytics_detect_args(ultralytics_args),
        "albumentations_spec": albumentations_spec,
        "metadata": {
            "features": features,
            "flags": flags,
            "schema_version": "1.0.0",
        },
    }
    validate_policy_dict(policy)

    decision_report = {
        "policy_name": policy["policy_name"],
        "generated_at": policy["generated_at"],
        "features": features,
        "flags": flags,
        "fired_rules": changes,
    }
    return policy, decision_report


def save_policy_artifacts(
    policy: dict,
    decision_report: dict,
    output_dir: str | Path = "artifacts/policy",
) -> dict[str, Path]:
    """Persist adaptive policy as JSON + YAML and save explainability report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_json_path = output_dir / "policy_adaptive.json"
    policy_yaml_path = output_dir / "policy_adaptive.yaml"
    decision_path = output_dir / "decision_report.json"

    dump_json(policy, policy_json_path)
    dump_yaml(policy["ultralytics_args"], policy_yaml_path)
    dump_json(decision_report, decision_path)

    return {
        "policy_json": policy_json_path,
        "policy_yaml": policy_yaml_path,
        "decision_report": decision_path,
    }


def run_rule_engine_from_paths(
    dataset_stats_path: str | Path,
    output_dir: str | Path = "artifacts/policy",
    project_config_path: str | Path | None = None,
) -> tuple[dict, dict]:
    """Convenience wrapper for CLI/Notebook usage."""
    stats = load_json(dataset_stats_path)
    config = RuleEngineConfig()
    if project_config_path is not None:
        project_cfg = load_yaml(project_config_path)
        config = RuleEngineConfig.from_project_config(project_cfg)

    policy, decision_report = generate_policy_from_stats(stats, cfg=config)
    save_policy_artifacts(policy=policy, decision_report=decision_report, output_dir=output_dir)
    return policy, decision_report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate adaptive augmentation policy from dataset stats.")
    parser.add_argument("--dataset-stats", type=str, required=True, help="Path to dataset_stats.json")
    parser.add_argument("--output-dir", type=str, default="artifacts/policy")
    parser.add_argument("--project-config", type=str, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    policy, report = run_rule_engine_from_paths(
        dataset_stats_path=args.dataset_stats,
        output_dir=args.output_dir,
        project_config_path=args.project_config,
    )
    LOGGER.info(
        "Generated policy: %s, fired_rules=%d",
        policy["policy_name"],
        len(report.get("fired_rules", [])),
    )


if __name__ == "__main__":
    main()
