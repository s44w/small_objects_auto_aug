from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.analysis.dataset_analyzer import DatasetAnalyzerConfig, analyze_dataset
from src.data.coco_small_manager import (
    CocoSmallPrepareConfig,
    prepare_coco_small_by_mode,
)
from src.data.visdrone_manager import prepare_dataset_by_mode, save_validation_report
from src.evaluation.coco_converter import convert_yolo_gt_to_coco, convert_yolo_pred_txt_to_coco
from src.evaluation.coco_eval_runner import run_coco_eval
from src.evaluation.metrics_report import build_markdown_report
from src.evaluation.predict_runner import pick_best_run_by_val_metric, predict_yolo_val_labels
from src.policy.rule_engine import RuleEngineConfig, generate_policy_from_stats, save_policy_artifacts
from src.training.train_runner import TrainRunConfig, run_mvp_training_suite
from src.utils.io import dump_yaml, load_yaml
from src.utils.logging import configure_logging, get_logger


LOGGER = get_logger(__name__)


def _dataset_class_names(cfg: dict, dataset_root: Path) -> list[str]:
    dataset_cfg = cfg.get("dataset", {})
    if isinstance(dataset_cfg.get("class_names"), list):
        return [str(name) for name in dataset_cfg["class_names"]]
    if isinstance(dataset_cfg.get("visdrone_classes"), list):
        return [str(name) for name in dataset_cfg["visdrone_classes"]]

    data_yaml_path = cfg.get("training", {}).get("data_yaml")
    if data_yaml_path:
        path = Path(str(data_yaml_path))
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.exists():
            payload = load_yaml(path)
            names = payload.get("names", [])
            if isinstance(names, dict):
                return [str(name) for _, name in sorted(names.items(), key=lambda item: int(item[0]))]
            if isinstance(names, list):
                return [str(name) for name in names]

    # Fallback: derive from generated dataset yaml if present.
    for candidate in ("visdrone.yaml", "coco_small.yaml"):
        yaml_path = dataset_root / candidate
        if yaml_path.exists():
            payload = load_yaml(yaml_path)
            names = payload.get("names", [])
            if isinstance(names, dict):
                return [str(name) for _, name in sorted(names.items(), key=lambda item: int(item[0]))]
            if isinstance(names, list):
                return [str(name) for name in names]

    raise ValueError(
        "Could not resolve class names. "
        "Set dataset.class_names in config or provide training.data_yaml with names."
    )


def _resolve_training_profile(training_cfg: dict, profile: str) -> dict[str, Any]:
    """
    Resolve training profile into concrete train parameters.

    Supported profiles:
      - fast: quick sanity run
      - final: config-driven final run
      - balanced / quality / hour / max_quality: notebook-aligned presets
    """
    profile_norm = str(profile).strip().lower()
    if profile_norm == "fast":
        return {
            "epochs": int(training_cfg.get("epochs_fast", 20)),
            "imgsz": int(training_cfg.get("imgsz", 640)),
            "batch": int(training_cfg.get("batch_fast", min(8, int(training_cfg.get("batch", 16))))),
            "workers": int(training_cfg.get("workers_fast", min(2, int(training_cfg.get("workers", 4))))),
            "fraction": float(training_cfg.get("fraction_fast", 0.2)),
            "multi_scale": bool(training_cfg.get("multi_scale_fast", False)),
        }
    if profile_norm == "final":
        return {
            "epochs": int(training_cfg.get("epochs_final", 100)),
            "imgsz": int(training_cfg.get("imgsz", 640)),
            "batch": int(training_cfg.get("batch", 16)),
            "workers": int(training_cfg.get("workers", 4)),
            "fraction": float(training_cfg.get("fraction_final", 1.0)),
            "multi_scale": bool(training_cfg.get("multi_scale", False)),
        }

    notebook_presets: dict[str, dict[str, Any]] = {
        "balanced": {
            "epochs": 15,
            "imgsz": 768,
            "batch": 8,
            "workers": 2,
            "fraction": 1.0,
            "multi_scale": False,
        },
        "quality": {
            "epochs": 25,
            "imgsz": 896,
            "batch": 8,
            "workers": 2,
            "fraction": 1.0,
            "multi_scale": False,
        },
        "hour": {
            "epochs": 40,
            "imgsz": 960,
            "batch": 6,
            "workers": 2,
            "fraction": 1.0,
            "multi_scale": False,
        },
        "max_quality": {
            "epochs": 60,
            "imgsz": 1024,
            "batch": 4,
            "workers": 2,
            "fraction": 1.0,
            "multi_scale": False,
        },
    }
    if profile_norm in notebook_presets:
        return notebook_presets[profile_norm]

    raise ValueError(
        f"Unsupported train_profile='{profile}'. "
        "Expected one of: fast, final, balanced, quality, hour, max_quality."
    )


def _write_runtime_data_yaml(
    dataset_root: Path,
    class_names: list[str],
    output_path: str | Path = "artifacts/runtime_visdrone.yaml",
) -> Path:
    output_path = Path(output_path)
    payload = {
        "path": dataset_root.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {idx: name for idx, name in enumerate(class_names)},
    }
    dump_yaml(payload, output_path)
    return output_path


def run_mvp_pipeline(
    project_config_path: str | Path = "configs/project_config.yaml",
    run_training: bool = False,
    run_eval: bool = False,
    pred_labels_dir: str | Path | None = None,
    train_profile: str = "final",
    eval_run_name: str | None = None,
    auto_predict_for_eval: bool = True,
) -> dict:
    """
    End-to-end MVP pipeline:
      1) Prepare/validate dataset structure
      2) Analyze dataset
      3) Build adaptive policy
      4) (Optional) run training suite
      5) (Optional) run COCO conversion + evaluation
    """
    cfg = load_yaml(project_config_path)

    dataset_cfg = cfg["dataset"]
    dataset_root = Path(dataset_cfg["root"])
    dataset_kind = str(dataset_cfg.get("kind", "visdrone")).strip().lower()
    dataset_mode = str(dataset_cfg.get("mode", "manual"))
    raw_root = dataset_cfg.get("raw_root")
    splits = tuple(dataset_cfg.get("splits", ["train", "val"]))
    image_ext = tuple(dataset_cfg.get("image_extensions", [".jpg", ".jpeg", ".png", ".bmp"]))

    if dataset_kind == "visdrone":
        validation_report = prepare_dataset_by_mode(
            dataset_root=dataset_root,
            mode=dataset_mode,
            raw_visdrone_root=raw_root,
            splits=splits,
            image_extensions=image_ext,
        )
    elif dataset_kind == "coco_small":
        coco_cfg_payload = dataset_cfg.get("coco_small", {})
        coco_prepare_cfg = CocoSmallPrepareConfig(
            small_max_area=float(coco_cfg_payload.get("small_max_area", cfg["analysis"]["small_max_area"])),
            keep_images_without_small=bool(coco_cfg_payload.get("keep_images_without_small", False)),
            include_crowd=bool(coco_cfg_payload.get("include_crowd", False)),
            splits=splits,
            link_images=bool(coco_cfg_payload.get("link_images", True)),
            image_extensions=image_ext,
            selected_category_ids=coco_cfg_payload.get("selected_category_ids"),
        )
        validation_report = prepare_coco_small_by_mode(
            dataset_root=dataset_root,
            mode=dataset_mode,
            raw_coco_root=raw_root,
            config=coco_prepare_cfg,
        )
    else:
        raise ValueError(
            f"Unsupported dataset.kind='{dataset_kind}'. "
            "Expected one of: visdrone, coco_small."
        )

    save_validation_report(validation_report, "artifacts/stats/validation_report.json")
    LOGGER.info("Dataset validation completed. is_valid=%s", validation_report["is_valid"])

    analysis_cfg = DatasetAnalyzerConfig(
        small_max_area=float(cfg["analysis"]["small_max_area"]),
        medium_max_area=float(cfg["analysis"]["medium_max_area"]),
        tiny_max_area=float(cfg["analysis"]["tiny_max_area"]),
        image_extensions=image_ext,
        generate_plots=bool(cfg["analysis"].get("generate_plots", True)),
    )
    stats = analyze_dataset(
        dataset_root=dataset_root,
        output_dir="artifacts/stats",
        splits=splits,
        config=analysis_cfg,
    )

    rule_cfg = RuleEngineConfig.from_project_config(cfg)
    policy, decision_report = generate_policy_from_stats(stats, cfg=rule_cfg)
    policy_paths = save_policy_artifacts(
        policy=policy,
        decision_report=decision_report,
        output_dir="artifacts/policy",
    )
    LOGGER.info("Adaptive policy generated: %s", policy_paths)

    outputs: dict[str, Any] = {
        "validation_report": "artifacts/stats/validation_report.json",
        "dataset_stats_json": "artifacts/stats/dataset_stats.json",
        "dataset_stats_csv": "artifacts/stats/dataset_stats.csv",
        "policy_json": policy_paths["policy_json"].as_posix(),
        "policy_yaml": policy_paths["policy_yaml"].as_posix(),
        "decision_report": policy_paths["decision_report"].as_posix(),
    }

    if run_training:
        training_cfg = cfg["training"]
        train_profile_cfg = _resolve_training_profile(training_cfg, profile=train_profile)
        class_names = _dataset_class_names(cfg=cfg, dataset_root=dataset_root)
        runtime_data_yaml = _write_runtime_data_yaml(
            dataset_root=dataset_root,
            class_names=class_names,
        )
        train_cfg = TrainRunConfig(
            data_yaml=runtime_data_yaml.as_posix(),
            model=str(training_cfg["model"]),
            epochs=int(train_profile_cfg["epochs"]),
            imgsz=int(train_profile_cfg["imgsz"]),
            batch=int(train_profile_cfg["batch"]),
            device=training_cfg.get("device"),
            workers=int(train_profile_cfg["workers"]),
            fraction=float(train_profile_cfg["fraction"]),
            project_dir=str(training_cfg["project_dir"]),
            seed=int(cfg["project"]["seed"]),
            deterministic=bool(cfg["project"]["deterministic"]),
            rect=bool(training_cfg.get("rect", False)),
            multi_scale=bool(train_profile_cfg["multi_scale"]),
            baseline_disable_albumentations=bool(
                training_cfg.get("baseline_disable_albumentations", True)
            ),
        )
        run_dirs = run_mvp_training_suite(
            config=train_cfg,
            baseline_yaml_path="configs/baseline.yaml",
            manual_yaml_path="configs/manual.yaml",
            adaptive_policy_json_path="artifacts/policy/policy_adaptive.json",
            run_ablation=bool(training_cfg.get("run_ablation", True)),
        )
        outputs["train_runs"] = run_dirs
        outputs["runtime_data_yaml"] = runtime_data_yaml.as_posix()
        outputs["train_profile"] = train_profile

    if run_eval:
        if pred_labels_dir is None and auto_predict_for_eval:
            if eval_run_name is not None:
                run_candidates = [eval_run_name]
            elif "train_runs" in outputs:
                run_candidates = list(outputs["train_runs"].keys())
            else:
                run_candidates = [
                    "adaptive",
                    "manual",
                    "baseline",
                    "adaptive_no_mosaic",
                    "adaptive_no_custom_albu",
                ]

            runs_root = Path(cfg["training"].get("project_dir", "runs"))
            best_run = pick_best_run_by_val_metric(runs_root=runs_root, run_names=run_candidates)
            if best_run is None:
                raise ValueError(
                    "pred_labels_dir is not provided and no suitable run weights were found for auto-predict."
                )

            pred_labels_dir = predict_yolo_val_labels(
                weights_path=best_run.weights_path,
                images_dir=dataset_root / "images" / "val",
                output_project=runs_root / "eval_predict",
                run_name=best_run.run_name,
                imgsz=int(cfg["evaluation"].get("imgsz", cfg["training"].get("imgsz", 640))),
                device=cfg["training"].get("device", 0),
                use_tta=bool(cfg["evaluation"].get("use_tta", True)),
            )
            outputs["eval_run_used"] = best_run.run_name
            outputs["pred_labels_dir"] = Path(pred_labels_dir).as_posix()
            LOGGER.info(
                "Evaluation predictions generated from run '%s' (score=%.5f).",
                best_run.run_name,
                best_run.score,
            )

        if pred_labels_dir is None:
            raise ValueError(
                "pred_labels_dir must be provided when run_eval=True, "
                "or enable auto_predict_for_eval with available run weights."
            )

        class_names = _dataset_class_names(cfg=cfg, dataset_root=dataset_root)
        images_val = dataset_root / "images" / "val"
        labels_val = dataset_root / "labels" / "val"
        coco_gt_path = Path("artifacts/eval/coco_gt.json")
        coco_dt_path = Path("artifacts/eval/coco_dt.json")

        convert_yolo_gt_to_coco(
            images_dir=images_val,
            labels_dir=labels_val,
            class_names=class_names,
            output_path=coco_gt_path,
            image_extensions=image_ext,
        )
        convert_yolo_pred_txt_to_coco(
            pred_labels_dir=pred_labels_dir,
            images_dir=images_val,
            output_path=coco_dt_path,
            image_extensions=image_ext,
        )
        metrics = run_coco_eval(
            coco_gt_path=coco_gt_path,
            coco_dt_path=coco_dt_path,
            output_path="artifacts/eval/coco_eval.json",
            use_tiny_eval=bool(cfg["evaluation"].get("use_tiny_eval", True)),
            tiny_max_area=float(cfg["analysis"]["tiny_max_area"]),
        )
        report_run_name = str(outputs.get("eval_run_used", eval_run_name or "adaptive"))
        report_path = build_markdown_report({report_run_name: metrics})
        outputs["coco_eval_json"] = "artifacts/eval/coco_eval.json"
        outputs["mvp_report"] = report_path.as_posix()

    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MVP pipeline for adaptive augmentations.")
    parser.add_argument("--project-config", type=str, default="configs/project_config.yaml")
    parser.add_argument("--run-training", action="store_true")
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--pred-labels-dir", type=str, default=None)
    parser.add_argument("--train-profile", type=str, default="final")
    parser.add_argument("--eval-run-name", type=str, default=None)
    parser.add_argument("--no-auto-predict", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_logging(args.log_level)
    outputs = run_mvp_pipeline(
        project_config_path=args.project_config,
        run_training=args.run_training,
        run_eval=args.run_eval,
        pred_labels_dir=args.pred_labels_dir,
        train_profile=args.train_profile,
        eval_run_name=args.eval_run_name,
        auto_predict_for_eval=not args.no_auto_predict,
    )
    LOGGER.info("Pipeline finished. Outputs: %s", outputs)


if __name__ == "__main__":
    main()
