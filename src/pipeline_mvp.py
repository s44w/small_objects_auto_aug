from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

from src.analysis.dataset_analyzer import DatasetAnalyzerConfig, analyze_dataset
from src.augmentation.object_bank import ObjectBank
from src.data.coco_small_manager import (
    CocoSmallPrepareConfig,
    prepare_coco_small_by_mode,
)
from src.data.visdrone_manager import (
    build_visdrone_scene_difficulty_report,
    prepare_dataset_by_mode,
    save_validation_report,
    validate_visdrone_yolo_structure,
)
from src.data.tiling import TilingConfig, tile_yolo_split
from src.evaluation.coco_converter import convert_yolo_gt_to_coco, convert_yolo_pred_txt_to_coco
from src.evaluation.coco_eval_runner import run_coco_eval
from src.evaluation.metrics_report import build_markdown_report
from src.evaluation.predict_runner import predict_yolo_val_labels
from src.policy.rule_engine import RuleEngineConfig, generate_policy_from_stats, save_policy_artifacts
from src.training.train_runner import (
    TrainRunConfig,
    run_mvp_training_suite,
    run_mvp_training_suite_multiseed,
)
from src.utils.io import dump_json, dump_yaml, load_yaml
from src.utils.logging import configure_logging, get_logger


LOGGER = get_logger(__name__)


def _resolve_project_root(project_config_path: str | Path) -> tuple[Path, Path]:
    """
    Resolve absolute config path and infer repository root.

    The root is detected as the nearest parent containing both `src/` and `configs/`.
    This makes pipeline execution independent from current working directory.
    """
    config_path = Path(project_config_path).expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    else:
        config_path = config_path.resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Project config not found: {config_path.as_posix()}")

    for candidate in [config_path.parent, *config_path.parents]:
        if (candidate / "src").exists() and (candidate / "configs").exists():
            return config_path, candidate

    raise FileNotFoundError(
        "Could not infer project root from project_config_path. "
        "Expected a parent directory containing both 'src/' and 'configs/'. "
        f"Config path: {config_path.as_posix()}"
    )


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
            "epochs": int(training_cfg.get("epochs_fast", 25)),
            "imgsz": int(training_cfg.get("imgsz", 960)),
            "batch": int(training_cfg.get("batch_fast", training_cfg.get("batch", 6))),
            "workers": int(training_cfg.get("workers_fast", training_cfg.get("workers", 2))),
            "fraction": float(training_cfg.get("fraction_fast", 1.0)),
            "multi_scale": bool(training_cfg.get("multi_scale_fast", False)),
        }
    if profile_norm == "final":
        return {
            "epochs": int(training_cfg.get("epochs_final", 25)),
            "imgsz": int(training_cfg.get("imgsz", 960)),
            "batch": int(training_cfg.get("batch", 6)),
            "workers": int(training_cfg.get("workers", 2)),
            "fraction": float(training_cfg.get("fraction_final", 1.0)),
            "multi_scale": bool(training_cfg.get("multi_scale", False)),
        }

    notebook_presets: dict[str, dict[str, Any]] = {
        "balanced": {
            "epochs": 25,
            "imgsz": 960,
            "batch": 6,
            "workers": 2,
            "fraction": 1.0,
            "multi_scale": False,
        },
        "quality": {
            "epochs": 25,
            "imgsz": 960,
            "batch": 6,
            "workers": 2,
            "fraction": 1.0,
            "multi_scale": False,
        },
        "hour": {
            "epochs": 25,
            "imgsz": 960,
            "batch": 6,
            "workers": 2,
            "fraction": 1.0,
            "multi_scale": False,
        },
        "max_quality": {
            "epochs": 25,
            "imgsz": 960,
            "batch": 6,
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


def _build_object_bank_artifact(
    dataset_root: Path,
    cfg: dict,
    image_extensions: tuple[str, ...],
    output_path: str | Path = "artifacts/policy/object_bank.json",
) -> Path:
    analysis_cfg = cfg["analysis"]
    policy_cfg = cfg.get("policy", {})
    bank_cfg = policy_cfg.get("object_bank", {})
    bank = ObjectBank(
        small_max_area=float(analysis_cfg["small_max_area"]),
        tiny_max_area=float(analysis_cfg["tiny_max_area"]),
        max_items_per_class=int(bank_cfg.get("max_items_per_class", 2000)),
        seed=int(cfg["project"].get("seed", 42)),
    )
    bank.build_from_dataset(
        images_dir=dataset_root / "images" / "train",
        labels_dir=dataset_root / "labels" / "train",
        image_extensions=image_extensions,
    )
    output_path = Path(output_path)
    bank.save(output_path)
    LOGGER.info("Object bank built: entries=%d, path=%s", bank.size, output_path.as_posix())
    return output_path


def _maybe_tile_yolo_dataset(
    dataset_cfg: dict,
    dataset_root: Path,
    raw_root: str | Path | None,
    splits: tuple[str, ...],
    image_extensions: tuple[str, ...],
) -> list[dict[str, Any]]:
    tiling_cfg = dataset_cfg.get("tiling", {})
    if not bool(tiling_cfg.get("enabled", False)):
        return []
    if raw_root is None:
        raise ValueError("dataset.raw_root must point to the source YOLO dataset when dataset.tiling.enabled=true.")
    raw_root_path = Path(raw_root)
    config = TilingConfig(
        tile_size=int(tiling_cfg.get("tile_size", 1024)),
        overlap=int(tiling_cfg.get("overlap", 256)),
        min_visibility=float(tiling_cfg.get("min_visibility", 0.30)),
        include_empty=bool(tiling_cfg.get("include_empty", False)),
        image_extensions=image_extensions,
    )
    reports: list[dict[str, Any]] = []
    for split in splits:
        reports.append(
            tile_yolo_split(
                images_dir=raw_root_path / "images" / split,
                labels_dir=raw_root_path / "labels" / split,
                output_root=dataset_root,
                split=split,
                config=config,
            )
        )
    return reports


def _evaluate_prediction_dir(
    run_name: str,
    pred_labels_dir: str | Path,
    dataset_root: Path,
    image_extensions: tuple[str, ...],
    class_names: list[str],
    cfg: dict,
) -> dict[str, Any]:
    eval_dir = Path("artifacts/eval") / run_name
    eval_dir.mkdir(parents=True, exist_ok=True)
    coco_gt_path = eval_dir / "coco_gt.json"
    coco_dt_path = eval_dir / "coco_dt.json"

    convert_yolo_gt_to_coco(
        images_dir=dataset_root / "images" / "val",
        labels_dir=dataset_root / "labels" / "val",
        class_names=class_names,
        output_path=coco_gt_path,
        image_extensions=image_extensions,
    )
    convert_yolo_pred_txt_to_coco(
        pred_labels_dir=pred_labels_dir,
        images_dir=dataset_root / "images" / "val",
        output_path=coco_dt_path,
        image_extensions=image_extensions,
    )
    metrics = run_coco_eval(
        coco_gt_path=coco_gt_path,
        coco_dt_path=coco_dt_path,
        output_path=eval_dir / "coco_eval.json",
        use_tiny_eval=bool(cfg["evaluation"].get("use_tiny_eval", True)),
        tiny_max_area=float(cfg["analysis"]["tiny_max_area"]),
    )
    metrics["coco_eval_path"] = (eval_dir / "coco_eval.json").as_posix()
    return metrics


def _evaluate_training_runs(
    cfg: dict,
    dataset_root: Path,
    image_extensions: tuple[str, ...],
    class_names: list[str],
    run_names: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    metrics_by_run: dict[str, dict[str, Any]] = {}
    pred_dirs: dict[str, str] = {}
    runs_root = Path(cfg["training"].get("project_dir", "runs"))

    for run_name in run_names:
        weights_path = runs_root / run_name / "weights" / "best.pt"
        if not weights_path.exists():
            LOGGER.warning("Skipping eval for '%s': missing weights at %s", run_name, weights_path.as_posix())
            continue

        pred_labels_dir = predict_yolo_val_labels(
            weights_path=weights_path,
            images_dir=dataset_root / "images" / "val",
            output_project=runs_root / "eval_predict",
            run_name=run_name,
            imgsz=int(cfg["evaluation"].get("imgsz", cfg["training"].get("imgsz", 960))),
            device=cfg["training"].get("device", 0),
            use_tta=bool(cfg["evaluation"].get("use_tta", True)),
        )
        pred_dirs[run_name] = Path(pred_labels_dir).as_posix()
        metrics_by_run[run_name] = _evaluate_prediction_dir(
            run_name=run_name,
            pred_labels_dir=pred_labels_dir,
            dataset_root=dataset_root,
            image_extensions=image_extensions,
            class_names=class_names,
            cfg=cfg,
        )
        LOGGER.info("COCOeval completed for run '%s'.", run_name)

    return metrics_by_run, pred_dirs


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
    timings: dict[str, float] = {}
    resolved_config_path, project_root = _resolve_project_root(project_config_path)
    cwd_before = Path.cwd().resolve()
    if cwd_before != project_root:
        LOGGER.info(
            "Switching working directory for pipeline execution: %s -> %s",
            cwd_before.as_posix(),
            project_root.as_posix(),
        )

    os.chdir(project_root)
    try:
        cfg = load_yaml(resolved_config_path)

        dataset_cfg = cfg["dataset"]
        dataset_root = Path(dataset_cfg["root"])
        dataset_kind = str(dataset_cfg.get("kind", "visdrone")).strip().lower()
        dataset_mode = str(dataset_cfg.get("mode", "manual"))
        raw_root = dataset_cfg.get("raw_root")
        splits = tuple(dataset_cfg.get("splits", ["train", "val"]))
        image_ext = tuple(dataset_cfg.get("image_extensions", [".jpg", ".jpeg", ".png", ".bmp"]))

        stage_start = time.perf_counter()
        tiling_reports = _maybe_tile_yolo_dataset(
            dataset_cfg=dataset_cfg,
            dataset_root=dataset_root,
            raw_root=raw_root,
            splits=splits,
            image_extensions=image_ext,
        )
        if tiling_reports:
            timings["tiling"] = time.perf_counter() - stage_start
            dump_json(tiling_reports, "artifacts/stats/tiling_reports.json")
            stage_start = time.perf_counter()

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
        elif dataset_kind == "yolo_generic":
            if dataset_mode != "manual":
                raise ValueError(
                    "dataset.kind='yolo_generic' currently supports only mode='manual'. "
                    "Provide prepared YOLO dataset in dataset.root."
                )
            validation_report = validate_visdrone_yolo_structure(
                dataset_root=dataset_root,
                splits=splits,
                image_extensions=image_ext,
            )
            validation_report["summary"] = {
                "mode": "manual",
                "kind": "yolo_generic",
            }
        else:
            raise ValueError(
                f"Unsupported dataset.kind='{dataset_kind}'. "
                "Expected one of: visdrone, coco_small, yolo_generic."
            )

        save_validation_report(validation_report, "artifacts/stats/validation_report.json")
        LOGGER.info("Dataset validation completed. is_valid=%s", validation_report["is_valid"])
        timings["dataset_validation"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
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
        timings["dataset_analysis"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        rule_cfg = RuleEngineConfig.from_project_config(cfg)
        policy, decision_report = generate_policy_from_stats(stats, cfg=rule_cfg)
        policy_paths = save_policy_artifacts(
            policy=policy,
            decision_report=decision_report,
            output_dir="artifacts/policy",
        )
        LOGGER.info("Adaptive policy generated: %s", policy_paths)
        timings["policy_generation"] = time.perf_counter() - stage_start

        outputs: dict[str, Any] = {
            "validation_report": "artifacts/stats/validation_report.json",
            "dataset_stats_json": "artifacts/stats/dataset_stats.json",
            "dataset_stats_csv": "artifacts/stats/dataset_stats.csv",
            "policy_json": policy_paths["policy_json"].as_posix(),
            "policy_yaml": policy_paths["policy_yaml"].as_posix(),
            "decision_report": policy_paths["decision_report"].as_posix(),
        }
        if tiling_reports:
            outputs["tiling_reports"] = "artifacts/stats/tiling_reports.json"
        artifact_paths: dict[str, str] = {
            "validation_report": outputs["validation_report"],
            "dataset_stats_json": outputs["dataset_stats_json"],
            "dataset_stats_csv": outputs["dataset_stats_csv"],
            "policy_json": outputs["policy_json"],
            "policy_yaml": outputs["policy_yaml"],
            "decision_report": outputs["decision_report"],
        }
        if tiling_reports:
            artifact_paths["tiling_reports"] = outputs["tiling_reports"]

        if dataset_kind == "visdrone" and raw_root and Path(str(raw_root)).exists():
            scene_report_path = Path("artifacts/stats/visdrone_scene_difficulty.json")
            build_visdrone_scene_difficulty_report(
                raw_visdrone_root=raw_root,
                output_path=scene_report_path,
            )
            outputs["scene_difficulty_report"] = scene_report_path.as_posix()
            artifact_paths["scene_difficulty_report"] = scene_report_path.as_posix()

        object_bank_path: Path | None = None
        if bool(cfg.get("policy", {}).get("use_object_bank", True)):
            stage_start = time.perf_counter()
            object_bank_path = _build_object_bank_artifact(
                dataset_root=dataset_root,
                cfg=cfg,
                image_extensions=image_ext,
            )
            timings["object_bank_build"] = time.perf_counter() - stage_start
            outputs["object_bank"] = object_bank_path.as_posix()
            artifact_paths["object_bank"] = object_bank_path.as_posix()

        if run_training:
            stage_start = time.perf_counter()
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
                require_custom_augmentations=bool(
                    training_cfg.get("require_custom_augmentations", True)
                ),
                val=bool(training_cfg.get("val_during_train", True)),
                cache=training_cfg.get("cache", False),
                save_period=int(training_cfg.get("save_period", -1)),
                patience=(
                    int(training_cfg["patience"])
                    if training_cfg.get("patience") is not None
                    else None
                ),
            )
            mode_overrides = training_cfg.get("mode_overrides")
            seeds = [int(seed) for seed in training_cfg.get("seeds", [])]
            if seeds:
                run_dirs_by_seed = run_mvp_training_suite_multiseed(
                    config=train_cfg,
                    seeds=seeds,
                    baseline_yaml_path="configs/baseline.yaml",
                    manual_yaml_path="configs/manual.yaml",
                    adaptive_policy_json_path="artifacts/policy/policy_adaptive.json",
                    run_ablation=bool(training_cfg.get("run_ablation", True)),
                    object_bank_path=object_bank_path,
                    mode_overrides=mode_overrides if isinstance(mode_overrides, dict) else None,
                )
                outputs["train_runs_by_seed"] = run_dirs_by_seed
            else:
                run_dirs = run_mvp_training_suite(
                    config=train_cfg,
                    baseline_yaml_path="configs/baseline.yaml",
                    manual_yaml_path="configs/manual.yaml",
                    adaptive_policy_json_path="artifacts/policy/policy_adaptive.json",
                    run_ablation=bool(training_cfg.get("run_ablation", True)),
                    object_bank_path=object_bank_path,
                    mode_overrides=mode_overrides if isinstance(mode_overrides, dict) else None,
                )
                outputs["train_runs"] = run_dirs
            outputs["runtime_data_yaml"] = runtime_data_yaml.as_posix()
            outputs["train_profile"] = train_profile
            artifact_paths["runtime_data_yaml"] = runtime_data_yaml.as_posix()
            timings["training_suite"] = time.perf_counter() - stage_start

        if run_eval:
            stage_start = time.perf_counter()
            class_names = _dataset_class_names(cfg=cfg, dataset_root=dataset_root)
            if pred_labels_dir is not None:
                report_run_name = str(eval_run_name or "external_predictions")
                metrics_by_run = {
                    report_run_name: _evaluate_prediction_dir(
                        run_name=report_run_name,
                        pred_labels_dir=pred_labels_dir,
                        dataset_root=dataset_root,
                        image_extensions=image_ext,
                        class_names=class_names,
                        cfg=cfg,
                    )
                }
                outputs["pred_labels_dir"] = Path(pred_labels_dir).as_posix()
            elif auto_predict_for_eval:
                if eval_run_name is not None:
                    run_candidates = [eval_run_name]
                elif "train_runs" in outputs:
                    run_candidates = list(outputs["train_runs"].keys())
                else:
                    run_candidates = [
                        "baseline",
                        "manual",
                        "adaptive",
                        "adaptive_no_mosaic",
                        "adaptive_no_custom_albu",
                    ]
                metrics_by_run, pred_dirs = _evaluate_training_runs(
                    cfg=cfg,
                    dataset_root=dataset_root,
                    image_extensions=image_ext,
                    class_names=class_names,
                    run_names=run_candidates,
                )
                outputs["pred_labels_dirs"] = pred_dirs
                if not metrics_by_run:
                    raise ValueError(
                        "No evaluation metrics were produced. Check that run weights exist "
                        "or pass --pred-labels-dir."
                    )
            else:
                raise ValueError(
                    "pred_labels_dir must be provided when run_eval=True, "
                    "or enable auto_predict_for_eval with available run weights."
                )

            timings["evaluation_suite"] = time.perf_counter() - stage_start
            run_dirs_for_report = outputs.get("train_runs") if isinstance(outputs.get("train_runs"), dict) else None
            report_path = build_markdown_report(
                metrics_by_run,
                timings=timings,
                artifact_paths=artifact_paths,
                run_dirs=run_dirs_for_report,
            )
            outputs["eval_metrics_by_run"] = metrics_by_run
            outputs["mvp_report"] = report_path.as_posix()
            artifact_paths["mvp_report"] = report_path.as_posix()

        outputs["timings"] = timings
        manifest_path = Path("artifacts/reports/experiment_manifest.json")
        dump_json(
            {
                "project_config": resolved_config_path.as_posix(),
                "outputs": outputs,
                "artifacts": artifact_paths,
                "timings": timings,
            },
            manifest_path,
        )
        outputs["experiment_manifest"] = manifest_path.as_posix()

        return outputs
    finally:
        os.chdir(cwd_before)


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
