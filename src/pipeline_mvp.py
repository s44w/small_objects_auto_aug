from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.dataset_analyzer import DatasetAnalyzerConfig, analyze_dataset
from src.data.visdrone_manager import save_validation_report, validate_visdrone_yolo_structure
from src.evaluation.coco_converter import convert_yolo_gt_to_coco, convert_yolo_pred_txt_to_coco
from src.evaluation.coco_eval_runner import run_coco_eval
from src.evaluation.metrics_report import build_markdown_report
from src.policy.rule_engine import RuleEngineConfig, generate_policy_from_stats, save_policy_artifacts
from src.training.train_runner import TrainRunConfig, run_mvp_training_suite
from src.utils.io import load_yaml
from src.utils.logging import configure_logging, get_logger


LOGGER = get_logger(__name__)


def run_mvp_pipeline(
    project_config_path: str | Path = "configs/project_config.yaml",
    run_training: bool = False,
    run_eval: bool = False,
    pred_labels_dir: str | Path | None = None,
) -> dict:
    """
    End-to-end MVP pipeline:
      1) Validate dataset structure
      2) Analyze dataset
      3) Build adaptive policy
      4) (Optional) run training suite
      5) (Optional) run COCO conversion + evaluation
    """
    cfg = load_yaml(project_config_path)

    dataset_root = Path(cfg["dataset"]["root"])
    splits = tuple(cfg["dataset"].get("splits", ["train", "val"]))
    image_ext = tuple(cfg["dataset"].get("image_extensions", [".jpg", ".jpeg", ".png", ".bmp"]))

    validation_report = validate_visdrone_yolo_structure(
        dataset_root=dataset_root,
        splits=splits,
        image_extensions=image_ext,
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

    outputs: dict = {
        "validation_report": "artifacts/stats/validation_report.json",
        "dataset_stats_json": "artifacts/stats/dataset_stats.json",
        "dataset_stats_csv": "artifacts/stats/dataset_stats.csv",
        "policy_json": policy_paths["policy_json"].as_posix(),
        "policy_yaml": policy_paths["policy_yaml"].as_posix(),
        "decision_report": policy_paths["decision_report"].as_posix(),
    }

    if run_training:
        train_cfg = TrainRunConfig(
            data_yaml=str(cfg["training"]["data_yaml"]),
            model=str(cfg["training"]["model"]),
            epochs=int(cfg["training"]["epochs_final"]),
            imgsz=int(cfg["training"]["imgsz"]),
            batch=int(cfg["training"]["batch"]),
            device=cfg["training"].get("device"),
            workers=int(cfg["training"]["workers"]),
            fraction=float(cfg["training"].get("fraction_fast", 1.0)),
            project_dir=str(cfg["training"]["project_dir"]),
            seed=int(cfg["project"]["seed"]),
            deterministic=bool(cfg["project"]["deterministic"]),
            rect=bool(cfg["training"].get("rect", False)),
            multi_scale=bool(cfg["training"].get("multi_scale", False)),
            baseline_disable_albumentations=bool(
                cfg["training"].get("baseline_disable_albumentations", True)
            ),
        )
        run_dirs = run_mvp_training_suite(
            config=train_cfg,
            baseline_yaml_path="configs/baseline.yaml",
            manual_yaml_path="configs/manual.yaml",
            adaptive_policy_json_path="artifacts/policy/policy_adaptive.json",
            run_ablation=True,
        )
        outputs["train_runs"] = run_dirs

    if run_eval:
        if pred_labels_dir is None:
            raise ValueError("pred_labels_dir must be provided when run_eval=True")

        class_names = cfg["dataset"]["visdrone_classes"]
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
        report_path = build_markdown_report({"adaptive": metrics})
        outputs["coco_eval_json"] = "artifacts/eval/coco_eval.json"
        outputs["mvp_report"] = report_path.as_posix()

    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MVP pipeline for adaptive augmentations.")
    parser.add_argument("--project-config", type=str, default="configs/project_config.yaml")
    parser.add_argument("--run-training", action="store_true")
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--pred-labels-dir", type=str, default=None)
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
    )
    LOGGER.info("Pipeline finished. Outputs: %s", outputs)


if __name__ == "__main__":
    main()

