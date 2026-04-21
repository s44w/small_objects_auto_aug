from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

from src.augmentation.policy_to_ultralytics import AugmentationPolicy
from src.policy.policy_schema import filter_ultralytics_detect_args
from src.utils.io import dump_yaml, load_json, load_yaml
from src.utils.logging import get_logger
from src.utils.reproducibility import set_seed


LOGGER = get_logger(__name__)


@dataclass
class TrainRunConfig:
    """Common train arguments shared across baseline/manual/adaptive runs."""

    data_yaml: str
    model: str = "yolo26n.pt"
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    device: str | int | None = None
    workers: int = 4
    fraction: float = 1.0
    project_dir: str = "runs"
    seed: int = 42
    deterministic: bool = True
    rect: bool = False
    multi_scale: bool = False
    baseline_disable_albumentations: bool = True
    plots: bool = False


def _to_yaml_safe(value: Any) -> Any:
    """Convert nested runtime objects to YAML-safe values for debug dumps."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return value.as_posix()

    if isinstance(value, dict):
        return {str(key): _to_yaml_safe(val) for key, val in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_yaml_safe(item) for item in value]

    if is_dataclass(value):
        payload: dict[str, Any] = {"__class__": value.__class__.__name__}
        for field in fields(value):
            # Skip private runtime internals (e.g. compiled objects, RNGs).
            if field.name.startswith("_"):
                continue
            try:
                field_value = getattr(value, field.name)
            except Exception:
                continue
            payload[field.name] = _to_yaml_safe(field_value)
        return payload

    # Fallback for callables/complex objects (Albumentations/custom transform wrappers).
    return {
        "__class__": value.__class__.__name__,
        "repr": repr(value),
    }


def _ultralytics_yolo():
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "Ultralytics is required for training. Install with `pip install ultralytics`."
        ) from exc
    return YOLO


def _default_train_args(config: TrainRunConfig) -> dict[str, Any]:
    args = {
        "data": config.data_yaml,
        "epochs": config.epochs,
        "imgsz": config.imgsz,
        "batch": config.batch,
        "device": config.device,
        "workers": config.workers,
        "fraction": config.fraction,
        "project": config.project_dir,
        "seed": config.seed,
        "deterministic": config.deterministic,
        "rect": config.rect,
        "multi_scale": config.multi_scale,
        "plots": config.plots,
        "exist_ok": False,
    }
    return args


def run_train_mode(
    mode: str,
    config: TrainRunConfig,
    mode_args: dict[str, Any],
    custom_augmentations: list[Any] | None = None,
) -> Path:
    """
    Run one training mode using Ultralytics Python API.

    mode in {"baseline", "manual", "adaptive", ...}
    """
    set_seed(config.seed, deterministic=config.deterministic)
    YOLO = _ultralytics_yolo()
    model = YOLO(config.model)

    train_args = _default_train_args(config=config)
    train_args["name"] = mode
    train_args.update(filter_ultralytics_detect_args(mode_args))

    if mode == "baseline" and config.baseline_disable_albumentations:
        # Empty list disables default Albumentations branch in Ultralytics.
        train_args["augmentations"] = []

    if custom_augmentations is not None:
        train_args["augmentations"] = custom_augmentations

    try:
        results = model.train(**train_args)
    except TypeError as exc:
        # Some Ultralytics versions may reject "augmentations".
        if "augmentations" not in str(exc):
            raise
        LOGGER.warning(
            "Current Ultralytics build does not accept 'augmentations'. "
            "Retrying mode '%s' without custom transforms.",
            mode,
        )
        train_args.pop("augmentations", None)
        results = model.train(**train_args)

    save_dir = Path(getattr(results, "save_dir", Path(config.project_dir) / mode))
    save_dir.mkdir(parents=True, exist_ok=True)
    try:
        dump_yaml(_to_yaml_safe(train_args), save_dir / "train_args.yaml")
    except Exception as exc:
        LOGGER.warning(
            "Failed to dump train args for mode '%s' due to serialization error: %s",
            mode,
            exc,
        )
    LOGGER.info("Mode '%s' completed. Artifacts saved to %s", mode, save_dir.as_posix())
    return save_dir


def run_mvp_training_suite(
    config: TrainRunConfig,
    baseline_yaml_path: str | Path,
    manual_yaml_path: str | Path,
    adaptive_policy_json_path: str | Path,
    run_ablation: bool = True,
) -> dict[str, str]:
    """
    Run baseline, manual, adaptive and two ablations:
    - adaptive_no_mosaic
    - adaptive_no_custom_albu
    """
    baseline_args = load_yaml(baseline_yaml_path)
    manual_args = load_yaml(manual_yaml_path)
    adaptive_payload = load_json(adaptive_policy_json_path)
    adaptive_policy = AugmentationPolicy(payload=adaptive_payload)

    run_dirs: dict[str, str] = {}
    run_dirs["baseline"] = run_train_mode(
        mode="baseline",
        config=config,
        mode_args=baseline_args,
        custom_augmentations=None,
    ).as_posix()

    run_dirs["manual"] = run_train_mode(
        mode="manual",
        config=config,
        mode_args=manual_args,
        custom_augmentations=None,
    ).as_posix()

    run_dirs["adaptive"] = run_train_mode(
        mode="adaptive",
        config=config,
        mode_args=adaptive_policy.get_ultralytics_train_args(),
        custom_augmentations=adaptive_policy.get_albumentations_transforms(seed=config.seed),
    ).as_posix()

    if run_ablation:
        adaptive_args_no_mosaic = dict(adaptive_policy.get_ultralytics_train_args())
        adaptive_args_no_mosaic["mosaic"] = 0.0
        run_dirs["adaptive_no_mosaic"] = run_train_mode(
            mode="adaptive_no_mosaic",
            config=config,
            mode_args=adaptive_args_no_mosaic,
            custom_augmentations=adaptive_policy.get_albumentations_transforms(seed=config.seed + 1),
        ).as_posix()

        run_dirs["adaptive_no_custom_albu"] = run_train_mode(
            mode="adaptive_no_custom_albu",
            config=config,
            mode_args=adaptive_policy.get_ultralytics_train_args(),
            custom_augmentations=None,
        ).as_posix()

    return run_dirs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MVP training modes.")
    parser.add_argument("--data-yaml", type=str, required=True)
    parser.add_argument("--model", type=str, default="yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--project-dir", type=str, default="runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--baseline-yaml", type=str, default="configs/baseline.yaml")
    parser.add_argument("--manual-yaml", type=str, default="configs/manual.yaml")
    parser.add_argument("--adaptive-policy-json", type=str, default="artifacts/policy/policy_adaptive.json")
    parser.add_argument("--no-ablation", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = TrainRunConfig(
        data_yaml=args.data_yaml,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        fraction=args.fraction,
        project_dir=args.project_dir,
        seed=args.seed,
        deterministic=bool(args.deterministic),
    )
    LOGGER.info("Train config: %s", asdict(config))
    run_dirs = run_mvp_training_suite(
        config=config,
        baseline_yaml_path=args.baseline_yaml,
        manual_yaml_path=args.manual_yaml,
        adaptive_policy_json_path=args.adaptive_policy_json,
        run_ablation=not args.no_ablation,
    )
    LOGGER.info("Completed training modes: %s", run_dirs)


if __name__ == "__main__":
    main()
