from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BestRunSelection:
    """Selected training run for evaluation inference."""

    run_name: str
    run_dir: Path
    weights_path: Path
    score: float


def _max_metric_from_results_csv(results_csv: Path, metric_column: str) -> float:
    if not results_csv.exists():
        return -1.0

    best = -1.0
    with results_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            raw_value = row.get(metric_column)
            if raw_value is None:
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if value > best:
                best = value
    return best


def pick_best_run_by_val_metric(
    runs_root: str | Path,
    run_names: list[str] | None = None,
    metric_column: str = "metrics/mAP50-95(B)",
) -> BestRunSelection | None:
    """
    Pick best run by validation metric from Ultralytics run directories.

    Each run is expected at:
      runs_root/<run_name>/
        - weights/best.pt
        - results.csv
    """
    runs_root = Path(runs_root)
    if run_names is None:
        run_names = [path.name for path in runs_root.iterdir() if path.is_dir()]

    best: BestRunSelection | None = None
    for run_name in run_names:
        run_dir = runs_root / run_name
        weights_path = run_dir / "weights" / "best.pt"
        if not weights_path.exists():
            continue

        score = _max_metric_from_results_csv(run_dir / "results.csv", metric_column=metric_column)
        candidate = BestRunSelection(
            run_name=run_name,
            run_dir=run_dir,
            weights_path=weights_path,
            score=score,
        )
        if best is None or candidate.score > best.score:
            best = candidate
    return best


def predict_yolo_val_labels(
    weights_path: str | Path,
    images_dir: str | Path,
    output_project: str | Path,
    run_name: str,
    imgsz: int = 640,
    device: str | int | None = 0,
    conf: float = 0.001,
    iou: float = 0.6,
    max_det: int = 500,
    use_tta: bool = True,
) -> Path:
    """
    Run YOLO inference and save txt labels for subsequent COCO conversion.

    Returns:
        Path to directory with predicted label txt files.
    """
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("Ultralytics is required for inference prediction.") from exc

    model = YOLO(str(weights_path))
    model.predict(
        source=str(images_dir),
        device=device,
        imgsz=int(imgsz),
        conf=float(conf),
        iou=float(iou),
        max_det=int(max_det),
        augment=bool(use_tta),
        save=False,
        save_txt=True,
        save_conf=True,
        project=str(output_project),
        name=str(run_name),
        exist_ok=True,
        verbose=False,
    )
    labels_dir = Path(output_project) / str(run_name) / "labels"
    if not labels_dir.exists():
        raise FileNotFoundError(f"Prediction labels directory not found: {labels_dir.as_posix()}")
    return labels_dir
