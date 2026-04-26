from pathlib import Path
import shutil

from src.evaluation.predict_runner import pick_best_run_by_val_metric


def _write_results_csv(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["epoch,metrics/mAP50-95(B)"]
    for idx, value in enumerate(values):
        lines.append(f"{idx},{value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _touch_weights(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_pick_best_run_by_val_metric_prefers_highest_score() -> None:
    tmp_root = Path("artifacts") / "_test_tmp" / "predict_runner_case"
    shutil.rmtree(tmp_root, ignore_errors=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    runs_root = tmp_root / "runs"

    _touch_weights(runs_root / "baseline" / "weights" / "best.pt")
    _write_results_csv(runs_root / "baseline" / "results.csv", [0.12, 0.15, 0.14])

    _touch_weights(runs_root / "adaptive" / "weights" / "best.pt")
    _write_results_csv(runs_root / "adaptive" / "results.csv", [0.20, 0.28, 0.31])

    # Missing weights should be ignored even if results.csv exists.
    _write_results_csv(runs_root / "manual" / "results.csv", [0.99])

    best = pick_best_run_by_val_metric(
        runs_root=runs_root,
        run_names=["baseline", "adaptive", "manual"],
    )

    assert best is not None
    assert best.run_name == "adaptive"
    assert abs(best.score - 0.31) < 1e-9
    assert best.weights_path.name == "best.pt"
