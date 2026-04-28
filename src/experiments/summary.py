from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from src.utils.io import dump_json, load_json


def summarize_metrics_mean_std(
    metrics_by_run: dict[str, dict[str, Any]],
    metric_names: tuple[str, ...] = ("AP_small", "AP@[0.5:0.95]", "AP50", "AP_tiny"),
) -> dict[str, dict[str, float]]:
    """
    Summarize repeated runs as mean/std/count per metric.

    The input is intentionally generic so it can be fed by multi-seed runs,
    AutoAug candidate runs, or hand-curated experiment groups.
    """
    summary: dict[str, dict[str, float]] = {}
    for metric_name in metric_names:
        values: list[float] = []
        for metrics in metrics_by_run.values():
            value = metrics.get(metric_name)
            if isinstance(value, (int, float)):
                values.append(float(value))
        if not values:
            continue
        summary[metric_name] = {
            "mean": float(statistics.fmean(values)),
            "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
            "count": float(len(values)),
        }
    return summary


def summarize_coco_eval_files(
    eval_files: list[str | Path],
    output_path: str | Path = "artifacts/reports/metrics_mean_std.json",
) -> dict[str, dict[str, float]]:
    metrics_by_run = {
        Path(path).parent.name: load_json(path)
        for path in eval_files
    }
    summary = summarize_metrics_mean_std(metrics_by_run)
    dump_json(summary, output_path)
    return summary
