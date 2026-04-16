from __future__ import annotations

from pathlib import Path

from src.utils.io import ensure_dir


def build_markdown_report(
    metrics_by_run: dict[str, dict[str, float | str]],
    output_path: str | Path = "artifacts/reports/mvp_report.md",
) -> Path:
    """
    Build a compact markdown report with comparable metrics across runs.
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    columns = [
        "AP@[0.5:0.95]",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR_small",
        "AP_tiny",
    ]

    lines: list[str] = []
    lines.append("# MVP Metrics Report")
    lines.append("")
    lines.append("| run | " + " | ".join(columns) + " |")
    lines.append("|---|" + "|".join(["---"] * len(columns)) + "|")

    for run_name, metrics in metrics_by_run.items():
        values = []
        for column in columns:
            value = metrics.get(column, "-")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + run_name + " | " + " | ".join(values) + " |")

    lines.append("")
    lines.append("Notes:")
    lines.append("- AP_small is standard COCOeval small range metric.")
    lines.append("- AP_tiny is optional non-standard extension with area <= 16^2 px.")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path

