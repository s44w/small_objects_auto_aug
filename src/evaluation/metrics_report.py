from __future__ import annotations

from pathlib import Path

from src.utils.io import ensure_dir


def build_markdown_report(
    metrics_by_run: dict[str, dict[str, float | str]],
    output_path: str | Path = "artifacts/reports/mvp_report.md",
    timings: dict[str, float] | None = None,
    artifact_paths: dict[str, str] | None = None,
    run_dirs: dict[str, str] | None = None,
    title: str = "MVP Metrics Report",
) -> Path:
    """
    Build a compact markdown report with comparable metrics and experiment context.
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
    lines.append(f"# {title}")
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

    if run_dirs:
        lines.append("")
        lines.append("## Training Runs")
        lines.append("")
        lines.append("| run | path |")
        lines.append("|---|---|")
        for run_name, path in sorted(run_dirs.items()):
            lines.append(f"| {run_name} | `{path}` |")

    if timings:
        lines.append("")
        lines.append("## Timings")
        lines.append("")
        lines.append("| stage | seconds |")
        lines.append("|---|---:|")
        for stage, seconds in sorted(timings.items()):
            lines.append(f"| {stage} | {float(seconds):.2f} |")

    if artifact_paths:
        lines.append("")
        lines.append("## Artifacts")
        lines.append("")
        lines.append("| artifact | path |")
        lines.append("|---|---|")
        for name, path in sorted(artifact_paths.items()):
            lines.append(f"| {name} | `{path}` |")

    lines.append("")
    lines.append("Notes:")
    lines.append("- AP_small is standard COCOeval small range metric.")
    lines.append("- AP_tiny is optional non-standard extension with area <= 16^2 px.")
    lines.append("- Higher AP_small is the primary success criterion for this project.")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
