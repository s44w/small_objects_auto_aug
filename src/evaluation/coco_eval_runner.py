from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.utils.io import dump_json


def _compute_ap_from_cocoeval(coco_eval, area_idx: int = 0, max_det_idx: int = -1) -> float:
    """
    Compute AP directly from COCOeval precision tensor.
    precision shape: [T, R, K, A, M]
    """
    precision = coco_eval.eval["precision"]  # noqa: N806
    area_precision = precision[:, :, :, area_idx, max_det_idx]
    valid = area_precision[area_precision > -1]
    if valid.size == 0:
        return 0.0
    return float(np.mean(valid))


def _compute_ar_from_cocoeval(coco_eval, area_idx: int = 0, max_det_idx: int = -1) -> float:
    """
    Compute AR directly from COCOeval recall tensor.
    recall shape: [T, K, A, M]
    """
    recall = coco_eval.eval["recall"]  # noqa: N806
    area_recall = recall[:, :, area_idx, max_det_idx]
    valid = area_recall[area_recall > -1]
    if valid.size == 0:
        return 0.0
    return float(np.mean(valid))


def run_coco_eval(
    coco_gt_path: str | Path,
    coco_dt_path: str | Path,
    output_path: str | Path | None = None,
    use_tiny_eval: bool = True,
    tiny_max_area: float = 16.0**2,
) -> dict:
    """
    Run COCOeval for bbox task and return summary metrics.
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except Exception as exc:
        raise RuntimeError("pycocotools is required for COCO evaluation.") from exc

    coco_gt = COCO(str(coco_gt_path))
    coco_dt = coco_gt.loadRes(str(coco_dt_path))

    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    metrics = {
        "AP@[0.5:0.95]": float(evaluator.stats[0]),
        "AP50": float(evaluator.stats[1]),
        "AP75": float(evaluator.stats[2]),
        "AP_small": float(evaluator.stats[3]),
        "AP_medium": float(evaluator.stats[4]),
        "AP_large": float(evaluator.stats[5]),
        "AR_small": float(evaluator.stats[9]),
    }

    if use_tiny_eval:
        tiny_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        tiny_eval.params.areaRng = [[0.0, float(tiny_max_area)]]
        tiny_eval.params.areaRngLbl = ["tiny"]
        tiny_eval.evaluate()
        tiny_eval.accumulate()

        metrics["AP_tiny"] = _compute_ap_from_cocoeval(tiny_eval, area_idx=0)
        metrics["AR_tiny"] = _compute_ar_from_cocoeval(tiny_eval, area_idx=0)
        metrics["tiny_definition_px2_max"] = float(tiny_max_area)
        metrics["tiny_note"] = "Non-standard extension. Not part of default COCO summary."

    if output_path is not None:
        dump_json(metrics, output_path)

    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run COCOeval for detection outputs.")
    parser.add_argument("--coco-gt", type=str, required=True)
    parser.add_argument("--coco-dt", type=str, required=True)
    parser.add_argument("--output", type=str, default="artifacts/eval/coco_eval.json")
    parser.add_argument("--no-tiny", action="store_true", help="Disable tiny extension eval")
    parser.add_argument("--tiny-max-area", type=float, default=16.0**2)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    metrics = run_coco_eval(
        coco_gt_path=args.coco_gt,
        coco_dt_path=args.coco_dt,
        output_path=args.output,
        use_tiny_eval=not args.no_tiny,
        tiny_max_area=args.tiny_max_area,
    )
    print(metrics)


if __name__ == "__main__":
    main()

