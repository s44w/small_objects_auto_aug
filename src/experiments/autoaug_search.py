from __future__ import annotations

import argparse
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.policy.policy_schema import filter_ultralytics_detect_args
from src.utils.io import dump_json, dump_yaml


@dataclass
class AutoAugCandidate:
    """One budget-aware AutoAug-style candidate policy."""

    candidate_id: str
    ultralytics_args: dict[str, float]
    search_space: str = "mvp_small_object_random_v1"


def generate_autoaug_candidates(num_candidates: int = 8, seed: int = 42) -> list[AutoAugCandidate]:
    """
    Generate a small random-search policy pool for budget-aware comparison.

    This is not full AutoAugment/RL. It is an explicit low-compute comparator:
    train several candidate policies under the same budget and select by val
    AP_small, then compare against the data-analysis rule policy.
    """
    rng = random.Random(seed)
    candidates: list[AutoAugCandidate] = []
    for idx in range(int(num_candidates)):
        args = {
            "mosaic": rng.choice([0.0, 0.3, 0.5, 0.7, 1.0]),
            "close_mosaic": 10,
            "hsv_h": rng.choice([0.0, 0.01, 0.015, 0.02]),
            "hsv_s": rng.choice([0.3, 0.5, 0.7, 0.9]),
            "hsv_v": rng.choice([0.2, 0.35, 0.45, 0.6]),
            "degrees": rng.choice([0.0, 3.0, 5.0, 10.0]),
            "translate": rng.choice([0.02, 0.05, 0.1, 0.15]),
            "scale": rng.choice([0.2, 0.3, 0.5, 0.7]),
            "perspective": 0.0,
            "fliplr": 0.5,
            "flipud": 0.0,
            "mixup": rng.choice([0.0, 0.05, 0.1]),
            "cutmix": rng.choice([0.0, 0.05, 0.1]),
        }
        candidates.append(
            AutoAugCandidate(
                candidate_id=f"autoaug_rs_{idx:02d}",
                ultralytics_args=filter_ultralytics_detect_args(args),
            )
        )
    return candidates


def save_autoaug_candidates(
    candidates: list[AutoAugCandidate],
    output_dir: str | Path = "artifacts/autoaug_candidates",
) -> dict[str, str]:
    """Save each candidate as YAML plus a manifest JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    manifest: list[dict[str, Any]] = []

    for candidate in candidates:
        yaml_path = output_dir / f"{candidate.candidate_id}.yaml"
        dump_yaml(candidate.ultralytics_args, yaml_path)
        paths[candidate.candidate_id] = yaml_path.as_posix()
        manifest.append({**asdict(candidate), "yaml_path": yaml_path.as_posix()})

    manifest_path = output_dir / "manifest.json"
    dump_json(manifest, manifest_path)
    paths["manifest"] = manifest_path.as_posix()
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate budget-aware AutoAug random-search candidates.")
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="artifacts/autoaug_candidates")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    candidates = generate_autoaug_candidates(num_candidates=args.num_candidates, seed=args.seed)
    paths = save_autoaug_candidates(candidates, output_dir=args.output_dir)
    print(paths)


if __name__ == "__main__":
    main()
