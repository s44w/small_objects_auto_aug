from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.augmentation.albumentations_transforms import build_custom_transforms
from src.augmentation.object_bank import ObjectBank
from src.policy.policy_schema import validate_policy_dict
from src.utils.io import dump_json, dump_yaml, load_json


@dataclass
class AugmentationPolicy:
    """
    Policy adapter that exposes:
    - Ultralytics scalar args (YAML-safe)
    - Custom augmentation callables (Python API only)
    """

    payload: dict[str, Any]

    def __post_init__(self) -> None:
        validate_policy_dict(self.payload)

    @property
    def policy_name(self) -> str:
        return str(self.payload["policy_name"])

    def get_ultralytics_train_args(self) -> dict[str, float]:
        return dict(self.payload.get("ultralytics_args", {}))

    def get_albumentations_spec(self) -> list[dict[str, Any]]:
        return list(self.payload.get("albumentations_spec", []))

    def get_albumentations_transforms(
        self,
        object_bank: ObjectBank | None = None,
        seed: int = 42,
    ) -> list[Any]:
        return build_custom_transforms(
            albumentations_spec=self.get_albumentations_spec(),
            object_bank=object_bank,
            seed=seed,
        )

    def save_yaml(self, output_path: str | Path) -> None:
        dump_yaml(self.get_ultralytics_train_args(), output_path)

    def save_json(self, output_path: str | Path) -> None:
        dump_json(self.payload, output_path)

    @classmethod
    def from_json(cls, path: str | Path) -> "AugmentationPolicy":
        data = load_json(path)
        return cls(payload=data)

