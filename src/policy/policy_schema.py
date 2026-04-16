from __future__ import annotations

from numbers import Number


class PolicySchemaError(ValueError):
    """Raised when policy structure is invalid."""


# We keep this explicit to avoid leaking unsupported fields into YAML.
ULTRALYTICS_DETECT_ARG_WHITELIST = {
    "mosaic",
    "close_mosaic",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "perspective",
    "fliplr",
    "flipud",
    "mixup",
    "cutmix",
}


def filter_ultralytics_detect_args(args: dict) -> dict:
    """Filter policy args down to Ultralytics detect-compatible scalar settings."""
    return {key: args[key] for key in ULTRALYTICS_DETECT_ARG_WHITELIST if key in args}


def validate_policy_dict(policy: dict) -> None:
    """Validate adaptive policy format produced by the rule engine."""
    required = {"policy_name", "ultralytics_args", "albumentations_spec", "metadata"}
    missing = required - set(policy.keys())
    if missing:
        raise PolicySchemaError(f"Policy is missing required keys: {sorted(missing)}")

    ultralytics_args = policy["ultralytics_args"]
    if not isinstance(ultralytics_args, dict):
        raise PolicySchemaError("'ultralytics_args' must be dictionary")

    for key, value in ultralytics_args.items():
        if key not in ULTRALYTICS_DETECT_ARG_WHITELIST:
            raise PolicySchemaError(f"Unsupported Ultralytics key in policy: '{key}'")
        if not isinstance(value, Number):
            raise PolicySchemaError(f"Ultralytics arg '{key}' must be numeric, got {type(value)}")

    albumentations_spec = policy["albumentations_spec"]
    if not isinstance(albumentations_spec, list):
        raise PolicySchemaError("'albumentations_spec' must be a list")
    for item in albumentations_spec:
        if not isinstance(item, dict):
            raise PolicySchemaError("Each item in 'albumentations_spec' must be a dictionary")
        if "name" not in item:
            raise PolicySchemaError("Each augmentation spec must include key 'name'")
        if "p" in item and not isinstance(item["p"], Number):
            raise PolicySchemaError("'p' in augmentation spec must be numeric")

