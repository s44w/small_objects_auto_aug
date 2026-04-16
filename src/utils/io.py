from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as Path."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def load_json(path: str | Path) -> Any:
    """Load JSON from disk."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def dump_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Store JSON with UTF-8 encoding and stable key order."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=indent, sort_keys=True)


def load_yaml(path: str | Path) -> Any:
    """Load YAML from disk."""
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def dump_yaml(data: Any, path: str | Path) -> None:
    """Store YAML in a human-readable way."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)


def dump_rows_to_csv(rows: list[Mapping[str, Any]], path: str | Path) -> None:
    """Write a list of flat dictionaries as CSV."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    if not rows:
        with output_path.open("w", encoding="utf-8") as file:
            file.write("")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def flatten_dict(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested mappings into a single-level dict with dotted keys."""
    flat: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(flatten_dict(value, prefix=full_key))
        else:
            flat[full_key] = value
    return flat

