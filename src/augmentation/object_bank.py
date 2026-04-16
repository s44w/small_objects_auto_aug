from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from src.data.yolo_label_reader import load_yolo_labels, yolo_bbox_to_xyxy_px
from src.utils.io import dump_json, load_json


@dataclass
class ObjectBankEntry:
    """Metadata for one object patch candidate."""

    image_path: str
    class_id: int
    bbox_xyxy: list[int]
    area_px2: float
    is_small: bool
    is_tiny: bool


class ObjectBank:
    """
    Object bank for bbox copy-paste augmentation.

    For MVP, we store metadata only and read patch pixels lazily on demand.
    """

    def __init__(
        self,
        small_max_area: float = 32.0**2,
        tiny_max_area: float = 16.0**2,
        max_items_per_class: int = 2000,
        seed: int = 42,
    ) -> None:
        self.small_max_area = float(small_max_area)
        self.tiny_max_area = float(tiny_max_area)
        self.max_items_per_class = int(max_items_per_class)
        self._rng = random.Random(seed)
        self.entries: list[ObjectBankEntry] = []
        self._by_class: dict[int, list[int]] = defaultdict(list)

    @property
    def size(self) -> int:
        return len(self.entries)

    def add_entry(self, entry: ObjectBankEntry) -> None:
        class_bucket = self._by_class[entry.class_id]
        if len(class_bucket) >= self.max_items_per_class:
            return
        class_bucket.append(len(self.entries))
        self.entries.append(entry)

    def build_from_dataset(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        image_extensions: Iterable[str] = (".jpg", ".jpeg", ".png", ".bmp"),
    ) -> None:
        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)
        ext_set = {ext.lower() for ext in image_extensions}

        image_paths = sorted(
            [
                path
                for path in images_dir.iterdir()
                if path.is_file() and path.suffix.lower() in ext_set
            ]
        )

        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            h, w = image.shape[:2]
            label_path = labels_dir / f"{image_path.stem}.txt"
            boxes = load_yolo_labels(label_path)

            for bbox in boxes:
                x1, y1, x2, y2 = yolo_bbox_to_xyxy_px(bbox=bbox, image_width=w, image_height=h)
                x1i = int(max(0, min(w - 1, round(x1))))
                y1i = int(max(0, min(h - 1, round(y1))))
                x2i = int(max(0, min(w, round(x2))))
                y2i = int(max(0, min(h, round(y2))))

                width = max(0, x2i - x1i)
                height = max(0, y2i - y1i)
                area = float(width * height)
                if area <= 0:
                    continue

                entry = ObjectBankEntry(
                    image_path=image_path.as_posix(),
                    class_id=int(bbox.class_id),
                    bbox_xyxy=[x1i, y1i, x2i, y2i],
                    area_px2=area,
                    is_small=area <= self.small_max_area,
                    is_tiny=area <= self.tiny_max_area,
                )
                self.add_entry(entry)

    def sample_entry(
        self,
        preferred_classes: set[int] | None = None,
        prefer_small: bool = False,
        prefer_tiny: bool = False,
    ) -> ObjectBankEntry | None:
        if not self.entries:
            return None

        candidates = self.entries

        if preferred_classes:
            class_filtered = [e for e in candidates if e.class_id in preferred_classes]
            if class_filtered:
                candidates = class_filtered

        if prefer_tiny:
            tiny_candidates = [e for e in candidates if e.is_tiny]
            if tiny_candidates:
                candidates = tiny_candidates
        elif prefer_small:
            small_candidates = [e for e in candidates if e.is_small]
            if small_candidates:
                candidates = small_candidates

        if not candidates:
            return None
        return self._rng.choice(candidates)

    def extract_patch(self, entry: ObjectBankEntry) -> np.ndarray | None:
        image = cv2.imread(entry.image_path)
        if image is None:
            return None
        x1, y1, x2, y2 = entry.bbox_xyxy
        patch = image[y1:y2, x1:x2]
        if patch.size == 0:
            return None
        return patch.copy()

    def save(self, output_path: str | Path) -> None:
        payload = {
            "small_max_area": self.small_max_area,
            "tiny_max_area": self.tiny_max_area,
            "entries": [asdict(entry) for entry in self.entries],
        }
        dump_json(payload, output_path)

    @classmethod
    def load(cls, path: str | Path, seed: int = 42) -> "ObjectBank":
        payload = load_json(path)
        bank = cls(
            small_max_area=float(payload["small_max_area"]),
            tiny_max_area=float(payload["tiny_max_area"]),
            seed=seed,
        )
        for item in payload["entries"]:
            bank.add_entry(ObjectBankEntry(**item))
        return bank

