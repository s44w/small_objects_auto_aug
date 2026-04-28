from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from src.data.yolo_label_reader import load_yolo_labels, yolo_bbox_to_xyxy_px
from src.utils.io import dump_json


@dataclass
class TilingConfig:
    """Offline YOLO tiling options for large aerial images."""

    tile_size: int = 1024
    overlap: int = 256
    min_visibility: float = 0.30
    include_empty: bool = False
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")


def _tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _intersect_xyxy(a: list[float], b: list[float]) -> list[float] | None:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _area_xyxy(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _to_yolo_line(class_id: int, box_xyxy: list[float], tile_x: int, tile_y: int, tile_w: int, tile_h: int) -> str:
    x1, y1, x2, y2 = box_xyxy
    x1 -= tile_x
    x2 -= tile_x
    y1 -= tile_y
    y2 -= tile_y
    xc = ((x1 + x2) / 2.0) / max(1.0, float(tile_w))
    yc = ((y1 + y2) / 2.0) / max(1.0, float(tile_h))
    bw = (x2 - x1) / max(1.0, float(tile_w))
    bh = (y2 - y1) / max(1.0, float(tile_h))
    return f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"


def tile_yolo_split(
    images_dir: str | Path,
    labels_dir: str | Path,
    output_root: str | Path,
    split: str,
    config: TilingConfig | None = None,
) -> dict:
    """
    Create tiled YOLO images/labels for one split.

    Bboxes are clipped to tile bounds and kept only when visible area / original
    area is at least `min_visibility`.
    """
    if config is None:
        config = TilingConfig()
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_root = Path(output_root)
    out_images = output_root / "images" / split
    out_labels = output_root / "labels" / split
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    ext_set = {ext.lower() for ext in config.image_extensions}
    image_paths = sorted(path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in ext_set)
    stride = max(1, int(config.tile_size) - int(config.overlap))

    written_tiles = 0
    written_objects = 0
    skipped_empty = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        boxes = load_yolo_labels(labels_dir / f"{image_path.stem}.txt")
        boxes_xyxy = [
            (
                bbox.class_id,
                list(yolo_bbox_to_xyxy_px(bbox=bbox, image_width=width, image_height=height)),
            )
            for bbox in boxes
        ]

        for y in _tile_starts(height, config.tile_size, stride):
            for x in _tile_starts(width, config.tile_size, stride):
                tile_w = min(config.tile_size, width - x)
                tile_h = min(config.tile_size, height - y)
                tile_box = [float(x), float(y), float(x + tile_w), float(y + tile_h)]
                label_lines: list[str] = []

                for class_id, box in boxes_xyxy:
                    clipped = _intersect_xyxy(list(map(float, box)), tile_box)
                    if clipped is None:
                        continue
                    visibility = _area_xyxy(clipped) / max(1e-9, _area_xyxy(list(map(float, box))))
                    if visibility < config.min_visibility:
                        continue
                    label_lines.append(_to_yolo_line(class_id, clipped, x, y, tile_w, tile_h))

                if not label_lines and not config.include_empty:
                    skipped_empty += 1
                    continue

                tile_name = f"{image_path.stem}_x{x}_y{y}{image_path.suffix.lower()}"
                cv2.imwrite(str(out_images / tile_name), image[y : y + tile_h, x : x + tile_w])
                (out_labels / f"{Path(tile_name).stem}.txt").write_text("".join(label_lines), encoding="utf-8")
                written_tiles += 1
                written_objects += len(label_lines)

    report = {
        "split": split,
        "config": asdict(config),
        "num_source_images": len(image_paths),
        "num_tiles": written_tiles,
        "num_objects": written_objects,
        "skipped_empty_tiles": skipped_empty,
        "output_root": output_root.as_posix(),
    }
    dump_json(report, output_root / f"tiling_report_{split}.json")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tile a YOLO split for overhead small-object datasets.")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--min-visibility", type=float, default=0.30)
    parser.add_argument("--include-empty", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = tile_yolo_split(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        output_root=args.output_root,
        split=args.split,
        config=TilingConfig(
            tile_size=args.tile_size,
            overlap=args.overlap,
            min_visibility=args.min_visibility,
            include_empty=args.include_empty,
        ),
    )
    print(report)


if __name__ == "__main__":
    main()
