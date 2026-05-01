from __future__ import annotations

from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "images" / "architecture-flow.png"

STEPS = [
    "Входной датасет и project_config.yaml",
    "Подготовка и валидация датасета",
    "Анализ датасета",
    "dataset_stats.json / dataset_stats.csv",
    "Rule engine и генерация adaptive policy",
    "policy_adaptive.json / policy_adaptive.yaml / decision_report.json",
    "Подготовка runtime data yaml и object bank",
    "Training suite: baseline / manual / adaptive / ablation",
    "Предсказания на validation split",
    "YOLO -> COCO conversion",
    "COCOeval и AP_small / AP_tiny",
    "mvp_report.md / experiment_manifest.json",
]


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                Path("C:/Windows/Fonts/arialbd.ttf"),
                Path("C:/Windows/Fonts/calibrib.ttf"),
            ]
        )
    candidates.extend(
        [
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/calibri.ttf"),
            Path("C:/Windows/Fonts/tahoma.ttf"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def wrap_lines(text: str, width: int = 28) -> str:
    return "\n".join(wrap(text, width=width, break_long_words=False, break_on_hyphens=False))


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    width = 2200
    margin_x = 240
    box_width = width - margin_x * 2
    box_height = 180
    vertical_gap = 80
    top_margin = 140
    bottom_margin = 140
    arrow_height = 34
    height = top_margin + bottom_margin + len(STEPS) * box_height + (len(STEPS) - 1) * vertical_gap

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    font = load_font(42)
    arrow_font = load_font(52, bold=True)

    border_color = (0, 0, 0)
    fill_color = (245, 245, 245)
    text_color = (0, 0, 0)
    arrow_color = (0, 0, 0)

    for index, raw_text in enumerate(STEPS):
        x0 = margin_x
        y0 = top_margin + index * (box_height + vertical_gap)
        x1 = x0 + box_width
        y1 = y0 + box_height

        draw.rounded_rectangle((x0, y0, x1, y1), radius=24, fill=fill_color, outline=border_color, width=4)

        text = wrap_lines(raw_text)
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=8, align="center")
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        tx = x0 + (box_width - text_w) / 2
        ty = y0 + (box_height - text_h) / 2 - 4
        draw.multiline_text((tx, ty), text, font=font, fill=text_color, spacing=8, align="center")

        if index < len(STEPS) - 1:
            center_x = width / 2
            line_y0 = y1
            line_y1 = y1 + vertical_gap - arrow_height
            draw.line((center_x, line_y0, center_x, line_y1), fill=arrow_color, width=6)
            arrow_base_y = line_y1
            draw.polygon(
                [
                    (center_x, arrow_base_y + arrow_height),
                    (center_x - 20, arrow_base_y),
                    (center_x + 20, arrow_base_y),
                ],
                fill=arrow_color,
            )

    title = "Схема работы программного конвейера adaptive augmentations"
    title_font = load_font(48, bold=True)
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_x = (width - (title_bbox[2] - title_bbox[0])) / 2
    draw.text((title_x, 40), title, font=title_font, fill=text_color)

    image.save(OUTPUT)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
