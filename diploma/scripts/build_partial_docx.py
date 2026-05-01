from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
ET.register_namespace("w", W_NS)

NS = {"w": W_NS}

STYLE_PARAGRAPH = "947"  # +Абзац с отступом 1-ой строки
STYLE_SPECIAL_HEADING = "971"  # +ЗАГОЛОВОК по центру
STYLE_CHAPTER = "979"  # +Заголовок 1 уровня
STYLE_SECTION = "980"  # +Заголовок 2 уровня
STYLE_TABLE_CAPTION = "975"  # +№ - Название таблицы
STYLE_FIGURE_CAPTION = "973"  # +№ - Название рисунка
STYLE_TABLE_TEXT = "989"  # +Текст в таблице


SPECIAL_H1 = {
    "введение",
    "заключение",
    "реферат",
    "содержание",
    "список использованных источников",
}


def w_tag(name: str) -> str:
    return f"{{{W_NS}}}{name}"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def normalize_inline(text: str) -> str:
    text = text.strip()
    text = text.replace("`", "")
    return text


def latex_to_text(expr: str) -> str:
    expr = expr.strip()
    expr = expr.replace(r"\_", "_")
    expr = expr.replace(r"\cdot", "·")
    expr = expr.replace(r"\geq", ">=")
    expr = expr.replace(r"\leq", "<=")
    expr = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1) / (\2)", expr)
    expr = re.sub(r"\s+", " ", expr)
    return expr.strip()


def paragraph_xml(
    text: str,
    style_id: str,
    *,
    bold: bool = False,
    align: str | None = None,
    preserve_space: bool = False,
) -> str:
    text = normalize_inline(text)
    if not text:
        return f'<w:p xmlns:w="{W_NS}"/>'
    attrs = ' xml:space="preserve"' if preserve_space else ""
    rpr = "<w:rPr><w:b/></w:rPr>" if bold else ""
    jc = f'<w:jc w:val="{align}"/>' if align else ""
    return (
        f'<w:p xmlns:w="{W_NS}">'
        f"<w:pPr><w:pStyle w:val=\"{style_id}\"/>{jc}</w:pPr>"
        f"<w:r>{rpr}<w:t{attrs}>{escape(text)}</w:t></w:r>"
        f"</w:p>"
    )


def table_xml(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    col_count = max(len(row) for row in rows)
    if col_count == 0:
        return ""
    cell_width = max(1200, int(9000 / col_count))
    grid = "".join(f'<w:gridCol w:w="{cell_width}"/>' for _ in range(col_count))
    body_parts: list[str] = [
        f'<w:tbl xmlns:w="{W_NS}">',
        "<w:tblPr>",
        '<w:tblW w:w="0" w:type="auto"/>',
        "<w:tblBorders>",
        '<w:top w:val="single" w:sz="8" w:space="0" w:color="auto"/>',
        '<w:left w:val="single" w:sz="8" w:space="0" w:color="auto"/>',
        '<w:bottom w:val="single" w:sz="8" w:space="0" w:color="auto"/>',
        '<w:right w:val="single" w:sz="8" w:space="0" w:color="auto"/>',
        '<w:insideH w:val="single" w:sz="8" w:space="0" w:color="auto"/>',
        '<w:insideV w:val="single" w:sz="8" w:space="0" w:color="auto"/>',
        "</w:tblBorders>",
        "</w:tblPr>",
        f"<w:tblGrid>{grid}</w:tblGrid>",
    ]
    for row_index, row in enumerate(rows):
        body_parts.append("<w:tr>")
        for cell in row:
            cell_text = normalize_inline(cell)
            bold = row_index == 0
            rpr = "<w:rPr><w:b/></w:rPr>" if bold else ""
            body_parts.append(
                "<w:tc>"
                "<w:tcPr>"
                f'<w:tcW w:w="{cell_width}" w:type="dxa"/>'
                "</w:tcPr>"
                "<w:p>"
                f'<w:pPr><w:pStyle w:val="{STYLE_TABLE_TEXT}"/></w:pPr>'
                f"<w:r>{rpr}<w:t>{escape(cell_text)}</w:t></w:r>"
                "</w:p>"
                "</w:tc>"
            )
        if len(row) < col_count:
            for _ in range(col_count - len(row)):
                body_parts.append(
                    "<w:tc>"
                    "<w:tcPr>"
                    f'<w:tcW w:w="{cell_width}" w:type="dxa"/>'
                    "</w:tcPr>"
                    "<w:p>"
                    f'<w:pPr><w:pStyle w:val="{STYLE_TABLE_TEXT}"/></w:pPr>'
                    "<w:r><w:t></w:t></w:r>"
                    "</w:p>"
                    "</w:tc>"
                )
        body_parts.append("</w:tr>")
    body_parts.append("</w:tbl>")
    return "".join(body_parts)


def parse_markdown_blocks(text: str) -> list[dict]:
    lines = text.splitlines()
    blocks: list[dict] = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("### "):
            blocks.append({"type": "heading", "level": 3, "text": stripped[4:]})
            i += 1
            continue
        if stripped.startswith("## "):
            blocks.append({"type": "heading", "level": 2, "text": stripped[3:]})
            i += 1
            continue
        if stripped.startswith("# "):
            blocks.append({"type": "heading", "level": 1, "text": stripped[2:]})
            i += 1
            continue
        if stripped.startswith("$$"):
            formula_lines: list[str] = []
            tail = ""
            i += 1
            while i < len(lines):
                current = lines[i].rstrip()
                if current.strip().startswith("$$"):
                    tail = current.strip()[2:].strip()
                    i += 1
                    break
                formula_lines.append(current.strip())
                i += 1
            formula = " ".join(formula_lines)
            if tail:
                formula = f"{formula} {tail}".strip()
            blocks.append({"type": "formula", "text": latex_to_text(formula)})
            continue
        if stripped.startswith("|"):
            table_lines: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            rows: list[list[str]] = []
            for idx, raw in enumerate(table_lines):
                if idx == 1 and re.fullmatch(r"\|[\-\s|:]+\|", raw):
                    continue
                cells = [cell.strip() for cell in raw.strip("|").split("|")]
                rows.append(cells)
            blocks.append({"type": "table", "rows": rows})
            continue
        if re.match(r"^\d+\.\s+", stripped) or re.match(r"^-+\s+", stripped):
            while i < len(lines):
                item = lines[i].strip()
                if not item:
                    i += 1
                    break
                if re.match(r"^\d+\.\s+", item) or re.match(r"^-+\s+", item):
                    blocks.append({"type": "paragraph", "text": item})
                    i += 1
                    continue
                break
            continue
        paragraph_lines = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].rstrip()
            nxt_stripped = nxt.strip()
            if not nxt_stripped:
                i += 1
                break
            if nxt_stripped.startswith(("# ", "## ", "### ", "$$", "|")):
                break
            paragraph_lines.append(nxt_stripped)
            i += 1
        blocks.append({"type": "paragraph", "text": " ".join(paragraph_lines)})
    return blocks


def blocks_to_xml(blocks: Iterable[dict]) -> list[str]:
    xml_parts: list[str] = []
    for block in blocks:
        block_type = block["type"]
        if block_type == "heading":
            text = block["text"]
            level = block["level"]
            if level == 1:
                style = STYLE_SPECIAL_HEADING if text.strip().lower() in SPECIAL_H1 else STYLE_CHAPTER
            else:
                style = STYLE_SECTION
            xml_parts.append(paragraph_xml(text, style))
        elif block_type == "paragraph":
            text = block["text"]
            if text.startswith("Таблица "):
                xml_parts.append(paragraph_xml(text, STYLE_TABLE_CAPTION))
            elif text.startswith("Рисунок "):
                xml_parts.append(paragraph_xml(text, STYLE_FIGURE_CAPTION, align="center"))
            else:
                xml_parts.append(paragraph_xml(text, STYLE_PARAGRAPH))
        elif block_type == "formula":
            xml_parts.append(paragraph_xml(block["text"], STYLE_PARAGRAPH, align="center", preserve_space=True))
        elif block_type == "table":
            xml_parts.append(table_xml(block["rows"]))
    return xml_parts


def build_partial_docx(template_path: Path, output_path: Path, markdown_paths: list[Path]) -> None:
    with zipfile.ZipFile(template_path, "r") as zf:
        files = {name: zf.read(name) for name in zf.namelist()}

    document_root = ET.fromstring(files["word/document.xml"])
    body = document_root.find("w:body", NS)
    if body is None:
        raise RuntimeError("word/document.xml does not contain w:body")

    sect_pr = body.find("w:sectPr", NS)
    if sect_pr is None:
        raise RuntimeError("template body does not contain w:sectPr")

    new_body_children: list[ET.Element] = []
    for md_path in markdown_paths:
        blocks = parse_markdown_blocks(read_text(md_path))
        for xml_chunk in blocks_to_xml(blocks):
            new_body_children.append(ET.fromstring(xml_chunk))

    for child in list(body):
        body.remove(child)
    for child in new_body_children:
        body.append(child)
    body.append(sect_pr)

    files["word/document.xml"] = ET.tostring(document_root, encoding="utf-8", xml_declaration=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    template_path = root / "template" / "blank-template.docx"
    output_path = root / "output" / "01-02-intro-review.docx"
    markdown_paths = [
        root / "text" / "01-intro.md",
        root / "text" / "02-review.md",
    ]
    build_partial_docx(template_path, output_path, markdown_paths)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
