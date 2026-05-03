from __future__ import annotations

import mimetypes
import re
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "template" / "blank-template.docx"
OUTPUT_PATH = ROOT / "output" / "VKR_small_objects_auto_aug_full.docx"
CHAPTERS = [
    ROOT / "text" / "00-titlepage.md",
    ROOT / "text" / "00a-referat.md",
    ROOT / "text" / "00c-contents.md",
    ROOT / "text" / "01-intro.md",
    ROOT / "text" / "02-review.md",
    ROOT / "text" / "03-requirements.md",
    ROOT / "text" / "04-architecture.md",
    ROOT / "text" / "05-implementation.md",
    ROOT / "text" / "06-experimental-design.md",
    ROOT / "text" / "07-experiments.md",
    ROOT / "text" / "08-conclusion.md",
    ROOT / "text" / "09-references.md",
]

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
PIC_NS = "http://schemas.openxmlformats.org/drawingml/2006/picture"
MAX_IMAGE_WIDTH_EMU = 5_040_000

ET.register_namespace("w", W_NS)
ET.register_namespace("r", R_NS)

NS = {"w": W_NS, "r": R_NS}

STYLE_PARAGRAPH = "947"
STYLE_TITLE_THEME = "945"
STYLE_TITLE_CENTER = "946"
STYLE_SPECIAL_HEADING = "971"
STYLE_REFERAT_TOC_HEADING = "988"
STYLE_CHAPTER = "979"
STYLE_SECTION = "980"
STYLE_TABLE_CAPTION = "975"
STYLE_FIGURE_CAPTION = "973"
STYLE_TABLE_TEXT = "989"

SPECIAL_H1 = {
    "перечень условных обозначений и сокращений",
}

IMAGE_LINE_RE = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)$")
TABLE_CAPTION_RE = re.compile(r"^Таблица\s+(\d+)\s+-\s+")
FIGURE_CAPTION_RE = re.compile(r"^Рисунок\s+(\d+)\s+-\s+")


@dataclass
class ImageAsset:
    rel_id: str
    target: str
    width_emu: int
    height_emu: int
    alt_text: str


def w_tag(name: str) -> str:
    return f"{{{W_NS}}}{name}"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def strip_internal_sources(text: str) -> str:
    marker = "\n## Источники раздела"
    pos = text.find(marker)
    if pos == -1:
        return text
    return text[:pos].rstrip() + "\n"


def normalize_inline(text: str) -> str:
    text = text.strip()
    text = text.replace("`", "")
    text = text.replace("**", "")
    text = text.replace("__", "")
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


def page_break_xml() -> str:
    return (
        f'<w:p xmlns:w="{W_NS}">'
        "<w:r><w:br w:type=\"page\"/></w:r>"
        "</w:p>"
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
        if row_index == 0:
            body_parts.append("<w:trPr><w:tblHeader/></w:trPr>")
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
        body_parts.append("</w:tr>")
    body_parts.append("</w:tbl>")
    return "".join(body_parts)


def image_xml(asset: ImageAsset, doc_pr_id: int) -> str:
    return (
        f'<w:p xmlns:w="{W_NS}" xmlns:r="{R_NS}" xmlns:wp="{WP_NS}" xmlns:a="{A_NS}" xmlns:pic="{PIC_NS}">'
        '<w:pPr><w:jc w:val="center"/></w:pPr>'
        "<w:r>"
        "<w:drawing>"
        "<wp:inline distT=\"0\" distB=\"0\" distL=\"0\" distR=\"0\">"
        f'<wp:extent cx="{asset.width_emu}" cy="{asset.height_emu}"/>'
        f'<wp:docPr id="{doc_pr_id}" name="Picture {doc_pr_id}" descr="{escape(asset.alt_text)}"/>'
        "<wp:cNvGraphicFramePr>"
        '<a:graphicFrameLocks noChangeAspect="1"/>'
        "</wp:cNvGraphicFramePr>"
        "<a:graphic>"
        '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        "<pic:pic>"
        "<pic:nvPicPr>"
        f'<pic:cNvPr id="{doc_pr_id}" name="Picture {doc_pr_id}"/>'
        "<pic:cNvPicPr/>"
        "</pic:nvPicPr>"
        "<pic:blipFill>"
        f'<a:blip r:embed="{asset.rel_id}"/>'
        "<a:stretch><a:fillRect/></a:stretch>"
        "</pic:blipFill>"
        "<pic:spPr>"
        "<a:xfrm>"
        '<a:off x="0" y="0"/>'
        f'<a:ext cx="{asset.width_emu}" cy="{asset.height_emu}"/>'
        "</a:xfrm>"
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        "</pic:spPr>"
        "</pic:pic>"
        "</a:graphicData>"
        "</a:graphic>"
        "</wp:inline>"
        "</w:drawing>"
        "</w:r>"
        "</w:p>"
    )


def parse_markdown_blocks(text: str, base_dir: Path) -> list[dict]:
    lines = text.splitlines()
    blocks: list[dict] = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith(":::title-center "):
            blocks.append({"type": "styled_paragraph", "style": STYLE_TITLE_CENTER, "text": stripped[len(":::title-center "):], "align": "center"})
            i += 1
            continue
        if stripped.startswith(":::title-theme "):
            blocks.append({"type": "styled_paragraph", "style": STYLE_TITLE_THEME, "text": stripped[len(":::title-theme "):], "align": "center"})
            i += 1
            continue
        if stripped.startswith(":::title-left "):
            blocks.append({"type": "styled_paragraph", "style": STYLE_TITLE_CENTER, "text": stripped[len(":::title-left "):], "align": "left"})
            i += 1
            continue
        if stripped.startswith("```"):
            fence_lang = stripped[3:].strip().lower()
            i += 1
            fence_lines: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                fence_lines.append(lines[i].rstrip())
                i += 1
            if i < len(lines):
                i += 1
            blocks.append({"type": "fence", "lang": fence_lang, "text": "\n".join(fence_lines)})
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
        image_match = IMAGE_LINE_RE.match(stripped)
        if image_match:
            image_path = image_match.group(2).split()[0]
            blocks.append(
                {
                    "type": "image",
                    "alt": image_match.group(1).strip() or "Иллюстрация",
                    "path": (base_dir / image_path).resolve(),
                }
            )
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
            if nxt_stripped.startswith(("# ", "## ", "### ", "$$", "|", "```")):
                break
            if IMAGE_LINE_RE.match(nxt_stripped):
                break
            paragraph_lines.append(nxt_stripped)
            i += 1
        blocks.append({"type": "paragraph", "text": " ".join(paragraph_lines)})
    return blocks


def ensure_content_type(root: ET.Element, extension: str, content_type: str) -> None:
    for child in root:
        if child.tag == f"{{{CT_NS}}}Default" and child.attrib.get("Extension") == extension:
            return
    ET.SubElement(root, f"{{{CT_NS}}}Default", Extension=extension, ContentType=content_type)


class DocxBuilder:
    def __init__(self, template_path: Path) -> None:
        with zipfile.ZipFile(template_path, "r") as zf:
            self.files = {name: zf.read(name) for name in zf.namelist()}
        self.document_root = ET.fromstring(self.files["word/document.xml"])
        self.body = self.document_root.find("w:body", NS)
        if self.body is None:
            raise RuntimeError("word/document.xml does not contain w:body")
        self.sect_pr = self.body.find("w:sectPr", NS)
        if self.sect_pr is None:
            raise RuntimeError("template body does not contain w:sectPr")

        self.rels_root = ET.fromstring(self.files["word/_rels/document.xml.rels"])
        self.content_types_root = ET.fromstring(self.files["[Content_Types].xml"])
        self.new_body_children: list[ET.Element] = []

        rel_numbers = []
        for rel in self.rels_root:
            rel_id = rel.attrib.get("Id", "")
            if rel_id.startswith("rId") and rel_id[3:].isdigit():
                rel_numbers.append(int(rel_id[3:]))
        self.next_rel_id = max(rel_numbers, default=0) + 1

        media_numbers = []
        for name in self.files:
            match = re.fullmatch(r"word/media/image(\d+)\.(\w+)", name)
            if match:
                media_numbers.append(int(match.group(1)))
        self.next_image_index = max(media_numbers, default=0) + 1
        self.next_doc_pr_id = 1

    def register_image(self, image_path: Path, alt_text: str) -> ImageAsset:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        suffix = image_path.suffix.lower().lstrip(".") or "png"
        content_type = mimetypes.types_map.get(f".{suffix}")
        if content_type is None:
            raise RuntimeError(f"Unsupported image type: {image_path.suffix}")

        target = f"media/image{self.next_image_index}.{suffix}"
        rel_id = f"rId{self.next_rel_id}"
        self.next_image_index += 1
        self.next_rel_id += 1

        image_bytes = image_path.read_bytes()
        with Image.open(BytesIO(image_bytes)) as img:
            width_px, height_px = img.size

        width_emu = int(width_px * 9525)
        height_emu = int(height_px * 9525)
        if width_emu > MAX_IMAGE_WIDTH_EMU:
            scale = MAX_IMAGE_WIDTH_EMU / width_emu
            width_emu = int(width_emu * scale)
            height_emu = int(height_emu * scale)

        self.files[f"word/{target}"] = image_bytes
        ET.SubElement(
            self.rels_root,
            f"{{{REL_NS}}}Relationship",
            Id=rel_id,
            Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
            Target=target,
        )
        ensure_content_type(self.content_types_root, suffix, content_type)

        return ImageAsset(rel_id, target, width_emu, height_emu, alt_text)

    def add_xml_chunk(self, xml_chunk: str) -> None:
        self.new_body_children.append(ET.fromstring(xml_chunk))

    def add_markdown(self, markdown_path: Path) -> None:
        text = strip_internal_sources(read_text(markdown_path))
        blocks = parse_markdown_blocks(text, markdown_path.parent)
        for block in blocks:
            block_type = block["type"]
            if block_type == "fence":
                continue
            if block_type == "heading":
                text = block["text"]
                level = block["level"]
                if level == 1:
                    lowered = text.strip().lower()
                    if lowered in {"реферат", "содержание"}:
                        style = STYLE_REFERAT_TOC_HEADING
                    elif lowered in {"введение", "заключение", "список использованных источников"} or lowered in SPECIAL_H1:
                        style = STYLE_SPECIAL_HEADING
                    else:
                        style = STYLE_CHAPTER
                else:
                    style = STYLE_SECTION
                self.add_xml_chunk(paragraph_xml(text, style))
                continue
            if block_type == "styled_paragraph":
                self.add_xml_chunk(
                    paragraph_xml(
                        block["text"],
                        block["style"],
                        align=block.get("align"),
                    )
                )
                continue
            if block_type == "paragraph":
                text = block["text"]
                if TABLE_CAPTION_RE.match(text):
                    self.add_xml_chunk(paragraph_xml(text, STYLE_TABLE_CAPTION))
                elif FIGURE_CAPTION_RE.match(text):
                    self.add_xml_chunk(paragraph_xml(text, STYLE_FIGURE_CAPTION, align="center"))
                else:
                    self.add_xml_chunk(paragraph_xml(text, STYLE_PARAGRAPH))
                continue
            if block_type == "formula":
                self.add_xml_chunk(paragraph_xml(block["text"], STYLE_PARAGRAPH, align="center", preserve_space=True))
                continue
            if block_type == "table":
                self.add_xml_chunk(table_xml(block["rows"]))
                continue
            if block_type == "image":
                asset = self.register_image(block["path"], block["alt"])
                self.add_xml_chunk(image_xml(asset, self.next_doc_pr_id))
                self.next_doc_pr_id += 1
                continue

    def save(self, output_path: Path) -> None:
        for child in list(self.body):
            self.body.remove(child)
        for child in self.new_body_children:
            self.body.append(child)
        self.body.append(self.sect_pr)

        self.files["word/document.xml"] = ET.tostring(self.document_root, encoding="utf-8", xml_declaration=True)
        self.files["word/_rels/document.xml.rels"] = ET.tostring(self.rels_root, encoding="utf-8", xml_declaration=True)
        self.files["[Content_Types].xml"] = ET.tostring(self.content_types_root, encoding="utf-8", xml_declaration=True)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_path = output_path
        if final_path.exists():
            try:
                final_path.unlink()
            except PermissionError:
                suffix = 1
                while True:
                    candidate = final_path.with_name(f"{final_path.stem}_{suffix}{final_path.suffix}")
                    if not candidate.exists():
                        final_path = candidate
                        break
                    suffix += 1
        with zipfile.ZipFile(final_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, content in self.files.items():
                zf.writestr(name, content)
        self.output_path = final_path


def validate_markdown(chapters: Iterable[Path]) -> dict[str, int]:
    chapter_list = list(chapters)
    table_numbers: list[int] = []
    figure_numbers: list[int] = []
    image_count = 0

    for chapter in chapter_list:
        text = read_text(chapter)
        if "\ufffd" in text:
            raise RuntimeError(f"Found replacement character in {chapter}")
        if "???" in text:
            raise RuntimeError(f"Found placeholder question marks in {chapter}")
        bad_controls = [
            ch for ch in text if ord(ch) < 32 and ch not in ("\n", "\r", "\t")
        ]
        if bad_controls:
            raise RuntimeError(f"Found control characters in {chapter}")

        for line in text.splitlines():
            table_match = TABLE_CAPTION_RE.match(line.strip())
            if table_match:
                table_numbers.append(int(table_match.group(1)))
            figure_match = FIGURE_CAPTION_RE.match(line.strip())
            if figure_match:
                figure_numbers.append(int(figure_match.group(1)))
            match = IMAGE_LINE_RE.match(line.strip())
            if not match:
                continue
            image_count += 1
            image_path = (chapter.parent / match.group(2).split()[0]).resolve()
            if not image_path.exists():
                raise FileNotFoundError(f"Broken image link in {chapter}: {image_path}")

    if table_numbers and table_numbers != list(range(1, len(table_numbers) + 1)):
        raise RuntimeError(f"Table numbering is not sequential: {table_numbers}")
    if figure_numbers and figure_numbers != list(range(1, len(figure_numbers) + 1)):
        raise RuntimeError(f"Figure numbering is not sequential: {figure_numbers}")

    return {
        "chapters": len(chapter_list),
        "tables": len(table_numbers),
        "figures": len(figure_numbers),
        "images": image_count,
    }


def main() -> int:
    summary = validate_markdown(CHAPTERS)
    builder = DocxBuilder(TEMPLATE_PATH)
    for index, chapter in enumerate(CHAPTERS):
        if index > 0:
            builder.add_xml_chunk(page_break_xml())
        builder.add_markdown(chapter)
    builder.save(OUTPUT_PATH)
    print(
        f"{builder.output_path}\n"
        f"chapters={summary['chapters']} tables={summary['tables']} "
        f"figures={summary['figures']} images={summary['images']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
