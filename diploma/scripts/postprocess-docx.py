"""Post-обработка собранного pandoc'ом docx.

Шаги:
1. Pandoc для каждого параграфа внутри ячеек таблицы навязывает встроенный
   стиль "Compact", игнорируя custom-style, прокинутый через Lua-фильтр на
   уровне Div. Этот пост-шаг подменяет ссылки <w:pStyle w:val="Compact"/>
   внутри document.xml на styleId "989" — стиль "+Текст в таблице" из
   template/blank-template.docx.
2. Для каждой таблицы выставляем <w:tblHeader/> на первой строке, чтобы
   Word повторял её на каждой следующей странице при переносе таблицы. По
   СТО СГАУ многостраничные таблицы должны иметь повторяющиеся заголовки
   столбцов на новой странице.
3. Для каждого <w:drawing> ограничиваем ширину 14 см и центрируем
   родительский параграф (СТО СГАУ требует центрирования рисунков и
   ширины не более ширины текстового блока).
4. Сбрасываем флаг w:dirty на field codes (TOC, INCLUDEPICTURE и т.п.),
   чтобы Word при открытии не показывал диалог "Документ содержит поля,
   которые ссылаются на другие файлы. Обновить поля в документе?".

Использование: python scripts/postprocess-docx.py <docx>
"""
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path


CELL_PARAGRAPH_STYLE_ID = "989"

# Pandoc для заголовка раздела содержания применяет встроенный стиль с
# именем "TOC Heading" (в шаблоне Маршуниной — styleId 957). По СТО СГАУ
# этот заголовок должен идти в стиле "+ЗаголРеферСодерж" (styleId 988):
# прописными по центру, как "ВВЕДЕНИЕ" / "ЗАКЛЮЧЕНИЕ" / "СПИСОК...".
TOC_HEADING_SOURCE_IDS = ("957", "TOCHeading")
TOC_HEADING_TARGET_ID = "988"

# 1 см = 360000 EMU; 14 см = 5040000 EMU. По СТО СГАУ ширина текстового
# блока на A4 при полях 25/15/20/20 мм составляет ≈170 мм; ограничиваем
# рисунки 140 мм с запасом, чтобы подпись и рамка не упирались в поля.
MAX_IMAGE_WIDTH_EMU = 5_040_000

TBL_OPEN_RE = re.compile(r"<w:tbl>")
TR_OPEN_RE = re.compile(r"<w:tr\b[^>]*>")
TRPR_SELF_RE = re.compile(r"<w:trPr\s*/>")
TRPR_OPEN_RE = re.compile(r"<w:trPr>")

DRAWING_BLOCK_RE = re.compile(r"<w:drawing\b[^>]*>.*?</w:drawing>", re.DOTALL)
EXTENT_RE = re.compile(r'<(wp:extent|a:ext)\s+cx="(\d+)"\s+cy="(\d+)"\s*/>')
P_OPEN_RE = re.compile(r"<w:p\b[^>]*>")
PPR_OPEN_RE = re.compile(r"<w:pPr\b[^/>]*>")
PPR_SELF_RE = re.compile(r"<w:pPr\s*/>")
JC_TAG_RE = re.compile(r"<w:jc\b[^/>]*/>")

DIRTY_TRUE_RE = re.compile(r'w:dirty="(?:true|1)"')


def ensure_tblheader_first_rows(doc_xml: str) -> tuple[str, int]:
    """Для каждого <w:tbl> добавить <w:tblHeader/> в <w:trPr> первой <w:tr>.

    Если у первой строки нет <w:trPr> — вставляем его. Если есть в форме
    самозакрывающегося тега — раскрываем. Если уже содержит <w:tblHeader> —
    оставляем как есть. Не пытаемся обрабатывать только многостраничные
    таблицы: установка флага на одностраничной таблице безопасна, Word её
    просто не дублирует визуально.

    Возвращает (новый XML, число модифицированных строк).
    """
    out: list[str] = []
    pos = 0
    modified = 0

    for m in TBL_OPEN_RE.finditer(doc_xml):
        if m.start() < pos:
            # Этот <w:tbl> уже внутри ранее обработанного диапазона
            # (теоретически невозможно: finditer возвращает не пересекающиеся
            # совпадения, но защищаемся на случай редактирования pos выше).
            continue
        m_tr = TR_OPEN_RE.search(doc_xml, m.end())
        if not m_tr:
            continue
        out.append(doc_xml[pos:m_tr.end()])
        pos = m_tr.end()
        rest = doc_xml[pos:]

        m_self = TRPR_SELF_RE.match(rest)
        m_open = TRPR_OPEN_RE.match(rest)
        if m_self:
            out.append("<w:trPr><w:tblHeader/></w:trPr>")
            pos += m_self.end()
            modified += 1
        elif m_open:
            close_pos = rest.find("</w:trPr>", m_open.end())
            if close_pos == -1:
                # Нелепый невалидный XML — пропускаем, выходим из обработки
                continue
            inner = rest[m_open.end():close_pos]
            if "<w:tblHeader" in inner:
                out.append(rest[: close_pos + len("</w:trPr>")])
                pos += close_pos + len("</w:trPr>")
            else:
                out.append("<w:trPr>" + inner + "<w:tblHeader/></w:trPr>")
                pos += close_pos + len("</w:trPr>")
                modified += 1
        else:
            out.append("<w:trPr><w:tblHeader/></w:trPr>")
            modified += 1

    out.append(doc_xml[pos:])
    return "".join(out), modified


def constrain_image_widths(doc_xml: str, max_emu: int = MAX_IMAGE_WIDTH_EMU) -> tuple[str, int]:
    """Для каждого <w:drawing> при ширине больше max_emu масштабировать пропорционально.

    Внутри одного <w:drawing> блока есть две группы экстентов: <wp:extent>
    (внешний контейнер inline/anchor) и <a:ext> (графический фрейм
    pic:spPr/a:xfrm). Они должны быть согласованы; берём максимум cx по
    блоку, считаем коэффициент scale = max_emu / max_cx, применяем ко всем
    парам cx/cy блока.

    Возвращает (новый XML, число масштабированных рисунков).
    """
    scaled = 0

    def process(m: "re.Match[str]") -> str:
        nonlocal scaled
        block = m.group(0)
        extents = EXTENT_RE.findall(block)
        if not extents:
            return block
        max_cx = max(int(cx) for _, cx, _ in extents)
        if max_cx <= max_emu:
            return block
        scale = max_emu / max_cx

        def rescale(em: "re.Match[str]") -> str:
            tag = em.group(1)
            new_cx = int(int(em.group(2)) * scale)
            new_cy = int(int(em.group(3)) * scale)
            return f'<{tag} cx="{new_cx}" cy="{new_cy}"/>'

        scaled += 1
        return EXTENT_RE.sub(rescale, block)

    new_xml = DRAWING_BLOCK_RE.sub(process, doc_xml)
    return new_xml, scaled


def center_image_paragraphs(doc_xml: str) -> tuple[str, int]:
    """Для каждого <w:p>, содержащего <w:drawing>, гарантировать <w:jc w:val="center"/>.

    Если у параграфа нет <w:pPr> — добавляем <w:pPr><w:jc w:val="center"/></w:pPr>
    сразу после открывающего <w:p>. Если есть самозакрывающийся <w:pPr/> —
    раскрываем. Если есть полный <w:pPr>...</w:pPr> — добавляем или заменяем
    <w:jc> внутри.

    Возвращает (новый XML, число центрированных параграфов).
    """
    out: list[str] = []
    pos = 0
    centered = 0

    while True:
        m = P_OPEN_RE.search(doc_xml, pos)
        if not m:
            break
        p_open_end = m.end()
        p_close = doc_xml.find("</w:p>", p_open_end)
        if p_close == -1:
            break
        block = doc_xml[p_open_end:p_close]
        if "<w:drawing" not in block:
            out.append(doc_xml[pos:p_close + len("</w:p>")])
            pos = p_close + len("</w:p>")
            continue

        out.append(doc_xml[pos:p_open_end])

        m_self = PPR_SELF_RE.match(block)
        m_open = PPR_OPEN_RE.match(block)
        if m_self:
            new_block = '<w:pPr><w:jc w:val="center"/></w:pPr>' + block[m_self.end():]
        elif m_open:
            ppr_close = block.find("</w:pPr>", m_open.end())
            if ppr_close == -1:
                # Невалидный XML — не трогаем
                out.append(block + "</w:p>")
                pos = p_close + len("</w:p>")
                continue
            inner = block[m_open.end():ppr_close]
            rest = block[ppr_close + len("</w:pPr>"):]
            if JC_TAG_RE.search(inner):
                new_inner = JC_TAG_RE.sub('<w:jc w:val="center"/>', inner)
            else:
                new_inner = inner + '<w:jc w:val="center"/>'
            new_block = m_open.group(0) + new_inner + "</w:pPr>" + rest
        else:
            new_block = '<w:pPr><w:jc w:val="center"/></w:pPr>' + block

        out.append(new_block)
        out.append("</w:p>")
        pos = p_close + len("</w:p>")
        centered += 1

    out.append(doc_xml[pos:])
    return "".join(out), centered


def clear_dirty_fields(doc_xml: str) -> tuple[str, int]:
    """Заменить w:dirty="true" на w:dirty="false" во всех field codes.

    Word при открытии docx с полями, у которых w:dirty="true", показывает
    диалог "Документ содержит поля, которые ссылаются на другие файлы.
    Обновить поля в документе?". Pandoc выставляет dirty="true" на TOC,
    чтобы Word на первом открытии заполнил оглавление. Однако сборка
    выполняет принудительный пересчёт TOC через LibreOffice (или Word
    через COM), поэтому флаг можно сбрасывать без потери содержимого.

    Возвращает (новый XML, число сброшенных флагов).
    """
    cleared = len(DIRTY_TRUE_RE.findall(doc_xml))
    if cleared == 0:
        return doc_xml, 0
    new_xml = DIRTY_TRUE_RE.sub('w:dirty="false"', doc_xml)
    return new_xml, cleared


def postprocess(docx_path: Path) -> int:
    if not docx_path.exists():
        print(f"[postprocess] нет файла: {docx_path}", file=sys.stderr)
        return 1

    with zipfile.ZipFile(docx_path, "r") as z:
        names = z.namelist()
        with z.open("word/document.xml") as f:
            doc_xml = f.read().decode("utf-8")
        files = {n: z.read(n) for n in names}

    new_doc_xml = doc_xml
    for token in (
        '<w:pStyle w:val="Compact" />',
        '<w:pStyle w:val="Compact"/>',
    ):
        new_doc_xml = new_doc_xml.replace(
            token, f'<w:pStyle w:val="{CELL_PARAGRAPH_STYLE_ID}" />'
        )
    compact_replaced = doc_xml.count("Compact") - new_doc_xml.count("Compact")

    toc_heading_replaced = 0
    for src_id in TOC_HEADING_SOURCE_IDS:
        for token in (
            f'<w:pStyle w:val="{src_id}" />',
            f'<w:pStyle w:val="{src_id}"/>',
        ):
            count_before = new_doc_xml.count(token)
            if count_before:
                new_doc_xml = new_doc_xml.replace(
                    token, f'<w:pStyle w:val="{TOC_HEADING_TARGET_ID}" />'
                )
                toc_heading_replaced += count_before

    new_doc_xml, header_added = ensure_tblheader_first_rows(new_doc_xml)
    new_doc_xml, images_scaled = constrain_image_widths(new_doc_xml)
    new_doc_xml, images_centered = center_image_paragraphs(new_doc_xml)
    new_doc_xml, dirty_cleared = clear_dirty_fields(new_doc_xml)

    if (
        compact_replaced == 0
        and toc_heading_replaced == 0
        and header_added == 0
        and images_scaled == 0
        and images_centered == 0
        and dirty_cleared == 0
    ):
        print(
            "[postprocess] нет ни Compact-стилей, ни TOC-заголовков, "
            "ни таблиц без tblHeader, ни рисунков для центрирования/масштабирования, "
            "ни dirty-полей."
        )
        return 0

    files["word/document.xml"] = new_doc_xml.encode("utf-8")

    import os
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".docx", dir=str(docx_path.parent))
    os.close(tmp_fd)
    Path(tmp_name).unlink()
    try:
        with zipfile.ZipFile(tmp_name, "w", zipfile.ZIP_DEFLATED) as zout:
            for n in names:
                zout.writestr(n, files[n])
        shutil.move(tmp_name, str(docx_path))
    except Exception:
        if Path(tmp_name).exists():
            Path(tmp_name).unlink()
        raise

    print(
        f"[postprocess] Compact->{CELL_PARAGRAPH_STYLE_ID}: {compact_replaced}, "
        f"TOC Heading->{TOC_HEADING_TARGET_ID}: {toc_heading_replaced}, "
        f"tblHeader на первой строке: {header_added} таблиц(ы), "
        f"рисунков отмасштабировано до 14см: {images_scaled}, "
        f"параграфов с рисунком центрировано: {images_centered}, "
        f"dirty-полей сброшено: {dirty_cleared}."
    )
    return 0


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python scripts/postprocess-docx.py <docx>", file=sys.stderr)
        return 2
    return postprocess(Path(sys.argv[1]))


if __name__ == "__main__":
    raise SystemExit(main())
