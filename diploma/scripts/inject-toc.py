"""Перенос обновлённого TOC и _Toc-закладок из tmp.docx в pandoc-output.

Контекст: Word через COM-автоматизацию (см. update-fields.ps1) умеет
обновлять TablesOfContents — после $toc.Update() блок TOC заполнен
гиперссылками на закладки заголовков (_TocNNNNNN) и реальными номерами
страниц. Однако SaveAs2 в Word'е перезаписывает word/document.xml целиком
и ремаппит styleId всех остальных стилей (Compact -> aff0 и т. п.), теряя
шаблон Маршуниной.

Решение — гибрид с двойным переносом:
1. На временной копии docx вызывается Word, который добавляет к каждому
   pandoc'овскому bookmark в заголовке (X..., введение, реферат и т. п.)
   парную _TocNNN-закладку и заполняет sdt-блок TOC гиперссылками на эти
   _Toc-закладки и номерами страниц.
2. В исходном pandoc-output (стили шаблона целы) для каждого совпавшего
   pandoc-bookmark в одном с ним <w:p> ставится точечная _Toc-закладка
   <w:bookmarkStart w:id="N" w:name="_TocNNN"/><w:bookmarkEnd w:id="N"/>,
   с уникальным id (max+1, max+2, ...).
3. Пустой sdt-блок TOC основного файла заменяется на заполненный sdt-блок
   из tmp (он использует только w- и w14-неймспейсы).
4. К <w:document> основного файла добавляется недостающий
   xmlns:w14, чтобы заполненный sdt не дал "unbound prefix".

Использование: python inject-toc.py <src.docx> <dst.docx>
  src — временная копия после update-fields.ps1 (с заполненным TOC).
  dst — исходный pandoc-output, в который инъектируется TOC и _Toc-закладки.
Изменяется dst.
"""
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

SDT_OPEN_RE = re.compile(r"<w:sdt>")
TOC_INSTR_RE = re.compile(r"<w:instrText[^>]*>\s*TOC\b")
PARA_RE = re.compile(r"<w:p\b[^>]*>.*?</w:p>", re.DOTALL)
BM_NAME_RE = re.compile(r'<w:bookmarkStart[^>]*?w:name="([^"]+)"[^/]*?/>')
TOC_NAME_RE = re.compile(r'<w:bookmarkStart[^>]*?w:name="(_Toc\d+)"[^/]*?/>')
BM_ID_RE = re.compile(r'<w:bookmarkStart[^>]*?w:id="(\d+)"')
DOCUMENT_TAG_RE = re.compile(r"<w:document\b[^>]*>")

# Word при сохранении ремаппит standard-TOC-стили в собственные числовые
# styleId. Возвращаем их в styleId шаблона Маршуниной по семантическому
# имени стиля (см. word/styles.xml шаблона blank-template.docx).
TOC_STYLE_REMAP = {
    "aff0": "988",   # TOC Heading      -> +ЗаголРеферСодерж
    "13":   "958",   # toc 1            -> toc 1 (template id)
    "25":   "960",   # toc 2            -> toc 2 (template id)
    "33":   "962",   # toc 3            -> toc 3 (template id)
}


def find_toc_sdt_span(xml: str) -> tuple[int, int]:
    """Найти границы <w:sdt>...</w:sdt>, содержащего field code TOC."""
    for m in SDT_OPEN_RE.finditer(xml):
        depth = 1
        pos = m.end()
        while depth > 0:
            next_open = xml.find("<w:sdt>", pos)
            next_close = xml.find("</w:sdt>", pos)
            if next_close == -1:
                return -1, -1
            if next_open != -1 and next_open < next_close:
                depth += 1
                pos = next_open + len("<w:sdt>")
            else:
                depth -= 1
                pos = next_close + len("</w:sdt>")
        if TOC_INSTR_RE.search(xml[m.start():pos]):
            return m.start(), pos
    return -1, -1


def build_bm_to_toc_map(tmp_xml: str) -> dict:
    """Построить mapping pandoc-bookmark-name -> _TocNNN-bookmark-name.

    Word добавляет _Toc-закладки в тот же <w:p>, что и оригинальные
    pandoc'овские. Используем абзацную близость как critirium.
    """
    mapping: dict[str, str] = {}
    for pmatch in PARA_RE.finditer(tmp_xml):
        para = pmatch.group(0)
        toc_bms = TOC_NAME_RE.findall(para)
        if not toc_bms:
            continue
        toc_name = toc_bms[0]
        all_bms = BM_NAME_RE.findall(para)
        non_toc = [n for n in all_bms if not n.startswith("_Toc")]
        for n in non_toc:
            mapping[n] = toc_name
    return mapping


def inject_toc_bookmarks(dst_xml: str, mapping: dict) -> tuple[str, int]:
    """Рядом с каждым pandoc-bookmark из mapping добавить точечный _Toc-bookmark.

    Pandoc кладёт <w:bookmarkStart .../> между <w:p>...</w:p> заголовками
    (за пределами абзаца), Word — внутри. Поэтому ищем bookmark'и в xml
    напрямую, без paragraph-контекста, и вставляем парный _Toc bookmark
    сразу после исходного bookmarkStart. Гиперссылка <w:hyperlink
    w:anchor="_TocNNN"> в TOC при клике перейдёт к этой позиции — то есть
    непосредственно к началу заголовка.

    Возвращает (новый xml, число вставленных bookmark'ов).
    """
    existing_ids = [int(x) for x in BM_ID_RE.findall(dst_xml)]
    next_id = (max(existing_ids) + 1) if existing_ids else 1

    used_toc_names: set[str] = set()
    inserts: list[tuple[int, str]] = []
    for bm in re.finditer(
            r'<w:bookmarkStart[^>]*?w:name="([^"]+)"[^/]*?/>', dst_xml):
        name = bm.group(1)
        if name.startswith("_Toc"):
            continue
        toc_name = mapping.get(name)
        if not toc_name or toc_name in used_toc_names:
            continue
        used_toc_names.add(toc_name)
        snippet = (
            f'<w:bookmarkStart w:id="{next_id}" w:name="{toc_name}"/>'
            f'<w:bookmarkEnd w:id="{next_id}"/>'
        )
        inserts.append((bm.end(), snippet))
        next_id += 1

    inserts.sort(key=lambda t: t[0], reverse=True)
    out = dst_xml
    for pos, snippet in inserts:
        out = out[:pos] + snippet + out[pos:]
    return out, len(inserts)


def ensure_namespace(xml: str, prefix: str, uri: str) -> str:
    """Добавить xmlns:prefix к открывающему <w:document>, если его нет."""
    m = DOCUMENT_TAG_RE.search(xml)
    if not m:
        return xml
    tag = m.group(0)
    needle = f'xmlns:{prefix}="'
    if needle in tag:
        return xml
    new_tag = tag.replace(
        "<w:document",
        f'<w:document xmlns:{prefix}="{uri}"',
        1,
    )
    return xml[:m.start()] + new_tag + xml[m.end():]


def inject(src_path: Path, dst_path: Path) -> int:
    if not src_path.exists():
        print(f"[inject-toc] нет источника: {src_path}", file=sys.stderr)
        return 1
    if not dst_path.exists():
        print(f"[inject-toc] нет целевого файла: {dst_path}", file=sys.stderr)
        return 1

    with zipfile.ZipFile(src_path, "r") as z:
        src_xml = z.read("word/document.xml").decode("utf-8")
    with zipfile.ZipFile(dst_path, "r") as z:
        names = z.namelist()
        dst_xml = z.read("word/document.xml").decode("utf-8")
        files = {n: z.read(n) for n in names}

    src_s, src_e = find_toc_sdt_span(src_xml)
    if src_s == -1:
        print("[inject-toc] в источнике не найден TOC sdt-блок",
              file=sys.stderr)
        return 1
    dst_s, dst_e = find_toc_sdt_span(dst_xml)
    if dst_s == -1:
        print("[inject-toc] в целевом файле не найден TOC sdt-блок",
              file=sys.stderr)
        return 1

    mapping = build_bm_to_toc_map(src_xml)
    if not mapping:
        print("[inject-toc] mapping pandoc-bookmark -> _Toc пуст",
              file=sys.stderr)
        return 1

    new_dst, inserted = inject_toc_bookmarks(dst_xml, mapping)
    if inserted == 0:
        print("[inject-toc] ни одного _Toc-bookmark'а не вставлено",
              file=sys.stderr)
        return 1

    # Re-find sdt span на изменённом xml.
    new_s, new_e = find_toc_sdt_span(new_dst)
    toc_block = src_xml[src_s:src_e]
    # Ремап ремапнутых Word'ом styleId внутри sdt-блока обратно в id'ы
    # шаблона Маршуниной — иначе TOC-paragraph'ы потеряют форматирование
    # (отсутствующий styleId в styles.xml = default-стиль).
    for src_sid, dst_sid in TOC_STYLE_REMAP.items():
        toc_block = re.sub(
            r'(<w:pStyle\s+w:val=")' + re.escape(src_sid) + r'("\s*/>)',
            r'\g<1>' + dst_sid + r'\g<2>',
            toc_block,
        )
    new_dst = new_dst[:new_s] + toc_block + new_dst[new_e:]

    # Гарантируем наличие w14 namespace, который используют sdt-атрибуты
    # типа w14:paraId/textId внутри перенесённого блока.
    new_dst = ensure_namespace(
        new_dst, "w14",
        "http://schemas.microsoft.com/office/word/2010/wordml",
    )

    files["word/document.xml"] = new_dst.encode("utf-8")

    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".docx", dir=str(dst_path.parent))
    os.close(tmp_fd)
    Path(tmp_name).unlink()
    try:
        with zipfile.ZipFile(tmp_name, "w", zipfile.ZIP_DEFLATED) as zout:
            for n in names:
                zout.writestr(n, files[n])
        shutil.move(tmp_name, str(dst_path))
    except Exception:
        if Path(tmp_name).exists():
            Path(tmp_name).unlink()
        raise

    print(
        f"[inject-toc] mapping={len(mapping)}, _Toc-bookmark'ов вставлено: "
        f"{inserted}, sdt-блок {src_e - src_s} -> {dst_e - dst_s} байт."
    )
    return 0


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python inject-toc.py <src.docx> <dst.docx>",
              file=sys.stderr)
        return 2
    return inject(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    raise SystemExit(main())
