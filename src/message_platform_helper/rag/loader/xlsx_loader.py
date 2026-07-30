"""Excel workbook document loader."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

from ..models import Document, DocumentMetadata, file_document_identity, normalize_tags, stable_document_id
from .base import DocumentLoader


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


class XlsxLoader(DocumentLoader):
    format = "xlsx"
    extensions = (".xlsx",)

    def load(self, source: str | Path, *, title: str | None = None, tags: list[str] | None = None) -> Document:
        path = Path(source)
        content = read_xlsx_text(path)
        if not content.strip():
            raise ValueError(f"XLSX text extraction returned empty content: {path}")
        document_title = title or path.stem.replace("_", " ")
        metadata = DocumentMetadata(format="xlsx", source=str(path), tags=normalize_tags(tags), extra=file_document_identity(str(path.resolve()), content))
        return Document(
            id=stable_document_id(str(path), document_title, content),
            title=document_title,
            content=content,
            metadata=metadata,
        )


def read_xlsx_text(path: Path) -> str:
    with ZipFile(path) as workbook:
        shared_strings = _read_shared_strings(workbook)
        sheet_parts = _sheet_parts(workbook)
        sections: list[str] = []
        for sheet_name, sheet_path in sheet_parts:
            rows = _read_sheet_rows(workbook, sheet_path, shared_strings)
            if rows:
                sections.append("\n".join([f"# {sheet_name}", *rows]))
        return "\n\n".join(sections)


def _read_shared_strings(workbook: ZipFile) -> list[str]:
    try:
        root = ElementTree.fromstring(workbook.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return [_cell_text(item, []) for item in root.findall(f"{{{MAIN_NS}}}si")]


def _sheet_parts(workbook: ZipFile) -> list[tuple[str, str]]:
    root = ElementTree.fromstring(workbook.read("xl/workbook.xml"))
    rels = _workbook_relationships(workbook)
    result: list[tuple[str, str]] = []
    for sheet in root.findall(f".//{{{MAIN_NS}}}sheet"):
        name = sheet.attrib.get("name") or "Sheet"
        rel_id = sheet.attrib.get(f"{{{OFFICE_REL_NS}}}id")
        target = rels.get(rel_id or "")
        if target:
            result.append((name, _normalize_sheet_target(target)))
    return result


def _workbook_relationships(workbook: ZipFile) -> dict[str, str]:
    root = ElementTree.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
    return {rel.attrib.get("Id", ""): rel.attrib.get("Target", "") for rel in root.findall(f"{{{REL_NS}}}Relationship")}


def _normalize_sheet_target(target: str) -> str:
    if target.startswith("/"):
        return target.lstrip("/")
    if target.startswith("xl/"):
        return target
    return f"xl/{target}"


def _read_sheet_rows(workbook: ZipFile, sheet_path: str, shared_strings: list[str]) -> list[str]:
    root = ElementTree.fromstring(workbook.read(sheet_path))
    rows: list[str] = []
    for row in root.findall(f".//{{{MAIN_NS}}}row"):
        values = [_cell_text(cell, shared_strings) for cell in row.findall(f"{{{MAIN_NS}}}c")]
        line = " | ".join(value for value in values if value)
        if line:
            rows.append(line)
    return rows


def _cell_text(element: ElementTree.Element, shared_strings: list[str]) -> str:
    cell_type = element.attrib.get("t")
    if cell_type == "inlineStr":
        return " ".join(text.text.strip() for text in element.findall(f".//{{{MAIN_NS}}}t") if text.text and text.text.strip())
    value = element.find(f"{{{MAIN_NS}}}v")
    if value is None or value.text is None:
        return ""
    raw = value.text.strip()
    if cell_type == "s":
        try:
            return shared_strings[int(raw)]
        except (IndexError, ValueError):
            return raw
    return raw
