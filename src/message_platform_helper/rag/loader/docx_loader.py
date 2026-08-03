"""Word document loader."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..models import Document, DocumentMetadata, file_document_identity, normalize_tags, stable_document_id
from .base import DocumentLoader


WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
DRAWING_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


class DocxLoader(DocumentLoader):
    format = "docx"
    extensions = (".docx",)

    def load(self, source: str | Path, *, title: str | None = None, tags: list[str] | None = None) -> Document:
        path = Path(source)
        content = read_docx_text(path)
        if not content.strip():
            raise ValueError(f"DOCX text extraction returned empty content: {path}")
        document_title = title or path.stem.replace("_", " ")
        metadata = DocumentMetadata(format="docx", source=str(path), tags=normalize_tags(tags), extra=file_document_identity(str(path.resolve()), content))
        LOGGER.info("Loaded DOCX file=%s text_length=%s", path.name, len(content))
        return Document(
            id=stable_document_id(str(path), document_title, content),
            title=document_title,
            content=content,
            metadata=metadata,
        )


LOGGER = logging.getLogger(__name__)


def read_docx_text(path: Path) -> str:
    try:
        from docx import Document as DocxDocument  # type: ignore[import-untyped]
        from docx.table import Table  # type: ignore[import-untyped]
        from docx.text.paragraph import Paragraph  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError("Word ingestion requires python-docx. Install the optional dependency 'python-docx'.") from exc
    document = DocxDocument(path)
    parts: list[str] = []
    image_index = 0
    for block in _iter_body_blocks(document):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if text:
                parts.append(_paragraph_text(block, text))
            for image_marker in _paragraph_images(block, start_index=image_index + 1):
                image_index += 1
                parts.append(image_marker)
        elif isinstance(block, Table):
            table_text = _table_text(block)
            if table_text:
                parts.append(table_text)
    return "\n\n".join(part for part in parts if part.strip())


def _iter_body_blocks(document: Any) -> list[Any]:
    from docx.table import Table  # type: ignore[import-untyped]
    from docx.text.paragraph import Paragraph  # type: ignore[import-untyped]

    blocks: list[Any] = []
    for child in document.element.body.iterchildren():
        if child.tag == f"{{{WORD_NS}}}p":
            blocks.append(Paragraph(child, document))
        elif child.tag == f"{{{WORD_NS}}}tbl":
            blocks.append(Table(child, document))
    return blocks


def _paragraph_text(paragraph: Any, text: str) -> str:
    style_name = paragraph.style.name if paragraph.style is not None else ""
    if style_name.lower().startswith("heading"):
        return f"{'#' * _heading_level(style_name)} {text}"
    return text


def _paragraph_images(paragraph: Any, *, start_index: int) -> list[str]:
    images: list[str] = []
    for offset, drawing in enumerate(paragraph._element.findall(f".//{{{DRAWING_NS}}}blip"), start=start_index):
        rel_id = drawing.attrib.get(f"{{{REL_NS}}}embed") or drawing.attrib.get(f"{{{REL_NS}}}link") or ""
        images.append(f"[Image {offset}: relationship={rel_id or 'unknown'}]")
    return images


def _table_text(table: Any) -> str:
    rows = [" | ".join(_dedupe_cell_text(cell.text) for cell in row.cells) for row in table.rows]
    return "\n".join(row for row in rows if row.strip())


def _heading_level(style_name: str) -> int:
    digits = "".join(char for char in style_name if char.isdigit())
    if not digits:
        return 1
    return max(1, min(int(digits), 6))


def _dedupe_cell_text(text: str) -> str:
    lines: list[str] = []
    for line in str(text or "").splitlines():
        normalized = line.strip()
        if normalized and normalized not in lines:
            lines.append(normalized)
    return " ".join(lines)
