"""Word document loader."""

from __future__ import annotations

import logging
from pathlib import Path

from ..models import Document, DocumentMetadata, file_document_identity, normalize_tags, stable_document_id
from .base import DocumentLoader


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
    except ImportError as exc:
        raise RuntimeError("Word ingestion requires python-docx. Install the optional dependency 'python-docx'.") from exc
    document = DocxDocument(path)
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    table_blocks: list[str] = []
    for table in document.tables:
        rows = [" | ".join(_dedupe_cell_text(cell.text) for cell in row.cells) for row in table.rows]
        table_blocks.append("\n".join(row for row in rows if row.strip()))
    return "\n\n".join(part for part in [*paragraphs, *table_blocks] if part.strip())


def _dedupe_cell_text(text: str) -> str:
    lines: list[str] = []
    for line in str(text or "").splitlines():
        normalized = line.strip()
        if normalized and normalized not in lines:
            lines.append(normalized)
    return " ".join(lines)
