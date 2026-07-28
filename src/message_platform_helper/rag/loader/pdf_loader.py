"""PDF document loader."""

from __future__ import annotations

import logging
from pathlib import Path

from ..models import Document, DocumentMetadata, file_document_identity, normalize_tags, stable_document_id
from .base import DocumentLoader


class PdfLoader(DocumentLoader):
    format = "pdf"
    extensions = (".pdf",)

    def load(self, source: str | Path, *, title: str | None = None, tags: list[str] | None = None) -> Document:
        path = Path(source)
        content = read_pdf_text(path)
        if not content.strip():
            raise ValueError(f"PDF text extraction returned empty content: {path}")
        document_title = title or path.stem.replace("_", " ")
        metadata = DocumentMetadata(format="pdf", source=str(path), tags=normalize_tags(tags), extra=file_document_identity(str(path.resolve()), content))
        LOGGER.info("Loaded PDF file=%s text_length=%s", path.name, len(content))
        return Document(
            id=stable_document_id(str(path), document_title, content),
            title=document_title,
            content=content,
            metadata=metadata,
        )


LOGGER = logging.getLogger(__name__)


def read_pdf_text(path: Path) -> str:
    try:
        import fitz  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError("PDF ingestion requires PyMuPDF. Install the optional dependency 'pymupdf'.") from exc
    pages: list[str] = []
    with fitz.open(path) as document:
        for index, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append(f"[Page {index}]\n{text}")
    return "\n\n".join(pages)
