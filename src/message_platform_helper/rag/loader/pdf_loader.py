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
            page_parts = [f"[Page {index}]"]
            text = page.get_text("text").strip()
            if text:
                page_parts.append(text)
            page_parts.extend(_extract_pdf_tables(page, index))
            page_parts.extend(_extract_pdf_images(page, index))
            if len(page_parts) > 1:
                pages.append("\n\n".join(page_parts))
    return "\n\n".join(pages)


def _extract_pdf_tables(page: object, page_number: int) -> list[str]:
    find_tables = getattr(page, "find_tables", None)
    if not callable(find_tables):
        return []
    try:
        tables = find_tables()
    except Exception as exc:  # pragma: no cover - depends on PDF internals
        LOGGER.debug("PDF table extraction failed page=%s error=%s", page_number, exc)
        return []

    blocks: list[str] = []
    for table_index, table in enumerate(getattr(tables, "tables", []) or [], start=1):
        try:
            rows = table.extract()
        except Exception as exc:  # pragma: no cover - depends on PDF internals
            LOGGER.debug("PDF table extract failed page=%s table=%s error=%s", page_number, table_index, exc)
            continue
        normalized_rows = [
            " | ".join(str(cell or "").strip().replace("\n", " ") for cell in row).strip()
            for row in rows
            if any(str(cell or "").strip() for cell in row)
        ]
        if normalized_rows:
            blocks.append("\n".join([f"[Table {page_number}.{table_index}]", *normalized_rows]))
    return blocks


def _extract_pdf_images(page: object, page_number: int) -> list[str]:
    try:
        image_infos = page.get_image_info()  # type: ignore[attr-defined]
    except Exception:
        try:
            image_infos = [{"bbox": None} for _ in page.get_images(full=True)]  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - depends on PDF internals
            LOGGER.debug("PDF image extraction failed page=%s error=%s", page_number, exc)
            return []

    markers: list[str] = []
    for image_index, image in enumerate(image_infos, start=1):
        bbox = image.get("bbox") if isinstance(image, dict) else None
        width = image.get("width") if isinstance(image, dict) else None
        height = image.get("height") if isinstance(image, dict) else None
        details = [f"page={page_number}"]
        if width and height:
            details.append(f"size={width}x{height}")
        if bbox:
            details.append(f"bbox={_format_bbox(bbox)}")
        markers.append(f"[Image {page_number}.{image_index}: {', '.join(details)}]")
    return markers


def _format_bbox(bbox: object) -> str:
    try:
        values = [round(float(value), 1) for value in bbox]  # type: ignore[union-attr]
    except (TypeError, ValueError):
        return "unknown"
    return ",".join(str(value) for value in values)
