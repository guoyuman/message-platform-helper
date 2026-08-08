"""Word document loader."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..models import Document, DocumentMetadata, file_document_identity, normalize_tags, stable_document_id
from ..vision import rag_image_dir
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
    image_dir = rag_image_dir() / path.stem
    for block in _iter_body_blocks(document):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if text:
                parts.append(_paragraph_text(block, text))
            for image_marker in _paragraph_images(block, start_index=image_index + 1, image_dir=image_dir):
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


def _paragraph_images(paragraph: Any, *, start_index: int, image_dir: Path) -> list[str]:
    images: list[str] = []
    for offset, drawing in enumerate(paragraph._element.findall(f".//{{{DRAWING_NS}}}blip"), start=start_index):
        rel_id = drawing.attrib.get(f"{{{REL_NS}}}embed") or drawing.attrib.get(f"{{{REL_NS}}}link") or ""
        saved = _save_docx_image(paragraph, rel_id, image_dir, offset)
        marker = f"[Image {offset}: relationship={rel_id or 'unknown'}]"
        if saved:
            marker = f"[Image {offset}: path={saved}]"
        images.append(marker)
    return images


def _save_docx_image(paragraph: Any, rel_id: str, image_dir: Path, index: int) -> str:
    """通过 relationship 提取图片二进制落盘，返回文件路径；失败返回空串。"""
    if not rel_id:
        return ""
    try:
        related = paragraph.part.related_parts.get(rel_id)
        blob = getattr(related, "blob", None)
        if not blob:
            return ""
        image_dir.mkdir(parents=True, exist_ok=True)
        content_type = str(getattr(related, "content_type", "") or "")
        extension = _image_extension(content_type)
        target = image_dir / f"{index}{extension}"
        target.write_bytes(blob)
        return str(target)
    except Exception as exc:  # pragma: no cover - depends on docx internals
        LOGGER.debug("DOCX image save failed rel=%s error=%s", rel_id, exc)
        return ""


def _image_extension(content_type: str) -> str:
    return {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/gif": ".gif",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
    }.get(content_type, ".png")


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
