"""Plain text document loader."""

from __future__ import annotations

from pathlib import Path

from ..models import Document, DocumentMetadata, file_document_identity, normalize_tags, stable_document_id
from .base import DocumentLoader


class TextLoader(DocumentLoader):
    format = "text"
    extensions = (".txt", ".text")

    def load(self, source: str | Path, *, title: str | None = None, tags: list[str] | None = None) -> Document:
        if isinstance(source, Path) or Path(str(source)).exists():
            path = Path(source)
            content = path.read_text(encoding="utf-8")
            document_source = str(path)
            document_title = title or path.stem.replace("_", " ")
            extra = file_document_identity(str(path.resolve()), content)
        else:
            content = str(source)
            document_source = "manual"
            document_title = title or "Untitled"
            extra = {}
        document_title = _required_title(document_title)
        metadata = DocumentMetadata(format="text", source=document_source, tags=normalize_tags(tags), extra=extra)
        return Document(
            id=stable_document_id(document_source, document_title, content),
            title=document_title,
            content=content,
            metadata=metadata,
        )


def _required_title(title: str) -> str:
    normalized = str(title or "").strip()
    if not normalized:
        raise ValueError("Document title is required.")
    return normalized
