"""Metadata helpers for RAG chunks."""

from __future__ import annotations

from .models import Document, DocumentSection, JsonDict


def build_chunk_metadata(
    document: Document,
    section: DocumentSection,
    *,
    chunk_index: int,
    chunk_count: int,
) -> JsonDict:
    return {
        "parent_document_id": document.id,
        "document": document.title,
        "section": section.title,
        "section_level": section.level,
        "section_kind": section.kind,
        "level": section.level,
        "title": section.title,
        "source": document.metadata.source,
        "tags": list(document.metadata.tags),
        "format": document.metadata.format,
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        **section.metadata,
    }
