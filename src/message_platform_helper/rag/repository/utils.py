"""Repository helpers."""

from __future__ import annotations

import uuid
from typing import Any

from ...models import KnowledgeChunk
from ..db import ChunkRecord, DocumentRecord


RAG_NAMESPACE = uuid.UUID("f7d5f5c4-06b6-4b19-98d1-0b47b3ed8bb4")


def stable_uuid(value: str | uuid.UUID) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except ValueError:
        return uuid.uuid5(RAG_NAMESPACE, str(value))


def chunk_to_knowledge(chunk: ChunkRecord, document: DocumentRecord | None = None, *, score: float = 0.0) -> KnowledgeChunk:
    metadata: dict[str, Any] = dict(chunk.metadata_ or {})
    document_metadata = dict(document.metadata_ or {}) if document is not None else {}
    tags = metadata.get("tags") or document_metadata.get("tags") or []
    title = str(metadata.get("title") or metadata.get("section") or (document.title if document is not None else "chunk"))
    source = str(metadata.get("source") or (document.source if document is not None else "postgres"))
    return KnowledgeChunk(
        id=str(chunk.id),
        title=title,
        content=chunk.content,
        source=source,
        tags=[str(tag) for tag in tags],
        score=score,
    )
