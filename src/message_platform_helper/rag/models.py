"""Shared document models for the RAG ingestion pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal

from ..models import KnowledgeChunk as LegacyKnowledgeChunk
from ..models import now_ts


JsonDict = dict[str, Any]
DocumentFormat = Literal["text", "markdown", "pdf", "docx", "unknown"]
SectionKind = Literal["heading", "paragraph", "list", "table", "text"]


@dataclass
class DocumentMetadata:
    format: DocumentFormat = "unknown"
    source: str = "manual"
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=now_ts)
    updated_at: float = field(default_factory=now_ts)
    extra: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        payload: JsonDict = {
            "format": self.format,
            "source": self.source,
            "tags": list(self.tags),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        payload.update(self.extra)
        return payload


@dataclass
class Document:
    id: str
    title: str
    content: str
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)


@dataclass
class DocumentSection:
    title: str
    content: str
    level: int = 0
    kind: SectionKind = "text"
    metadata: JsonDict = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    title: str
    content: str
    parent_document_id: str
    metadata: JsonDict = field(default_factory=dict)
    parent_id: str | None = None

    @property
    def source(self) -> str:
        return str(self.metadata.get("source") or "manual")

    @property
    def tags(self) -> list[str]:
        tags = self.metadata.get("tags") or []
        return [str(tag) for tag in tags]

    def to_legacy(self, *, score: float = 0.0) -> LegacyKnowledgeChunk:
        return LegacyKnowledgeChunk(
            id=self.id,
            title=self.title,
            content=self.content,
            source=self.source,
            tags=self.tags,
            score=score,
        )


@dataclass
class ParentDocument:
    """Stored parent document for future parent-child retrieval."""

    id: str
    title: str
    content: str
    metadata: JsonDict = field(default_factory=dict)


def stable_document_id(source: str, title: str, content: str = "") -> str:
    raw = f"{source}\n{title}\n{content}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def file_document_identity(path: str, content: str) -> JsonDict:
    return {
        "document_key": stable_document_id("file", str(path)),
        "content_sha1": stable_document_id("content", "", content),
    }


def stable_chunk_id(parent_document_id: str, chunk_index: int, title: str = "") -> str:
    raw = f"{parent_document_id}\n{chunk_index}\n{title}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def normalize_tags(tags: list[str] | None) -> list[str]:
    result: list[str] = []
    for tag in tags or []:
        normalized = str(tag).strip()
        if normalized and normalized not in result:
            result.append(normalized)
    return result
