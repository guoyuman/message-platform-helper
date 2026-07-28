"""pgvector retriever."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ...models import KnowledgeChunk
from ..embedding import EmbeddingProvider


class VectorSearchRepository(Protocol):
    def search(self, embedding: list[float], *, limit: int = 20, filters: dict[str, Any] | None = None) -> list[KnowledgeChunk]:
        raise NotImplementedError


@dataclass
class VectorRetriever:
    repository: VectorSearchRepository
    embedding_provider: EmbeddingProvider
    top_k: int = 20

    def retrieve(self, query: str, *, limit: int = 5, tags: list[str] | None = None, filters: dict[str, Any] | None = None) -> list[KnowledgeChunk]:
        merged_filters = dict(filters or {})
        if tags:
            merged_filters["tags"] = tags
        embedding = self.embedding_provider.embed([query])[0]
        return self.repository.search(embedding, limit=max(limit, self.top_k), filters=merged_filters)[:limit]
