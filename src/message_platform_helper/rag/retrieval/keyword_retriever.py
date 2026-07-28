"""PostgreSQL full-text retriever."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ...models import KnowledgeChunk


class KeywordSearchRepository(Protocol):
    def keyword_search(self, query: str, *, limit: int = 20, filters: dict[str, Any] | None = None) -> list[KnowledgeChunk]:
        raise NotImplementedError


@dataclass
class KeywordRetriever:
    repository: KeywordSearchRepository
    top_k: int = 20

    def retrieve(self, query: str, *, limit: int = 5, tags: list[str] | None = None, filters: dict[str, Any] | None = None) -> list[KnowledgeChunk]:
        merged_filters = dict(filters or {})
        if tags:
            merged_filters["tags"] = tags
        return self.repository.keyword_search(query, limit=max(limit, self.top_k), filters=merged_filters)[:limit]
