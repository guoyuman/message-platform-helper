"""Retriever interfaces and KnowledgeBase adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ..models import KnowledgeChunk
from .knowledge_base import KnowledgeBase


class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
        tags: list[str] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[KnowledgeChunk]:
        raise NotImplementedError


@dataclass
class KnowledgeBaseRetriever:
    knowledge_base: KnowledgeBase

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
        tags: list[str] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[KnowledgeChunk]:
        return self.knowledge_base.search(query, limit=limit, tags=tags)


__all__ = ["KnowledgeBaseRetriever", "Retriever"]
