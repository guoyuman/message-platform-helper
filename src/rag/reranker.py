"""Reranker boundary for RAG results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..models import KnowledgeChunk


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[KnowledgeChunk], *, limit: int = 5) -> list[KnowledgeChunk]:
        raise NotImplementedError


@dataclass
class ScoreReranker:
    def rerank(self, query: str, chunks: list[KnowledgeChunk], *, limit: int = 5) -> list[KnowledgeChunk]:
        return sorted(chunks, key=lambda chunk: chunk.score, reverse=True)[:limit]


__all__ = ["Reranker", "ScoreReranker"]
