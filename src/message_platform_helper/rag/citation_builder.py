"""Citation builder for RAG responses."""

from __future__ import annotations

from dataclasses import dataclass

from ..models import JsonDict, KnowledgeChunk


@dataclass
class CitationBuilder:
    def build(self, chunks: list[KnowledgeChunk]) -> list[JsonDict]:
        return [
            {
                "id": chunk.id,
                "title": chunk.title,
                "source": chunk.source,
                "tags": chunk.tags,
                "score": round(chunk.score, 4),
            }
            for chunk in chunks
        ]


__all__ = ["CitationBuilder"]
