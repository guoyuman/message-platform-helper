"""Context construction for RAG answers."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import JsonDict, KnowledgeChunk
from .citation_builder import CitationBuilder
from .query_analyzer import QueryAnalysis


DEFAULT_INTENT_LIMITS = {
    "troubleshooting": 3,
    "manual": 5,
    "template": 1,
}


@dataclass
class BuiltContext:
    chunks: list[KnowledgeChunk]
    context: str
    citations: list[JsonDict]


@dataclass
class ContextBuilder:
    citation_builder: CitationBuilder = field(default_factory=CitationBuilder)
    intent_limits: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_INTENT_LIMITS))
    max_excerpt_chars: int = 900

    def build(self, query: str, chunks: list[KnowledgeChunk], analysis: QueryAnalysis) -> BuiltContext:
        limit = self.intent_limits.get(analysis.intent, 5)
        selected = chunks[:limit]
        sections: list[str] = []
        for index, chunk in enumerate(selected, start=1):
            excerpt = " ".join(chunk.content.split())[: self.max_excerpt_chars]
            sections.append(f"[{index}] {chunk.title}\nSource: {chunk.source}\n{excerpt}")
        return BuiltContext(
            chunks=selected,
            context="\n\n".join(sections),
            citations=self.citation_builder.build(selected),
        )
