"""Prompt builder for RAG answer generation."""

from __future__ import annotations

from dataclasses import dataclass

from ..models import KnowledgeChunk


@dataclass
class RagPromptBuilder:
    max_excerpt_chars: int = 700

    def build(self, question: str, chunks: list[KnowledgeChunk]) -> str:
        sections = [f"Question: {question}", "Enterprise knowledge:"]
        for index, chunk in enumerate(chunks, start=1):
            excerpt = " ".join(chunk.content.split())[: self.max_excerpt_chars]
            sections.append(f"[{index}] {chunk.title}\nSource: {chunk.source}\n{excerpt}")
        return "\n\n".join(sections)


__all__ = ["RagPromptBuilder"]
