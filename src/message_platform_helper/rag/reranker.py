"""Reranker boundary for RAG results."""

from __future__ import annotations

import re
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


@dataclass
class RuleBasedReranker:
    def rerank(self, query: str, chunks: list[KnowledgeChunk], *, limit: int = 5) -> list[KnowledgeChunk]:
        terms = _query_terms(query)
        scored = [_with_score(chunk, chunk.score + _rule_score(chunk, terms)) for chunk in chunks]
        return sorted(scored, key=lambda chunk: chunk.score, reverse=True)[:limit]


class CrossEncoderReranker:
    def rerank(self, query: str, chunks: list[KnowledgeChunk], *, limit: int = 5) -> list[KnowledgeChunk]:
        raise NotImplementedError("CrossEncoderReranker is reserved for a future cross-encoder model integration.")


def _query_terms(query: str) -> list[str]:
    terms: list[str] = []
    for part in re.findall(r"[a-zA-Z0-9_.#/-]+|[\u4e00-\u9fff]{2,}", query or ""):
        lowered = part.lower()
        terms.append(lowered)
        if re.fullmatch(r"[\u4e00-\u9fff]{4,}", part):
            terms.extend(_char_ngrams(lowered, 2))
            terms.extend(_char_ngrams(lowered, 3))
    return list(dict.fromkeys(terms))


def _char_ngrams(text: str, size: int) -> list[str]:
    return [text[index : index + size] for index in range(0, max(len(text) - size + 1, 0))]


def _rule_score(chunk: KnowledgeChunk, terms: list[str]) -> float:
    title = chunk.title.lower()
    content = chunk.content.lower()
    source = chunk.source.lower()
    tags = {tag.lower() for tag in chunk.tags}
    score = 0.0
    for term in terms:
        if term in title:
            score += 2.4
        if term in tags:
            score += 1.8
        if term in source:
            score += 0.8
        if term in content:
            score += 0.55
    if terms and all(term in f"{title}\n{content}" for term in terms[: min(len(terms), 4)]):
        score += 1.2
    return score


def _with_score(chunk: KnowledgeChunk, score: float) -> KnowledgeChunk:
    return KnowledgeChunk(
        id=chunk.id,
        title=chunk.title,
        content=chunk.content,
        source=chunk.source,
        tags=list(chunk.tags),
        score=score,
    )


__all__ = ["CrossEncoderReranker", "Reranker", "RuleBasedReranker", "ScoreReranker"]
