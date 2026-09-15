"""Reranker boundary for RAG results."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

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


@dataclass
class BGEReranker:
    """BGE cross-encoder reranker with bounded metadata/rule adjustments."""

    model_name: str = "BAAI/bge-reranker-v2-m3"
    batch_size: int = 16
    model_weight: float = 0.85
    metadata_weight: float = 0.10
    rule_weight: float = 0.05
    fallback: Reranker | None = field(default_factory=ScoreReranker)
    _model: Any = field(default=None, init=False, repr=False)

    def rerank(self, query: str, chunks: list[KnowledgeChunk], *, limit: int = 5) -> list[KnowledgeChunk]:
        if not chunks:
            return []
        try:
            model_scores = self._predict(query, chunks)
        except ImportError:
            if self.fallback is None:
                raise
            return self.fallback.rerank(query, chunks, limit=limit)

        metadata_scores = _normalize([_metadata_score(chunk) for chunk in chunks])
        rule_scores = _normalize([_rule_score(chunk, _query_terms(query)) for chunk in chunks])
        ranked: list[KnowledgeChunk] = []
        for chunk, model_score, metadata_score, rule_score in zip(
            chunks,
            _normalize(model_scores),
            metadata_scores,
            rule_scores,
        ):
            final_score = (
                self.model_weight * model_score
                + self.metadata_weight * metadata_score
                + self.rule_weight * rule_score
            )
            metadata = dict(chunk.metadata)
            metadata.update(
                {
                    "rerank_model_score": model_score,
                    "rerank_metadata_score": metadata_score,
                    "rerank_rule_score": rule_score,
                    "rerank_score": final_score,
                }
            )
            ranked.append(
                KnowledgeChunk(
                    id=chunk.id,
                    title=chunk.title,
                    content=chunk.content,
                    source=chunk.source,
                    tags=list(chunk.tags),
                    score=final_score,
                    metadata=metadata,
                )
            )
        return sorted(ranked, key=lambda chunk: chunk.score, reverse=True)[:limit]

    def _predict(self, query: str, chunks: list[KnowledgeChunk]) -> list[float]:
        model = self._get_model()
        pairs = [(query, f"{chunk.title}\n{chunk.content}") for chunk in chunks]
        scores = model.predict(
            pairs,
            batch_size=max(1, self.batch_size),
            show_progress_bar=False,
        )
        return [float(score) for score in scores]

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("BGE reranking requires sentence-transformers to be installed.") from exc
        self._model = CrossEncoder(self.model_name)
        return self._model


class CrossEncoderReranker(BGEReranker):
    """Backward-compatible name for the BGE cross-encoder implementation."""


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
    metadata = dict(chunk.metadata)
    return KnowledgeChunk(
        id=chunk.id,
        title=chunk.title,
        content=chunk.content,
        source=chunk.source,
        tags=list(chunk.tags),
        score=score,
        metadata=metadata,
    )


def _metadata_score(chunk: KnowledgeChunk) -> float:
    value = chunk.metadata.get("metadata_score", chunk.metadata.get("retrieval_score", chunk.score))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high <= low:
        return [1.0 if high > 0 else 0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


__all__ = ["BGEReranker", "CrossEncoderReranker", "Reranker", "RuleBasedReranker", "ScoreReranker"]
