"""Hybrid retrieval scoring helpers."""

from __future__ import annotations

from dataclasses import dataclass

from ...models import KnowledgeChunk


@dataclass(frozen=True)
class RetrievalConfig:
    keyword_weight: float = 0.7
    vector_weight: float = 0.3
    similarity_threshold: float = 0.2
    rrf_k: int = 60


def weighted_score_fusion(
    keyword_results: list[KnowledgeChunk],
    vector_results: list[KnowledgeChunk],
    *,
    config: RetrievalConfig,
    limit: int,
) -> list[KnowledgeChunk]:
    keyword_scores = _normalized_scores(keyword_results, fallback_k=config.rrf_k)
    vector_scores = _bounded_scores(vector_results)
    chunks: dict[str, KnowledgeChunk] = {}
    for chunk in [*keyword_results, *vector_results]:
        chunks.setdefault(chunk.id, chunk)

    scored: list[KnowledgeChunk] = []
    for chunk_id, chunk in chunks.items():
        score = config.keyword_weight * keyword_scores.get(chunk_id, 0.0)
        score += config.vector_weight * vector_scores.get(chunk_id, 0.0)
        if score >= config.similarity_threshold:
            scored.append(_with_score(chunk, score))
    return sorted(scored, key=lambda chunk: chunk.score, reverse=True)[:limit]


def _normalized_scores(chunks: list[KnowledgeChunk], *, fallback_k: int) -> dict[str, float]:
    if not chunks:
        return {}
    raw_scores = [max(float(chunk.score or 0.0), 0.0) for chunk in chunks]
    max_score = max(raw_scores)
    min_score = min(raw_scores)
    result: dict[str, float] = {}
    for rank, chunk in enumerate(chunks, start=1):
        score = max(float(chunk.score or 0.0), 0.0)
        if max_score > min_score:
            normalized = 0.25 + 0.75 * ((score - min_score) / (max_score - min_score))
        elif max_score > 0:
            normalized = 1.0
        else:
            normalized = 1.0 / (fallback_k + rank)
        normalized += 1.0 / (fallback_k + rank)
        result[chunk.id] = max(result.get(chunk.id, 0.0), normalized)
    return result


def _bounded_scores(chunks: list[KnowledgeChunk]) -> dict[str, float]:
    result: dict[str, float] = {}
    for chunk in chunks:
        score = min(max(float(chunk.score or 0.0), 0.0), 1.0)
        result[chunk.id] = max(result.get(chunk.id, 0.0), score)
    return result


def _with_score(chunk: KnowledgeChunk, score: float) -> KnowledgeChunk:
    return KnowledgeChunk(
        id=chunk.id,
        title=chunk.title,
        content=chunk.content,
        source=chunk.source,
        tags=list(chunk.tags),
        score=score,
    )


__all__ = ["RetrievalConfig", "weighted_score_fusion"]
