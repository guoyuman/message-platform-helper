"""Reciprocal Rank Fusion."""

from __future__ import annotations

from ...models import KnowledgeChunk


def reciprocal_rank_fusion(result_sets: list[list[KnowledgeChunk]], *, limit: int = 20, k: int = 60) -> list[KnowledgeChunk]:
    fused: dict[str, KnowledgeChunk] = {}
    scores: dict[str, float] = {}
    for results in result_sets:
        for rank, chunk in enumerate(results, start=1):
            fused.setdefault(chunk.id, chunk)
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
    ranked = sorted(fused.values(), key=lambda chunk: scores[chunk.id], reverse=True)
    return [_with_score(chunk, scores[chunk.id]) for chunk in ranked[:limit]]


def _with_score(chunk: KnowledgeChunk, score: float) -> KnowledgeChunk:
    metadata = dict(chunk.metadata)
    metadata["retrieval_score"] = score
    return KnowledgeChunk(
        id=chunk.id,
        title=chunk.title,
        content=chunk.content,
        source=chunk.source,
        tags=list(chunk.tags),
        score=score,
        metadata=metadata,
    )
