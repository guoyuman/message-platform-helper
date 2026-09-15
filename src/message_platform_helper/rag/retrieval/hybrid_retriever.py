"""Hybrid keyword + vector retriever."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from ...models import KnowledgeChunk
from .scoring import RetrievalConfig, weighted_score_fusion


LOGGER = logging.getLogger(__name__)


class RetrievalBackend(Protocol):
    def retrieve(self, query: str, *, limit: int = 5, tags: list[str] | None = None, filters: dict[str, Any] | None = None) -> list[KnowledgeChunk]:
        raise NotImplementedError


@dataclass
class HybridRetriever:
    keyword_retriever: RetrievalBackend
    vector_retriever: RetrievalBackend
    keyword_top_k: int = 20
    vector_top_k: int = 20
    final_top_k: int = 50
    config: RetrievalConfig = RetrievalConfig()

    def retrieve(self, query: str, *, limit: int = 5, tags: list[str] | None = None, filters: dict[str, Any] | None = None) -> list[KnowledgeChunk]:
        keyword_results = _retrieve_or_empty("keyword", self.keyword_retriever, query, limit=self.keyword_top_k, tags=tags, filters=filters)
        vector_results = _retrieve_or_empty("vector", self.vector_retriever, query, limit=self.vector_top_k, tags=tags, filters=filters)
        fused = weighted_score_fusion(
            keyword_results,
            vector_results,
            config=self.config,
            # This is the candidate pool for reranking, not the answer limit.
            limit=max(limit, self.final_top_k),
        )
        _log_retrieval_event(
            "rag.retrieval.hybrid.fused",
            query=query,
            limit=limit,
            tags=tags,
            filters=filters,
            keyword_count=len(keyword_results),
            vector_count=len(vector_results),
            fused_count=len(fused),
            keyword_top_chunks=_chunk_snapshots(keyword_results),
            vector_top_chunks=_chunk_snapshots(vector_results),
            fused_top_chunks=_chunk_snapshots(fused),
            weights={"keyword": self.config.keyword_weight, "vector": self.config.vector_weight},
            similarity_threshold=self.config.similarity_threshold,
        )
        return fused


def _retrieve_or_empty(
    stage: str,
    backend: RetrievalBackend,
    query: str,
    *,
    limit: int,
    tags: list[str] | None,
    filters: dict[str, Any] | None,
) -> list[KnowledgeChunk]:
    try:
        results = backend.retrieve(query, limit=limit, tags=tags, filters=filters)
        _log_retrieval_event(
            "rag.retrieval.backend.completed",
            stage=stage,
            query=query,
            limit=limit,
            tags=tags,
            filters=filters,
            count=len(results),
            top_chunks=_chunk_snapshots(results),
        )
        return results
    except Exception as exc:
        LOGGER.exception(
            json.dumps(
                {
                    "event": "rag.retrieval.backend.failed",
                    "stage": stage,
                    "query": query,
                    "limit": limit,
                    "tags": tags,
                    "filters": filters,
                    "error": str(exc),
                },
                ensure_ascii=False,
                default=str,
            )
        )
        return []


def _log_retrieval_event(event: str, **payload: object) -> None:
    LOGGER.info(json.dumps({"event": event, **payload}, ensure_ascii=False, default=str))


def _chunk_snapshots(chunks: list[KnowledgeChunk], *, limit: int = 8) -> list[dict[str, object]]:
    return [
        {
            "id": chunk.id,
            "title": chunk.title,
            "source": chunk.source,
            "tags": chunk.tags,
            "score": round(float(chunk.score or 0.0), 6),
            "content_preview": " ".join(chunk.content.split())[:180],
        }
        for chunk in chunks[:limit]
    ]
