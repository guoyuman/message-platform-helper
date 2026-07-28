"""Chunk repository."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy import desc, func, or_, select
from sqlalchemy.orm import Session

from ...models import KnowledgeChunk
from ..db import ChunkRecord, DocumentRecord
from ..models import Chunk
from .utils import chunk_to_knowledge, stable_uuid


TOKEN_RE = re.compile(r"[a-zA-Z0-9_.#/-]+|[\u4e00-\u9fff]{2,}")


@dataclass
class ChunkRepository:
    session: Session

    def save_many(self, chunks: list[Chunk], embeddings: list[list[float]]) -> list[ChunkRecord]:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length.")
        records: list[ChunkRecord] = []
        for chunk, embedding in zip(chunks, embeddings):
            record = self.session.get(ChunkRecord, stable_uuid(chunk.id))
            if record is None:
                record = ChunkRecord(id=stable_uuid(chunk.id))
                self.session.add(record)
            record.document_id = stable_uuid(chunk.parent_document_id)
            record.parent_id = stable_uuid(chunk.parent_id) if chunk.parent_id else None
            record.content = chunk.content
            record.embedding = embedding
            record.metadata_ = dict(chunk.metadata)
            record.chunk_index = int(chunk.metadata.get("chunk_index") or 0)
            records.append(record)
        return records

    def keyword_search(self, query: str, *, limit: int = 20, filters: dict[str, Any] | None = None) -> list[KnowledgeChunk]:
        results = self._fts_search(query, limit=limit, filters=filters)
        if len(results) >= limit:
            return results[:limit]
        seen = {chunk.id for chunk in results}
        for chunk in self._substring_search(query, limit=limit, filters=filters):
            if chunk.id not in seen:
                results.append(chunk)
                seen.add(chunk.id)
            if len(results) >= limit:
                break
        return results

    def _fts_search(self, query: str, *, limit: int, filters: dict[str, Any] | None) -> list[KnowledgeChunk]:
        try:
            ts_query = func.websearch_to_tsquery("simple", query)
            rank = func.ts_rank_cd(ChunkRecord.content_tsv, ts_query).label("rank")
            stmt = (
                select(ChunkRecord, DocumentRecord, rank)
                .join(DocumentRecord, ChunkRecord.document_id == DocumentRecord.id)
                .where(ChunkRecord.content_tsv.op("@@")(ts_query))
                .order_by(desc(rank))
                .limit(limit)
            )
            rows = self.session.execute(_apply_filters(stmt, filters)).all()
        except Exception:
            self.session.rollback()
            return []
        return [chunk_to_knowledge(chunk, document, score=float(rank_value or 0.0)) for chunk, document, rank_value in rows]

    def _substring_search(self, query: str, *, limit: int, filters: dict[str, Any] | None) -> list[KnowledgeChunk]:
        terms = _query_terms(query)
        if not terms:
            return []
        conditions = [ChunkRecord.content.ilike(f"%{term}%") for term in terms]
        conditions.extend(DocumentRecord.title.ilike(f"%{term}%") for term in terms)
        stmt = (
            select(ChunkRecord, DocumentRecord)
            .join(DocumentRecord, ChunkRecord.document_id == DocumentRecord.id)
            .where(or_(*conditions))
            .order_by(ChunkRecord.updated_at.desc())
            .limit(limit * 3)
        )
        rows = self.session.execute(_apply_filters(stmt, filters)).all()
        scored = [chunk_to_knowledge(chunk, document, score=_substring_score(query, terms, chunk.content, document.title)) for chunk, document in rows]
        return sorted(scored, key=lambda chunk: chunk.score, reverse=True)[:limit]


def _apply_filters(stmt: Any, filters: dict[str, Any] | None) -> Any:
    tags = list((filters or {}).get("tags") or [])
    if tags:
        stmt = stmt.where(ChunkRecord.metadata_["tags"].contains(tags))
    return stmt


def _query_terms(query: str) -> list[str]:
    normalized = str(query or "").strip()
    if not normalized:
        return []
    terms = [term.lower() for term in TOKEN_RE.findall(normalized)]
    for term in list(terms):
        if re.fullmatch(r"[\u4e00-\u9fff]{4,}", term):
            terms.extend(_char_ngrams(term, 2))
            terms.extend(_char_ngrams(term, 3))
    if len(normalized) >= 2:
        terms.append(normalized.lower())
    result: list[str] = []
    for term in terms:
        if term not in result:
            result.append(term)
    return result[:32]


def _char_ngrams(text: str, size: int) -> list[str]:
    return [text[index : index + size] for index in range(0, max(len(text) - size + 1, 0))]


def _substring_score(query: str, terms: list[str], content: str, title: str) -> float:
    haystack = f"{title}\n{content}".lower()
    score = 0.0
    if query.lower() in haystack:
        score += 5.0
    for term in terms:
        if term in title.lower():
            score += 2.0
        if term in haystack:
            score += 0.5
    return score
