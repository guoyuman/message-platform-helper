"""Vector repository."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ...models import KnowledgeChunk
from ..db import ChunkRecord, DocumentRecord
from .chunk_repository import _apply_filters
from .utils import chunk_to_knowledge


@dataclass
class VectorRepository:
    session: Session

    def search(self, embedding: list[float], *, limit: int = 20, filters: dict[str, Any] | None = None) -> list[KnowledgeChunk]:
        distance = ChunkRecord.embedding.cosine_distance(embedding).label("distance")
        stmt = (
            select(ChunkRecord, DocumentRecord, distance)
            .join(DocumentRecord, ChunkRecord.document_id == DocumentRecord.id)
            .order_by(distance)
            .limit(limit)
        )
        stmt = _apply_filters(stmt, filters)
        rows = self.session.execute(stmt).all()
        return [chunk_to_knowledge(chunk, document, score=1.0 - float(distance_value or 0.0)) for chunk, document, distance_value in rows]
