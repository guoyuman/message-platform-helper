"""Parent-child chunking strategy."""

from __future__ import annotations

from collections.abc import Callable

from ..models import Chunk, Document, DocumentSection, ParentDocument
from .base import ChunkStrategy
from .recursive import RecursiveChunkStrategy

EmbeddingFn = Callable[[list[str]], list[list[float]]]


class ParentChildChunkStrategy(ChunkStrategy):
    def __init__(
        self,
        child_chunk_size: int = 900,
        child_chunk_overlap: int = 120,
        *,
        max_child_chars: int | None = None,
        max_child_size: int | None = None,
        embedding_fn: EmbeddingFn | None = None,
    ) -> None:
        self._child_strategy = RecursiveChunkStrategy(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            max_chunk_chars=max_child_chars,
            max_chunk_size=max_child_size,
            embedding_fn=embedding_fn,
        )

    def parent(self, document: Document) -> ParentDocument:
        return ParentDocument(
            id=document.id,
            title=document.title,
            content=document.content,
            metadata=document.metadata.to_dict(),
        )

    def chunk(self, document: Document, sections: list[DocumentSection]) -> list[Chunk]:
        chunks = self._child_strategy.chunk(document, sections)
        for chunk in chunks:
            chunk.metadata["retrieval_role"] = "child"
            chunk.metadata["parent_document_id"] = document.id
        return chunks
