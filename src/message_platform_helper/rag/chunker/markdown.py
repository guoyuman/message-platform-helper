"""Markdown-aware chunking strategy."""

from __future__ import annotations

from collections.abc import Callable

from ..models import Chunk, Document, DocumentSection
from .base import ChunkStrategy
from .recursive import RecursiveChunkStrategy

EmbeddingFn = Callable[[list[str]], list[list[float]]]


class MarkdownChunkStrategy(ChunkStrategy):
    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 50,
        *,
        max_chunk_chars: int | None = None,
        max_chunk_size: int | None = None,
        embedding_fn: EmbeddingFn | None = None,
    ) -> None:
        self._recursive = RecursiveChunkStrategy(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_chunk_chars=max_chunk_chars,
            max_chunk_size=max_chunk_size,
            embedding_fn=embedding_fn,
        )

    def chunk(self, document: Document, sections: list[DocumentSection]) -> list[Chunk]:
        markdown_sections = [section for section in sections if section.level <= 3]
        return self._recursive.chunk(document, markdown_sections or sections)
