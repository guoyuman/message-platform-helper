"""Document ingestion orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..chunker import ChunkStrategy, ParentChildChunkStrategy
from ..loader.base import default_loader_factory
from ..models import Chunk, Document
from ..parser.base import parser_for


@dataclass(frozen=True)
class IngestionResult:
    document: Document
    chunks: list[Chunk]


class DocumentIngestionPipeline:
    """Load, parse, and structure-first chunk documents."""

    def __init__(self, chunk_strategy: ChunkStrategy | None = None) -> None:
        self.chunk_strategy = chunk_strategy or ParentChildChunkStrategy()

    def ingest_path(self, path: str | Path, *, title: str | None = None, tags: list[str] | None = None) -> IngestionResult:
        loader = default_loader_factory().create(path)
        document = loader.load(path, title=title, tags=tags)
        return self.ingest_document(document)

    def ingest_text(self, title: str, content: str, *, source: str = "manual", tags: list[str] | None = None) -> IngestionResult:
        from ..loader import TextLoader

        document = TextLoader().load(content, title=title, tags=tags)
        document.metadata.source = source
        return self.ingest_document(document)

    def ingest_document(self, document: Document) -> IngestionResult:
        sections = parser_for(document).parse(document)
        chunks = self.chunk_strategy.chunk(document, sections)
        return IngestionResult(document=document, chunks=chunks)
