from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from message_platform_helper.models import KnowledgeChunk
from message_platform_helper.rag import Document, DocumentIngestionPipeline
from message_platform_helper.rag.knowledge_base import normalize_content, normalize_title, phrase_boost, tokenize
from message_platform_helper.rag.models import Chunk, normalize_tags


@dataclass
class InMemoryKnowledgeBase:
    chunks: dict[str, KnowledgeChunk] = field(default_factory=dict)
    documents: dict[tuple[str, str], Document] = field(default_factory=dict)
    database_url: str = "memory://rag"
    path: str = "memory://rag"
    embedding_provider: object | None = None
    last_ingestion: object | None = None

    def ingest(self, chunks: list[Chunk] | str, content: str | None = None, **kwargs):
        if isinstance(chunks, str):
            if content is None:
                raise ValueError("Knowledge content is required.")
            result = self.ingest_text(chunks, content, source=kwargs.get("source", "manual"), tags=kwargs.get("tags"))
            return result[0]
        return [self._save_chunk(chunk) for chunk in chunks]

    def ingest_text(self, title: str, content: str, *, source: str = "manual", tags: list[str] | None = None, replace: bool = True, strategy=None):
        from message_platform_helper.rag.loader import TextLoader

        document = TextLoader().load(content, title=title, tags=tags)
        document.metadata.source = source
        return self.ingest_document(document, strategy=strategy, replace=replace)

    def ingest_file(self, path: Path, tags: list[str] | None = None, *, strategy=None, replace: bool = True):
        from message_platform_helper.rag.loader.base import default_loader_factory

        document = default_loader_factory().create(path).load(path, tags=tags)
        return self.ingest_document(document, strategy=strategy, replace=replace)

    def ingest_document(self, document: Document, *, strategy=None, replace: bool = True):
        if replace:
            self.delete_document(document.title, document.metadata.source)
        result = DocumentIngestionPipeline(chunk_strategy=strategy).ingest_document(document)
        if not result.chunks:
            from message_platform_helper.rag.chunker import RecursiveChunkStrategy

            result = DocumentIngestionPipeline(chunk_strategy=RecursiveChunkStrategy()).ingest_document(document)
        self.documents[(document.title, document.metadata.source)] = document
        return [self._save_chunk(chunk) for chunk in result.chunks]

    def search(self, query: str, *, limit: int = 5, tags: list[str] | None = None):
        query_tokens = tokenize(query)
        query_tokens.extend(_char_ngrams(query, 2))
        tag_filter = set(tags or [])
        results: list[KnowledgeChunk] = []
        for chunk in self.chunks.values():
            if tag_filter and not tag_filter.intersection(chunk.tags):
                continue
            content_tokens = tokenize(chunk.title + "\n" + chunk.content)
            content_tokens.extend(_char_ngrams(chunk.title + "\n" + chunk.content, 2))
            score = sum(1 for token in query_tokens if token in content_tokens) + phrase_boost(query, chunk.title, chunk.content)
            if score > 0:
                results.append(_with_score(chunk, float(score)))
        return sorted(results, key=lambda item: item.score, reverse=True)[:limit]

    def count(self) -> int:
        return len(self.chunks)

    def stats(self):
        return {"database_url": self.database_url, "total_chunks": self.count(), "sources": [], "tags": [], "latest": []}

    def delete_source(self, source: str) -> int:
        ids = [chunk.id for chunk in self.chunks.values() if chunk.source == source]
        for chunk_id in ids:
            del self.chunks[chunk_id]
        self.documents = {
            key: document
            for key, document in self.documents.items()
            if document.metadata.source != source
        }
        return len(ids)

    def document_versions(self, source_prefix: str) -> dict[str, str]:
        return {
            document.metadata.source: str(document.metadata.extra.get("file_sha1") or "")
            for document in self.documents.values()
            if document.metadata.source.startswith(source_prefix)
        }

    def delete_file_sources_under(self, directory: Path) -> int:
        root = Path(directory).resolve()
        sources = {
            document.metadata.source
            for document in self.documents.values()
            if not document.metadata.source.startswith("default:")
            and _source_is_under_directory(document.metadata.source, root)
        }
        return sum(self.delete_source(source) for source in sources)

    def delete_document(self, title: str, source: str) -> int:
        normalized_title = normalize_title(title)
        ids = [chunk.id for chunk in self.chunks.values() if chunk.source == source and (chunk.title == normalized_title or chunk.title.startswith(normalized_title))]
        for chunk_id in ids:
            del self.chunks[chunk_id]
        return len(ids)

    def _save_chunk(self, chunk: Chunk) -> KnowledgeChunk:
        source = str(chunk.metadata.get("source") or "manual")
        tags = normalize_tags([str(tag) for tag in chunk.metadata.get("tags", [])])
        legacy = KnowledgeChunk(
            id=chunk.id,
            title=normalize_title(chunk.title),
            content=normalize_content(chunk.content),
            source=source,
            tags=tags,
            score=0.0,
        )
        self.chunks[legacy.id] = legacy
        return legacy


def _with_score(chunk: KnowledgeChunk, score: float) -> KnowledgeChunk:
    return KnowledgeChunk(id=chunk.id, title=chunk.title, content=chunk.content, source=chunk.source, tags=list(chunk.tags), score=score)


def _char_ngrams(text: str, size: int) -> list[str]:
    chars = [char for char in str(text or "") if "\u4e00" <= char <= "\u9fff"]
    return ["".join(chars[index : index + size]) for index in range(0, max(len(chars) - size + 1, 0))]


def _source_is_under_directory(source: str, root: Path) -> bool:
    candidate = Path(source)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        candidate.resolve().relative_to(root)
    except ValueError:
        return False
    return True
