"""PostgreSQL-backed RAG knowledge base facade."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import overload

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from ..models import JsonDict, KnowledgeChunk
from .chunker import ChunkStrategy, MarkdownChunkStrategy, RecursiveChunkStrategy
from .db import ChunkRecord, DocumentRecord, build_session_factory, rag_database_url
from .embedding import EmbeddingProvider, build_embedding_provider
from .ingestion import DocumentIngestionPipeline
from .loader import TextLoader
from .models import Chunk, Document, DocumentMetadata, normalize_tags, stable_document_id
from .repository import ChunkRepository, DocumentRepository
from .repository.utils import chunk_to_knowledge, stable_uuid


LOGGER = logging.getLogger(__name__)
TOKEN_RE = re.compile(r"[a-zA-Z0-9_.#/-]+|[\u4e00-\u9fff]")
DEFAULT_KNOWLEDGE_DIR = Path(__file__).resolve().parents[3] / "knowledge"


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "") if token.strip()]


@dataclass
class KnowledgeDocument:
    title: str
    content: str
    source: str
    tags: list[str]


@dataclass
class IngestionSummary:
    file_name: str
    file_type: str
    text_length: int
    chunk_count: int
    embedding_dimensions: int

    def to_dict(self) -> JsonDict:
        return {
            "file_name": self.file_name,
            "file_type": self.file_type,
            "text_length": self.text_length,
            "chunk_count": self.chunk_count,
            "embedding_dimensions": self.embedding_dimensions,
        }


class KnowledgeBase:
    """Compatibility facade over the PostgreSQL RAG tables."""

    def __init__(
        self,
        database_url: str | None = None,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        session_factory: sessionmaker[Session] | None = None,
    ) -> None:
        self.database_url = database_url or rag_database_url()
        self.path = self.database_url
        self.embedding_provider = embedding_provider or build_embedding_provider()
        self._factory = session_factory or build_session_factory(self.database_url)
        self.last_ingestion: IngestionSummary | None = None

    @overload
    def ingest(self, chunks: list[Chunk]) -> list[KnowledgeChunk]:
        ...

    @overload
    def ingest(
        self,
        chunks: str,
        content: str,
        *,
        source: str = "manual",
        tags: list[str] | None = None,
    ) -> KnowledgeChunk:
        ...

    def ingest(
        self,
        chunks: list[Chunk] | str,
        content: str | None = None,
        *,
        source: str = "manual",
        tags: list[str] | None = None,
    ) -> list[KnowledgeChunk] | KnowledgeChunk:
        if isinstance(chunks, str):
            if content is None:
                raise ValueError("Knowledge content is required.")
            return self.ingest_legacy(chunks, content, source=source, tags=tags)
        return self.save_chunks(chunks)

    def ingest_legacy(
        self,
        title: str,
        content: str,
        *,
        source: str = "manual",
        tags: list[str] | None = None,
    ) -> KnowledgeChunk:
        chunks = self.ingest_text(title, content, source=source, tags=tags)
        if not chunks:
            raise ValueError("Knowledge content did not produce chunks.")
        return chunks[0]

    def ingest_text(
        self,
        title: str,
        text: str,
        *,
        source: str = "manual",
        tags: list[str] | None = None,
        replace: bool = True,
        strategy: ChunkStrategy | None = None,
    ) -> list[KnowledgeChunk]:
        document = TextLoader().load(text, title=title, tags=tags)
        document.metadata.source = str(source or "manual")
        return self.ingest_document(document, strategy=strategy or RecursiveChunkStrategy(), replace=replace)

    def ingest_file(
        self,
        path: Path,
        tags: list[str] | None = None,
        *,
        strategy: ChunkStrategy | None = None,
        replace: bool = True,
    ) -> list[KnowledgeChunk]:
        result = DocumentIngestionPipeline(chunk_strategy=strategy).ingest_path(path, tags=tags)
        return self._persist_ingestion(result.document, result.chunks, replace=replace, file_name=Path(path).name)

    def ingest_document(
        self,
        document: Document,
        *,
        strategy: ChunkStrategy | None = None,
        replace: bool = True,
    ) -> list[KnowledgeChunk]:
        result = DocumentIngestionPipeline(chunk_strategy=strategy).ingest_document(document)
        return self._persist_ingestion(result.document, result.chunks, replace=replace, file_name=document.title)

    def save_chunks(self, chunks: list[Chunk]) -> list[KnowledgeChunk]:
        if not chunks:
            return []
        embeddings = self.embedding_provider.embed([chunk.content for chunk in chunks])
        self._validate_embeddings(embeddings)
        with self._factory() as session:
            ChunkRepository(session).save_many(chunks, embeddings)
            session.commit()
        return [chunk.to_legacy() for chunk in chunks]

    def delete(self, chunk_id: str) -> int:
        with self._factory() as session:
            result = session.execute(delete(ChunkRecord).where(ChunkRecord.id == stable_uuid(chunk_id)))
            session.commit()
            return int(result.rowcount or 0)

    def delete_document(self, title: str, source: str) -> int:
        title = normalize_title(title)
        source = str(source or "manual")
        with self._factory() as session:
            ids = [
                row[0]
                for row in session.execute(
                    select(DocumentRecord.id).where(DocumentRecord.title == title, DocumentRecord.source == source)
                ).all()
            ]
            if not ids:
                return 0
            chunk_count = int(session.scalar(select(func.count()).select_from(ChunkRecord).where(ChunkRecord.document_id.in_(ids))) or 0)
            session.execute(delete(DocumentRecord).where(DocumentRecord.id.in_(ids)))
            session.commit()
            return chunk_count

    def delete_source(self, source: str) -> int:
        source = str(source or "").strip()
        if not source:
            return 0
        with self._factory() as session:
            ids = [row[0] for row in session.execute(select(DocumentRecord.id).where(DocumentRecord.source == source)).all()]
            if not ids:
                return 0
            chunk_count = int(session.scalar(select(func.count()).select_from(ChunkRecord).where(ChunkRecord.document_id.in_(ids))) or 0)
            session.execute(delete(DocumentRecord).where(DocumentRecord.id.in_(ids)))
            session.commit()
            return chunk_count

    def search(self, query: str, *, limit: int = 5, tags: list[str] | None = None) -> list[KnowledgeChunk]:
        from .repository import VectorRepository
        from .retrieval import HybridRetriever, KeywordRetriever, VectorRetriever

        with self._factory() as session:
            retriever = HybridRetriever(
                KeywordRetriever(ChunkRepository(session)),
                VectorRetriever(VectorRepository(session), self.embedding_provider),
            )
            return retriever.retrieve(query, limit=limit, tags=tags)

    def list_chunks(self, *, limit: int = 20) -> list[KnowledgeChunk]:
        with self._factory() as session:
            rows = session.execute(
                select(ChunkRecord, DocumentRecord)
                .join(DocumentRecord, ChunkRecord.document_id == DocumentRecord.id)
                .order_by(ChunkRecord.updated_at.desc())
                .limit(limit)
            ).all()
            return [chunk_to_knowledge(chunk, document) for chunk, document in rows]

    def count(self) -> int:
        with self._factory() as session:
            return int(session.scalar(select(func.count()).select_from(ChunkRecord)) or 0)

    def stats(self) -> JsonDict:
        with self._factory() as session:
            total = int(session.scalar(select(func.count()).select_from(ChunkRecord)) or 0)
            source_rows = session.execute(
                select(DocumentRecord.source, func.count(ChunkRecord.id))
                .join(ChunkRecord, ChunkRecord.document_id == DocumentRecord.id)
                .group_by(DocumentRecord.source)
                .order_by(func.count(ChunkRecord.id).desc(), DocumentRecord.source)
            ).all()
            latest_rows = session.execute(
                select(ChunkRecord, DocumentRecord)
                .join(DocumentRecord, ChunkRecord.document_id == DocumentRecord.id)
                .order_by(ChunkRecord.updated_at.desc())
                .limit(8)
            ).all()
            tag_rows = session.execute(select(ChunkRecord.metadata_)).all()
        tag_counts: dict[str, int] = {}
        for (metadata,) in tag_rows:
            for tag in list((metadata or {}).get("tags") or []):
                tag_counts[str(tag)] = tag_counts.get(str(tag), 0) + 1
        return {
            "database_url": self.database_url,
            "total_chunks": total,
            "sources": [{"source": source, "chunks": count} for source, count in source_rows],
            "tags": [{"tag": tag, "chunks": count} for tag, count in sorted(tag_counts.items(), key=lambda item: (-item[1], item[0]))],
            "latest": [
                {"id": str(chunk.id), "title": chunk_to_knowledge(chunk, document).title, "source": document.source, "updated_at": chunk.updated_at.isoformat()}
                for chunk, document in latest_rows
            ],
        }

    def import_from(self, source_path: Path) -> int:
        LOGGER.info("Legacy local import is disabled; ignoring source path %s", source_path)
        return 0

    def _persist_ingestion(self, document: Document, chunks: list[Chunk], *, replace: bool, file_name: str) -> list[KnowledgeChunk]:
        document.content = normalize_content(document.content)
        if not chunks:
            raise ValueError(f"Document produced no chunks: {document.title}")
        embeddings = self.embedding_provider.embed([chunk.content for chunk in chunks])
        dimensions = self._validate_embeddings(embeddings)
        if replace:
            self.delete_document(document.title, document.metadata.source)
        with self._factory() as session:
            DocumentRepository(session).save(document)
            ChunkRepository(session).save_many(chunks, embeddings)
            session.commit()
        self.last_ingestion = IngestionSummary(
            file_name=file_name,
            file_type=document.metadata.format,
            text_length=len(document.content),
            chunk_count=len(chunks),
            embedding_dimensions=dimensions,
        )
        LOGGER.info(
            "RAG ingest completed file=%s type=%s text_length=%s chunks=%s embedding_dimensions=%s",
            file_name,
            document.metadata.format,
            len(document.content),
            len(chunks),
            dimensions,
        )
        return [chunk.to_legacy() for chunk in chunks]

    def _validate_embeddings(self, embeddings: list[list[float]]) -> int:
        if not embeddings:
            raise ValueError("Embedding provider returned no vectors.")
        dimensions = len(embeddings[0])
        if dimensions <= 0:
            raise ValueError("Embedding vectors must not be empty.")
        if any(len(vector) != dimensions for vector in embeddings):
            raise ValueError("Embedding dimensions are inconsistent.")
        return dimensions


def split_text(text: str, max_chars: int = 900) -> list[str]:
    return [chunk.content for chunk in RecursiveChunkStrategy(chunk_size=max_chars).chunk(TextLoader().load(text, title="text"), [])]


def bm25_like(query_tokens: list[str], doc_tokens: list[str]) -> float:
    counts: dict[str, int] = {}
    for token in doc_tokens:
        counts[token] = counts.get(token, 0) + 1
    score = 0.0
    doc_len = max(len(doc_tokens), 1)
    for token in query_tokens:
        tf = counts.get(token, 0)
        if tf:
            score += (1.0 + math.log(tf + 1.0)) / math.sqrt(doc_len / 80.0)
    return score


def phrase_boost(query: str, title: str, content: str) -> float:
    normalized_query = (query or "").strip().lower()
    if not normalized_query:
        return 0.0
    haystacks = (title.lower(), content.lower())
    score = 0.0
    if any(normalized_query in item for item in haystacks):
        score += 4.0
    for token in set(tokenize(query)):
        if len(token) > 1 and any(token in item for item in haystacks):
            score += 0.4
    return score


def normalize_title(title: str) -> str:
    normalized = str(title or "").strip()
    if not normalized:
        raise ValueError("Knowledge title is required.")
    return normalized


def normalize_content(content: str) -> str:
    normalized = str(content or "").strip()
    if not normalized:
        raise ValueError("Knowledge content is empty after parsing.")
    return normalized


def seed_default_knowledge(kb: KnowledgeBase, knowledge_dir: Path | None = None) -> None:
    documents = load_default_knowledge_documents(knowledge_dir or DEFAULT_KNOWLEDGE_DIR)
    if not documents:
        return
    kb.delete_source("default")
    for knowledge_document in documents:
        document = Document(
            id=stable_document_id(knowledge_document.source, knowledge_document.title, knowledge_document.content),
            title=knowledge_document.title,
            content=knowledge_document.content,
            metadata=DocumentMetadata(format="markdown", source=knowledge_document.source, tags=knowledge_document.tags),
        )
        kb.ingest_document(document, replace=True, strategy=MarkdownChunkStrategy())


def load_default_knowledge_documents(knowledge_dir: Path | None = None) -> list[KnowledgeDocument]:
    root = knowledge_dir or DEFAULT_KNOWLEDGE_DIR
    if not root.exists():
        return []
    paths = sorted(path for path in root.rglob("*.md") if path.is_file())
    return [_read_knowledge_document(path, root) for path in paths]


def _read_knowledge_document(path: Path, root: Path) -> KnowledgeDocument:
    from .loader.markdown_loader import MarkdownLoader

    document = MarkdownLoader().load(path)
    return KnowledgeDocument(title=document.title, content=document.content, source=_default_document_source(path, root), tags=document.metadata.tags)


def _default_document_source(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = Path(path.name)
    return f"default:{relative.as_posix()}"


def _chunk_id(title: str, content: str, source: str) -> str:
    raw = f"{source}\n{title}\n{content}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()
