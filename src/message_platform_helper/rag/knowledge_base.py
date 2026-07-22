"""SQLite-backed RAG knowledge base."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import overload

from ..models import JsonDict, KnowledgeChunk, now_ts
from .chunker import ChunkStrategy, MarkdownChunkStrategy, RecursiveChunkStrategy
from .loader import TextLoader
from .loader.base import default_loader_factory
from .models import Chunk, Document, DocumentMetadata, normalize_tags, stable_document_id
from .parser.base import parser_for


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
class KnowledgeBase:
    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with closing(sqlite3.connect(self.path)) as conn:
            conn.executescript(
                """
                create table if not exists parent_documents (
                  id text primary key,
                  title text not null,
                  content text not null,
                  metadata_json text not null,
                  created_at real not null,
                  updated_at real not null
                );

                create table if not exists knowledge_chunks (
                  id text primary key,
                  title text not null,
                  content text not null,
                  source text not null,
                  tags_json text not null,
                  tokens_json text not null,
                  metadata_json text not null default '{}',
                  parent_document_id text not null default '',
                  created_at real not null,
                  updated_at real not null
                );
                """
            )
            columns = {row[1] for row in conn.execute("pragma table_info(knowledge_chunks)").fetchall()}
            if "updated_at" not in columns:
                conn.execute("alter table knowledge_chunks add column updated_at real")
                conn.execute("update knowledge_chunks set updated_at = created_at where updated_at is null")
            if "metadata_json" not in columns:
                conn.execute("alter table knowledge_chunks add column metadata_json text not null default '{}'")
            if "parent_document_id" not in columns:
                conn.execute("alter table knowledge_chunks add column parent_document_id text not null default ''")
            conn.execute("create index if not exists idx_knowledge_chunks_source on knowledge_chunks(source)")
            conn.execute("create index if not exists idx_knowledge_chunks_updated_at on knowledge_chunks(updated_at)")
            conn.execute("create index if not exists idx_knowledge_chunks_parent on knowledge_chunks(parent_document_id)")
            conn.commit()

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
        return [self.save(chunk) for chunk in chunks]

    def save(self, chunk: Chunk) -> KnowledgeChunk:
        source = str(chunk.metadata.get("source") or "manual")
        tags = normalize_tags([str(tag) for tag in chunk.metadata.get("tags", [])])
        tokens = tokenize(chunk.title + "\n" + chunk.content)
        timestamp = now_ts()
        with closing(sqlite3.connect(self.path)) as conn:
            conn.execute(
                """
                insert into knowledge_chunks(
                  id, title, content, source, tags_json, tokens_json, metadata_json,
                  parent_document_id, created_at, updated_at
                )
                values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(id) do update set
                  title=excluded.title,
                  content=excluded.content,
                  source=excluded.source,
                  tags_json=excluded.tags_json,
                  tokens_json=excluded.tokens_json,
                  metadata_json=excluded.metadata_json,
                  parent_document_id=excluded.parent_document_id,
                  updated_at=excluded.updated_at
                """,
                (
                    chunk.id,
                    normalize_title(chunk.title),
                    normalize_content(chunk.content),
                    source,
                    json.dumps(tags, ensure_ascii=False),
                    json.dumps(tokens, ensure_ascii=False),
                    json.dumps(chunk.metadata, ensure_ascii=False),
                    chunk.parent_document_id,
                    timestamp,
                    timestamp,
                ),
            )
            conn.commit()
        return chunk.to_legacy()

    def save_parent_document(self, document: Document) -> None:
        timestamp = now_ts()
        with closing(sqlite3.connect(self.path)) as conn:
            conn.execute(
                """
                insert into parent_documents(id, title, content, metadata_json, created_at, updated_at)
                values(?, ?, ?, ?, ?, ?)
                on conflict(id) do update set
                  title=excluded.title,
                  content=excluded.content,
                  metadata_json=excluded.metadata_json,
                  updated_at=excluded.updated_at
                """,
                (
                    document.id,
                    normalize_title(document.title),
                    normalize_content(document.content),
                    json.dumps(document.metadata.to_dict(), ensure_ascii=False),
                    timestamp,
                    timestamp,
                ),
            )
            conn.commit()

    def ingest_legacy(
        self,
        title: str,
        content: str,
        *,
        source: str = "manual",
        tags: list[str] | None = None,
    ) -> KnowledgeChunk:
        document = TextLoader().load(content, title=title, tags=tags)
        document.metadata.source = str(source or "manual")
        chunk = Chunk(
            id=_chunk_id(title, content, source),
            title=normalize_title(title),
            content=normalize_content(content),
            parent_document_id=document.id,
            metadata={
                "parent_document_id": document.id,
                "document": normalize_title(title),
                "section": normalize_title(title),
                "section_level": 0,
                "level": 0,
                "title": normalize_title(title),
                "source": document.metadata.source,
                "tags": normalize_tags(tags),
                "format": "text",
                "chunk_index": 0,
                "chunk_count": 1,
            },
        )
        self.save_parent_document(document)
        return self.save(chunk)

    def ingest_file(
        self,
        path: Path,
        tags: list[str] | None = None,
        *,
        strategy: ChunkStrategy | None = None,
        replace: bool = True,
    ) -> list[KnowledgeChunk]:
        loader = default_loader_factory().create(path)
        document = loader.load(path, tags=tags)
        return self.ingest_document(document, strategy=strategy, replace=replace)

    def ingest_document(
        self,
        document: Document,
        *,
        strategy: ChunkStrategy | None = None,
        replace: bool = True,
    ) -> list[KnowledgeChunk]:
        if replace:
            self.delete_document(document.title, document.metadata.source)
        self.save_parent_document(document)
        parser = parser_for(document)
        sections = parser.parse(document)
        chunk_strategy = strategy or _default_strategy(document)
        chunks = chunk_strategy.chunk(document, sections)
        return self.ingest(chunks)

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
        if replace:
            self.delete_document(document.title, document.metadata.source)
        self.save_parent_document(document)
        sections = parser_for(document).parse(document)
        chunks = (strategy or RecursiveChunkStrategy()).chunk(document, sections)
        return self.ingest(chunks)

    def delete(self, chunk_id: str) -> int:
        with closing(sqlite3.connect(self.path)) as conn:
            cursor = conn.execute("delete from knowledge_chunks where id = ?", (chunk_id,))
            deleted = cursor.rowcount if cursor.rowcount is not None else 0
            conn.commit()
        return deleted

    def delete_document(self, title: str, source: str) -> int:
        title = normalize_title(title)
        source = str(source or "manual")
        pattern = f"{title} #%"
        with closing(sqlite3.connect(self.path)) as conn:
            cursor = conn.execute(
                "delete from knowledge_chunks where source = ? and (title = ? or title like ? or json_extract(metadata_json, '$.document') = ?)",
                (source, title, pattern, title),
            )
            deleted = cursor.rowcount if cursor.rowcount is not None else 0
            conn.execute("delete from parent_documents where title = ? and json_extract(metadata_json, '$.source') = ?", (title, source))
            conn.commit()
        return deleted

    def delete_source(self, source: str) -> int:
        source = str(source or "").strip()
        if not source:
            return 0
        with closing(sqlite3.connect(self.path)) as conn:
            cursor = conn.execute("delete from knowledge_chunks where source = ?", (source,))
            deleted = cursor.rowcount if cursor.rowcount is not None else 0
            conn.execute("delete from parent_documents where json_extract(metadata_json, '$.source') = ?", (source,))
            conn.commit()
        return deleted

    def search(self, query: str, *, limit: int = 5, tags: list[str] | None = None) -> list[KnowledgeChunk]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        tag_filter = set(tags or [])
        with closing(sqlite3.connect(self.path)) as conn:
            rows = conn.execute("select id, title, content, source, tags_json, tokens_json from knowledge_chunks").fetchall()
        scored: list[KnowledgeChunk] = []
        for row in rows:
            chunk_tags = json.loads(row[4])
            if tag_filter and not tag_filter.intersection(chunk_tags):
                continue
            tokens = json.loads(row[5])
            score = bm25_like(query_tokens, tokens) + phrase_boost(query, row[1], row[2])
            if score <= 0:
                continue
            scored.append(KnowledgeChunk(id=row[0], title=row[1], content=row[2], source=row[3], tags=chunk_tags, score=score))
        return sorted(scored, key=lambda item: item.score, reverse=True)[:limit]

    def list_chunks(self, *, limit: int = 20) -> list[KnowledgeChunk]:
        with closing(sqlite3.connect(self.path)) as conn:
            rows = conn.execute(
                """
                select id, title, content, source, tags_json
                from knowledge_chunks
                order by coalesce(updated_at, created_at) desc
                limit ?
                """,
                (limit,),
            ).fetchall()
        return [KnowledgeChunk(id=row[0], title=row[1], content=row[2], source=row[3], tags=json.loads(row[4])) for row in rows]

    def count(self) -> int:
        with closing(sqlite3.connect(self.path)) as conn:
            return int(conn.execute("select count(*) from knowledge_chunks").fetchone()[0])

    def stats(self) -> JsonDict:
        with closing(sqlite3.connect(self.path)) as conn:
            total = int(conn.execute("select count(*) from knowledge_chunks").fetchone()[0])
            sources = [
                {"source": row[0], "chunks": row[1]}
                for row in conn.execute("select source, count(*) from knowledge_chunks group by source order by count(*) desc, source").fetchall()
            ]
            tag_rows = conn.execute("select tags_json from knowledge_chunks").fetchall()
            latest = [
                {"id": row[0], "title": row[1], "source": row[2], "updated_at": row[3]}
                for row in conn.execute(
                    """
                    select id, title, source, coalesce(updated_at, created_at)
                    from knowledge_chunks
                    order by coalesce(updated_at, created_at) desc
                    limit 8
                    """
                ).fetchall()
            ]
        tag_counts: dict[str, int] = {}
        for row in tag_rows:
            for tag in json.loads(row[0]):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        tags = [{"tag": tag, "chunks": count} for tag, count in sorted(tag_counts.items(), key=lambda item: (-item[1], item[0]))]
        return {"db_path": str(self.path), "total_chunks": total, "sources": sources, "tags": tags, "latest": latest}

    def import_from(self, source_path: Path) -> int:
        if not source_path.exists():
            return 0
        try:
            if self.path.resolve() == source_path.resolve():
                return 0
        except OSError:
            return 0
        imported = 0
        with closing(sqlite3.connect(source_path)) as source_conn:
            source_tables = {row[0] for row in source_conn.execute("select name from sqlite_master where type='table'").fetchall()}
            if "knowledge_chunks" not in source_tables:
                return 0
            rows = source_conn.execute("select * from knowledge_chunks").fetchall()
            columns = [column[0] for column in source_conn.execute("select * from knowledge_chunks limit 0").description]
        with closing(sqlite3.connect(self.path)) as target_conn:
            for raw_row in rows:
                row = dict(zip(columns, raw_row))
                if target_conn.execute("select 1 from knowledge_chunks where id = ?", (row["id"],)).fetchone():
                    continue
                target_conn.execute(
                    """
                    insert into knowledge_chunks(
                      id, title, content, source, tags_json, tokens_json, metadata_json,
                      parent_document_id, created_at, updated_at
                    )
                    values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["id"],
                        row["title"],
                        row["content"],
                        row["source"],
                        row["tags_json"],
                        row["tokens_json"],
                        row.get("metadata_json") or "{}",
                        row.get("parent_document_id") or "",
                        row["created_at"],
                        row.get("updated_at") or row["created_at"],
                    ),
                )
                imported += 1
            target_conn.commit()
        return imported


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
        raise ValueError("Knowledge content is required.")
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


def _default_strategy(document: Document) -> ChunkStrategy:
    if document.metadata.format == "markdown":
        return MarkdownChunkStrategy()
    return RecursiveChunkStrategy()


def _chunk_id(title: str, content: str, source: str) -> str:
    raw = f"{source}\n{title}\n{content}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()
