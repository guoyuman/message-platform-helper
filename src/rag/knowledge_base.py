"""Lightweight RAG knowledge base."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from ..models import JsonDict, KnowledgeChunk, now_ts


TOKEN_RE = re.compile(r"[a-zA-Z0-9_.#/-]+|[\u4e00-\u9fff]")
DEFAULT_KNOWLEDGE_DIR = Path(__file__).resolve().parents[3] / "knowledge"


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "") if token.strip()]


@dataclass
class KnowledgeDocument:
    title: str
    content: str
    source: str
    tags: List[str]


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
                create table if not exists knowledge_chunks (
                  id text primary key,
                  title text not null,
                  content text not null,
                  source text not null,
                  tags_json text not null,
                  tokens_json text not null,
                  created_at real not null,
                  updated_at real not null
                );
                """
            )
            columns = {row[1] for row in conn.execute("pragma table_info(knowledge_chunks)").fetchall()}
            if "updated_at" not in columns:
                conn.execute("alter table knowledge_chunks add column updated_at real")
                conn.execute("update knowledge_chunks set updated_at = created_at where updated_at is null")
            conn.execute("create index if not exists idx_knowledge_chunks_source on knowledge_chunks(source)")
            conn.execute("create index if not exists idx_knowledge_chunks_updated_at on knowledge_chunks(updated_at)")
            conn.commit()

    def ingest(self, title: str, content: str, *, source: str = "manual", tags: List[str] | None = None) -> KnowledgeChunk:
        title = normalize_title(title)
        content = normalize_content(content)
        source = str(source or "manual")
        tags = normalize_tags(tags)
        chunk_id = _chunk_id(title, content, source)
        tokens = tokenize(title + "\n" + content)
        timestamp = now_ts()
        with closing(sqlite3.connect(self.path)) as conn:
            conn.execute(
                """
                insert into knowledge_chunks(id, title, content, source, tags_json, tokens_json, created_at, updated_at)
                values(?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(id) do update set
                  title=excluded.title,
                  content=excluded.content,
                  source=excluded.source,
                  tags_json=excluded.tags_json,
                  tokens_json=excluded.tokens_json,
                  updated_at=excluded.updated_at
                """,
                (
                    chunk_id,
                    title,
                    content,
                    source,
                    json.dumps(tags, ensure_ascii=False),
                    json.dumps(tokens, ensure_ascii=False),
                    timestamp,
                    timestamp,
                ),
            )
            conn.commit()
        return KnowledgeChunk(id=chunk_id, title=title, content=content, source=source, tags=tags)

    def ingest_file(self, path: Path, tags: List[str] | None = None) -> List[KnowledgeChunk]:
        text = path.read_text(encoding="utf-8")
        return self.ingest_text(path.stem, text, source=str(path), tags=tags)

    def ingest_text(
        self,
        title: str,
        text: str,
        *,
        source: str = "manual",
        tags: List[str] | None = None,
        replace: bool = True,
    ) -> List[KnowledgeChunk]:
        title = normalize_title(title)
        text = normalize_content(text)
        source = str(source or "manual")
        tags = normalize_tags(tags)
        if replace:
            self.delete_document(title, source)
        chunks: List[KnowledgeChunk] = []
        parts = split_text(text)
        for index, part in enumerate(parts):
            suffix = f" #{index + 1}" if len(parts) > 1 else ""
            chunk_title = title + suffix
            chunk_id = _stable_chunk_id(source, title, index)
            chunks.append(self._ingest_chunk(chunk_id, chunk_title, part, source=source, tags=tags))
        return chunks

    def _ingest_chunk(self, chunk_id: str, title: str, content: str, *, source: str, tags: List[str]) -> KnowledgeChunk:
        tokens = tokenize(title + "\n" + content)
        timestamp = now_ts()
        with closing(sqlite3.connect(self.path)) as conn:
            conn.execute(
                """
                insert into knowledge_chunks(id, title, content, source, tags_json, tokens_json, created_at, updated_at)
                values(?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(id) do update set
                  title=excluded.title,
                  content=excluded.content,
                  source=excluded.source,
                  tags_json=excluded.tags_json,
                  tokens_json=excluded.tokens_json,
                  updated_at=excluded.updated_at
                """,
                (
                    chunk_id,
                    title,
                    content,
                    source,
                    json.dumps(tags, ensure_ascii=False),
                    json.dumps(tokens, ensure_ascii=False),
                    timestamp,
                    timestamp,
                ),
            )
            conn.commit()
        return KnowledgeChunk(id=chunk_id, title=title, content=content, source=source, tags=tags)

    def delete_document(self, title: str, source: str) -> int:
        title = normalize_title(title)
        source = str(source or "manual")
        pattern = f"{title} #%"
        with closing(sqlite3.connect(self.path)) as conn:
            cursor = conn.execute("delete from knowledge_chunks where source = ? and (title = ? or title like ?)", (source, title, pattern))
            deleted = cursor.rowcount if cursor.rowcount is not None else 0
            conn.commit()
        return deleted

    def delete_source(self, source: str) -> int:
        source = str(source or "").strip()
        if not source:
            return 0
        with closing(sqlite3.connect(self.path)) as conn:
            cursor = conn.execute("delete from knowledge_chunks where source = ?", (source,))
            deleted = cursor.rowcount if cursor.rowcount is not None else 0
            conn.commit()
        return deleted

    def search(self, query: str, *, limit: int = 5, tags: List[str] | None = None) -> List[KnowledgeChunk]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        tag_filter = set(tags or [])
        with closing(sqlite3.connect(self.path)) as conn:
            rows = conn.execute("select id, title, content, source, tags_json, tokens_json from knowledge_chunks").fetchall()
        scored: List[KnowledgeChunk] = []
        for row in rows:
            chunk_tags = json.loads(row[4])
            if tag_filter and not tag_filter.intersection(chunk_tags):
                continue
            tokens = json.loads(row[5])
            score = bm25_like(query_tokens, tokens) + phrase_boost(query, row[1], row[2])
            if score <= 0:
                continue
            scored.append(
                KnowledgeChunk(
                    id=row[0],
                    title=row[1],
                    content=row[2],
                    source=row[3],
                    tags=chunk_tags,
                    score=score,
                )
            )
        return sorted(scored, key=lambda item: item.score, reverse=True)[:limit]

    def list_chunks(self, *, limit: int = 20) -> List[KnowledgeChunk]:
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
        return [
            KnowledgeChunk(
                id=row[0],
                title=row[1],
                content=row[2],
                source=row[3],
                tags=json.loads(row[4]),
            )
            for row in rows
        ]

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
        tag_counts: Dict[str, int] = {}
        for row in tag_rows:
            for tag in json.loads(row[0]):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        tags = [{"tag": tag, "chunks": count} for tag, count in sorted(tag_counts.items(), key=lambda item: (-item[1], item[0]))]
        return {
            "db_path": str(self.path),
            "total_chunks": total,
            "sources": sources,
            "tags": tags,
            "latest": latest,
        }

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
            rows = source_conn.execute("select id, title, content, source, tags_json, tokens_json, created_at from knowledge_chunks").fetchall()
        with closing(sqlite3.connect(self.path)) as target_conn:
            for row in rows:
                exists = target_conn.execute("select 1 from knowledge_chunks where id = ?", (row[0],)).fetchone()
                if exists:
                    continue
                target_conn.execute(
                    """
                    insert into knowledge_chunks(id, title, content, source, tags_json, tokens_json, created_at, updated_at)
                    values(?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[6]),
                )
                imported += 1
            target_conn.commit()
        return imported


def split_text(text: str, max_chars: int = 900) -> List[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: List[str] = []
    current = ""
    for paragraph in paragraphs or [text]:
        if len(current) + len(paragraph) + 2 <= max_chars:
            current = (current + "\n\n" + paragraph).strip()
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= max_chars:
            current = paragraph
        else:
            chunks.extend(paragraph[index : index + max_chars] for index in range(0, len(paragraph), max_chars))
            current = ""
    if current:
        chunks.append(current)
    return chunks


def bm25_like(query_tokens: List[str], doc_tokens: List[str]) -> float:
    counts: Dict[str, int] = {}
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


def _chunk_id(title: str, content: str, source: str) -> str:
    raw = f"{source}\n{title}\n{content}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _stable_chunk_id(source: str, title: str, index: int) -> str:
    raw = f"{source}\n{title}\n{index}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


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


def normalize_tags(tags: List[str] | None) -> List[str]:
    result: List[str] = []
    for tag in tags or []:
        normalized = str(tag).strip()
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def seed_default_knowledge(kb: KnowledgeBase, knowledge_dir: Path | None = None) -> None:
    documents = load_default_knowledge_documents(knowledge_dir or DEFAULT_KNOWLEDGE_DIR)
    if not documents:
        return
    kb.delete_source("default")
    for document in documents:
        kb.ingest_text(document.title, document.content, source=document.source, tags=document.tags, replace=True)


def load_default_knowledge_documents(knowledge_dir: Path | None = None) -> List[KnowledgeDocument]:
    root = knowledge_dir or DEFAULT_KNOWLEDGE_DIR
    if not root.exists():
        return []
    paths = sorted(path for path in root.rglob("*.md") if path.is_file())
    return [_read_knowledge_document(path, root) for path in paths]


def _read_knowledge_document(path: Path, root: Path) -> KnowledgeDocument:
    metadata, content = _split_front_matter(path.read_text(encoding="utf-8"))
    title = normalize_title(metadata.get("title") or _first_markdown_heading(content) or path.stem.replace("_", " "))
    tags = normalize_tags(_parse_document_tags(metadata.get("tags", "")))
    return KnowledgeDocument(title=title, content=content, source=_default_document_source(path, root), tags=tags)


def _split_front_matter(text: str) -> tuple[Dict[str, str], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, normalize_content(text)
    for end_index, line in enumerate(lines[1:], start=1):
        if line.strip() != "---":
            continue
        metadata: Dict[str, str] = {}
        for metadata_line in lines[1:end_index]:
            key, separator, value = metadata_line.partition(":")
            if separator:
                metadata[key.strip().lower()] = value.strip()
        return metadata, normalize_content("\n".join(lines[end_index + 1 :]))
    return {}, normalize_content(text)


def _first_markdown_heading(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return ""


def _parse_document_tags(value: str) -> List[str]:
    value = value.strip()
    if not value:
        return []
    if value.startswith("["):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = []
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return [part.strip() for part in value.split(",")]


def _default_document_source(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = Path(path.name)
    return f"default:{relative.as_posix()}"
