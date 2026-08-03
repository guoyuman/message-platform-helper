"""Recursive section-aware chunking."""

from __future__ import annotations

import re

from ..metadata import build_chunk_metadata
from ..models import Chunk, Document, DocumentSection, stable_chunk_id
from .base import ChunkStrategy


class RecursiveChunkStrategy(ChunkStrategy):
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 120, *, max_chunk_chars: int | None = None) -> None:
        self.max_chunk_chars = int(max_chunk_chars or chunk_size)
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: Document, sections: list[DocumentSection]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section in sections or [DocumentSection(title=document.title, level=0, content=document.content)]:
            parts = split_section(section.content, self.max_chunk_chars, section_kind=section.kind)
            for part_index, part in enumerate(parts):
                global_index = len(chunks)
                metadata = build_chunk_metadata(document, section, chunk_index=part_index, chunk_count=len(parts))
                chunk_title = section.title if len(parts) == 1 else f"{section.title} #{part_index + 1}"
                chunks.append(
                    Chunk(
                        id=stable_chunk_id(document.id, global_index, section.title),
                        title=chunk_title,
                        content=part,
                        parent_document_id=document.id,
                        metadata=metadata,
                    )
                )
        return chunks


def split_section(text: str, max_chunk_chars: int = 600, chunk_overlap: int = 50, *, section_kind: str = "text") -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []
    if section_kind == "image":
        return [normalized]
    if section_kind == "table":
        return _split_table(normalized, max_chunk_chars)

    chunks: list[str] = []
    for block in _semantic_blocks(normalized, section_kind):
        if len(block) <= max_chunk_chars:
            chunks.append(block)
        else:
            chunks.extend(_split_long_semantic_block(block, max_chunk_chars))
    return chunks


def _semantic_blocks(text: str, section_kind: str) -> list[str]:
    blocks = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if section_kind == "list":
        return _list_items(text) or blocks or [text]
    return blocks or [text]


def _list_items(text: str) -> list[str]:
    items: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if re.match(r"^\s*(?:[-*+]|\d+\.)\s+", line) and current:
            items.append("\n".join(current).strip())
            current = [line.strip()]
        else:
            current.append(line.strip())
    if current:
        items.append("\n".join(current).strip())
    return [item for item in items if item]


def _split_table(text: str, chunk_size: int) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) <= 1:
        return [text]
    header = lines[0]
    chunks: list[str] = []
    current: list[str] = [header]
    for row in lines[1:]:
        candidate = "\n".join([*current, row])
        if len(candidate) <= chunk_size or len(current) == 1:
            current.append(row)
            continue
        chunks.append("\n".join(current))
        current = [header, row]
    if current:
        chunks.append("\n".join(current))
    return chunks


def _split_long_semantic_block(text: str, chunk_size: int) -> list[str]:
    sentences = _semantic_units(text)
    if len(sentences) <= 1:
        return [text]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current}{sentence}" if current else sentence
        if len(candidate) <= chunk_size or not current:
            current = candidate
            continue
        chunks.append(current.strip())
        current = sentence
    if current:
        chunks.append(current.strip())
    return [chunk for chunk in chunks if chunk]


def _semantic_units(text: str) -> list[str]:
    matches = re.findall(r".+?(?:[。！？；.!?;]+|$)", text, flags=re.S)
    return [match.strip() for match in matches if match.strip()]
