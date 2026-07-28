"""Recursive section-aware chunking."""

from __future__ import annotations

import re

from ..metadata import build_chunk_metadata
from ..models import Chunk, Document, DocumentSection, stable_chunk_id
from .base import ChunkStrategy


class RecursiveChunkStrategy(ChunkStrategy):
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 120) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: Document, sections: list[DocumentSection]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section in sections or [DocumentSection(title=document.title, level=0, content=document.content)]:
            parts = split_section(section.content, self.chunk_size, self.chunk_overlap)
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


def split_section(text: str, chunk_size: int, chunk_overlap: int = 50) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        return [normalized]

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs or [normalized]:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= chunk_size:
            current = paragraph
        else:
            chunks.extend(_split_long_paragraph(paragraph, chunk_size, chunk_overlap))
            current = ""
    if current:
        chunks.append(current)
    return chunks


def _split_long_paragraph(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    separators = "。！？；.!?;\n"
    chunks: list[str] = []
    start = 0
    step = max(chunk_size - chunk_overlap, 1)
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = max(text.rfind(separator, start, end) for separator in separators)
            if boundary > start + max(chunk_size // 2, 1):
                end = boundary + 1
        chunks.append(text[start:end].strip())
        start = max(end - chunk_overlap, start + step)
    return [chunk for chunk in chunks if chunk]
