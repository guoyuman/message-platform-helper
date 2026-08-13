"""Recursive section-aware chunking."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence

from ..metadata import build_chunk_metadata
from ..models import Chunk, Document, DocumentSection, stable_chunk_id
from .base import ChunkStrategy

EmbeddingFn = Callable[[list[str]], list[list[float]]]
TextSizer = Callable[[str], int]


class RecursiveChunkStrategy(ChunkStrategy):
    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 120,
        *,
        target_chunk_size: int | None = None,
        max_chunk_size: int | None = None,
        max_chunk_chars: int | None = None,
        semantic_threshold: float = 0.35,
        embedding_fn: EmbeddingFn | None = None,
        text_sizer: TextSizer | None = None,
    ) -> None:
        self.target_chunk_size = int(target_chunk_size or chunk_size)
        self.max_chunk_size = int(max_chunk_size or max_chunk_chars or chunk_size)
        self.max_chunk_chars = self.max_chunk_size
        self.chunk_overlap = int(chunk_overlap)
        self.semantic_merger = SemanticMerger(
            threshold=semantic_threshold,
            embedding_fn=embedding_fn,
            text_sizer=text_sizer,
        )
        self.text_sizer = text_sizer or get_text_size

    def chunk(self, document: Document, sections: list[DocumentSection]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section in sections or [DocumentSection(title=document.title, level=0, content=document.content)]:
            parts = split_section(
                section.content,
                self.target_chunk_size,
                self.chunk_overlap,
                max_chunk_size=self.max_chunk_size,
                section_kind=section.kind,
                semantic_merger=self.semantic_merger,
                text_sizer=self.text_sizer,
            )
            for part_index, part in enumerate(parts):
                global_index = len(chunks)
                metadata = build_chunk_metadata(document, section, chunk_index=part_index, chunk_count=len(parts))
                metadata["split_reason"] = "semantic" if len(parts) > 1 else "section"
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


class SemanticMerger:
    def __init__(
        self,
        *,
        threshold: float = 0.35,
        embedding_fn: EmbeddingFn | None = None,
        text_sizer: TextSizer | None = None,
    ) -> None:
        self.threshold = threshold
        self.embedding_fn = embedding_fn
        self.text_sizer = text_sizer or get_text_size

    def merge(self, paragraphs: list[str], *, max_size: int) -> list[str]:
        if len(paragraphs) <= 1:
            return paragraphs
        similarities = self._similarities(paragraphs)
        blocks: list[str] = []
        current = [paragraphs[0]]
        for index, paragraph in enumerate(paragraphs[1:], start=1):
            candidate = "\n\n".join([*current, paragraph])
            if similarities[index - 1] >= self.threshold and self.text_sizer(candidate) <= max_size:
                current.append(paragraph)
                continue
            blocks.append("\n\n".join(current))
            current = [paragraph]
        blocks.append("\n\n".join(current))
        return blocks

    def _similarities(self, paragraphs: list[str]) -> list[float]:
        if self.embedding_fn is not None:
            vectors = self.embedding_fn(paragraphs)
            return [_cosine(vectors[index - 1], vectors[index]) for index in range(1, len(vectors))]
        return [_heuristic_similarity(paragraphs[index - 1], paragraphs[index]) for index in range(1, len(paragraphs))]


def split_section(
    text: str,
    target_chunk_size: int = 600,
    chunk_overlap: int = 50,
    *,
    max_chunk_size: int | None = None,
    max_chunk_chars: int | None = None,
    section_kind: str = "text",
    semantic_merger: SemanticMerger | None = None,
    text_sizer: TextSizer | None = None,
) -> list[str]:
    text_sizer = text_sizer or get_text_size
    max_size = int(max_chunk_size or max_chunk_chars or target_chunk_size)
    normalized = text.strip()
    if not normalized:
        return []
    if section_kind == "image":
        return [normalized]
    if section_kind == "table":
        return _split_table(normalized, max_size, text_sizer=text_sizer)

    chunks: list[str] = []
    merger = semantic_merger or SemanticMerger(text_sizer=text_sizer)
    for block in merger.merge(_candidate_paragraphs(normalized, section_kind), max_size=max_size):
        chunks.extend(_split_long_semantic_block(block, max_size, text_sizer=text_sizer))
    return _apply_overlap(chunks, int(chunk_overlap), max_size, text_sizer=text_sizer)


def _candidate_paragraphs(text: str, section_kind: str) -> list[str]:
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


def _split_table(text: str, chunk_size: int, *, text_sizer: TextSizer) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) <= 1:
        return _split_long_semantic_block(text, chunk_size, text_sizer=text_sizer)
    header = lines[0]
    if text_sizer(header) >= chunk_size:
        return _split_long_semantic_block(text, chunk_size, text_sizer=text_sizer)
    chunks: list[str] = []
    current: list[str] = [header]
    for row in lines[1:]:
        candidate = "\n".join([*current, row])
        if text_sizer(candidate) <= chunk_size:
            current.append(row)
            continue
        if len(current) > 1:
            chunks.append("\n".join(current))
        row_candidate = "\n".join([header, row])
        if text_sizer(row_candidate) <= chunk_size:
            current = [header, row]
            continue
        budget = max(chunk_size - text_sizer(header) - 1, 1)
        chunks.extend(f"{header}\n{part}" for part in _split_long_semantic_block(row, budget, text_sizer=text_sizer))
        current = [header]
    if len(current) > 1:
        chunks.append("\n".join(current))
    return chunks


def _split_long_semantic_block(text: str, chunk_size: int, *, text_sizer: TextSizer | None = None) -> list[str]:
    text_sizer = text_sizer or get_text_size
    if text_sizer(text) <= chunk_size:
        return [text.strip()]
    return _recursive_split(text, chunk_size, level=0, text_sizer=text_sizer)


def _recursive_split(text: str, chunk_size: int, *, level: int, text_sizer: TextSizer) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []
    if text_sizer(normalized) <= chunk_size:
        return [normalized]
    if level >= len(_SEPARATORS):
        return [normalized[index : index + chunk_size] for index in range(0, len(normalized), chunk_size)]

    units = _split_by_separator(normalized, _SEPARATORS[level])
    if len(units) <= 1:
        return _recursive_split(normalized, chunk_size, level=level + 1, text_sizer=text_sizer)

    pieces: list[str] = []
    for unit in units:
        if text_sizer(unit) <= chunk_size:
            pieces.append(unit)
        else:
            pieces.extend(_recursive_split(unit, chunk_size, level=level + 1, text_sizer=text_sizer))
    return _pack_units(pieces, chunk_size, text_sizer=text_sizer)


def _pack_units(units: Sequence[str], chunk_size: int, *, text_sizer: TextSizer) -> list[str]:
    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}{unit}" if current else unit
        if text_sizer(candidate.strip()) <= chunk_size:
            current = candidate
            continue
        chunks.append(current.strip())
        current = unit
    if current:
        chunks.append(current.strip())
    return [chunk for chunk in chunks if chunk]


_SEPARATORS = ("sentence", "clause", "comma", "whitespace")


def _split_by_separator(text: str, separator: str) -> list[str]:
    if separator == "sentence":
        return _regex_units(text, r".+?(?:[。！？.!?]+|$)")
    if separator == "clause":
        return _regex_units(text, r".+?(?:[；;]+|$)")
    if separator == "comma":
        return _regex_units(text, r".+?(?:[，,、]+|$)")
    if separator == "whitespace":
        return re.findall(r"\S+\s*", text, flags=re.S)
    return [text]


def _regex_units(text: str, pattern: str) -> list[str]:
    return [match for match in re.findall(pattern, text, flags=re.S) if match.strip()]


def _semantic_units(text: str) -> list[str]:
    return [unit.strip() for unit in _split_by_separator(text, "sentence") if unit.strip()]


def _apply_overlap(chunks: list[str], chunk_overlap: int, max_size: int, *, text_sizer: TextSizer) -> list[str]:
    if chunk_overlap <= 0 or len(chunks) <= 1:
        return chunks
    overlapped = [chunks[0]]
    for index, chunk in enumerate(chunks[1:], start=1):
        budget = min(chunk_overlap, max_size - text_sizer(chunk) - 1)
        overlap = _overlap_text(chunks[index - 1], budget, text_sizer=text_sizer) if budget > 0 else ""
        candidate = f"{overlap}\n{chunk}" if overlap and not chunk.startswith(overlap) else chunk
        overlapped.append(candidate if text_sizer(candidate) <= max_size else chunk)
    return overlapped


def _overlap_text(text: str, budget: int, *, text_sizer: TextSizer) -> str:
    if budget <= 0:
        return ""
    for separator in ("sentence", "clause", "comma", "whitespace"):
        for unit in reversed(_split_by_separator(text, separator)):
            normalized = unit.strip()
            if normalized and text_sizer(normalized) <= budget:
                return normalized
    return text.strip()[-budget:]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _heuristic_similarity(left: str, right: str) -> float:
    left_tokens = set(_tokens(left))
    right_tokens = set(_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9_.#/-]+|[\u4e00-\u9fff]", text.lower())
    return [token for token in tokens if token.strip()]


def get_text_size(text: str) -> int:
    return len(text)
