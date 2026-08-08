"""Document ingestion orchestration."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from ..chunker import ChunkStrategy, ParentChildChunkStrategy
from ..loader.base import default_loader_factory
from ..models import Chunk, Document, DocumentSection
from ..parser.base import parser_for
from ..vision import enrich_image


@dataclass(frozen=True)
class IngestionResult:
    document: Document
    chunks: list[Chunk]


_IMAGE_PATH_RE = re.compile(r"path=([^\]\s]+)")
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")
_ENRICHMENT_ENV = "MESSAGE_HELPER_IMAGE_ENRICHMENT"


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
        if _image_enrichment_enabled():
            sections = _enrich_image_sections(sections)
        chunks = self.chunk_strategy.chunk(document, sections)
        return IngestionResult(document=document, chunks=chunks)


def _image_enrichment_enabled() -> bool:
    return os.environ.get(_ENRICHMENT_ENV, "1").lower() not in {"0", "false", "no", "off"}


def _enrich_image_sections(sections: list[DocumentSection]) -> list[DocumentSection]:
    """对 image section 做 OCR + VLM 描述，并把结果拼进内容、图片路径写入元数据。

    解析不到本地图片路径（如 Markdown 远程 URL）或 enrich 无产出时，原样保留。
    """
    enriched: list[DocumentSection] = []
    for section in sections:
        if section.kind != "image":
            enriched.append(section)
            continue
        image_path = _image_path_from_section(section)
        if not image_path:
            enriched.append(section)
            continue
        info = enrich_image(image_path, section_title=section.title)
        if not info:
            # 无 OCR/VLM 后端时保留原样，但图片路径仍结构化进元数据（引用/展示可用）
            enriched.append(
                DocumentSection(
                    title=section.title,
                    level=section.level,
                    kind=section.kind,
                    content=section.content,
                    metadata={**section.metadata, "image_path": image_path},
                )
            )
            continue
        parts = [section.content]
        if info.get("caption"):
            parts.append(f"[图片描述] {info['caption']}")
        if info.get("ocr_text"):
            parts.append(f"[图中文字] {info['ocr_text']}")
        enriched.append(
            DocumentSection(
                title=section.title,
                level=section.level,
                kind=section.kind,
                content="\n".join(parts),
                metadata={**section.metadata, "image_path": image_path},
            )
        )
    return enriched


def _image_path_from_section(section: DocumentSection) -> str:
    match = _IMAGE_PATH_RE.search(section.content)
    if match:
        return match.group(1)
    # Markdown 图片：![alt](local/path.png) 且文件真实存在
    match = _MARKDOWN_IMAGE_RE.search(section.content)
    if match and Path(match.group(1)).exists():
        return match.group(1)
    return ""
