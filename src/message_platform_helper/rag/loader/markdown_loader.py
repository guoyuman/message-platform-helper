"""Markdown document loader."""

from __future__ import annotations

import json
from pathlib import Path

from ..models import Document, DocumentMetadata, file_document_identity, normalize_tags, stable_document_id
from .base import DocumentLoader


class MarkdownLoader(DocumentLoader):
    format = "markdown"
    extensions = (".md", ".markdown")

    def load(self, source: str | Path, *, title: str | None = None, tags: list[str] | None = None) -> Document:
        path = Path(source)
        content = path.read_text(encoding="utf-8")
        front_matter, body = split_front_matter(content)
        document_title = title or front_matter.get("title") or first_markdown_heading(body) or path.stem.replace("_", " ")
        document_tags = tags if tags is not None else parse_document_tags(front_matter.get("tags", ""))
        metadata = DocumentMetadata(
            format="markdown",
            source=str(path),
            tags=normalize_tags(document_tags),
            extra={"front_matter": front_matter, **file_document_identity(str(path.resolve()), body)},
        )
        return Document(
            id=stable_document_id(str(path), document_title, body),
            title=document_title,
            content=body,
            metadata=metadata,
        )


def split_front_matter(text: str) -> tuple[dict[str, str], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text.strip()
    for end_index, line in enumerate(lines[1:], start=1):
        if line.strip() != "---":
            continue
        metadata: dict[str, str] = {}
        for metadata_line in lines[1:end_index]:
            key, separator, value = metadata_line.partition(":")
            if separator:
                metadata[key.strip().lower()] = value.strip()
        return metadata, "\n".join(lines[end_index + 1 :]).strip()
    return {}, text.strip()


def first_markdown_heading(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return ""


def parse_document_tags(value: str) -> list[str]:
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
