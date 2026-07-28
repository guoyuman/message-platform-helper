"""Markdown document parser."""

from __future__ import annotations

import re

from ..models import Document, DocumentSection, SectionKind


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+")
TABLE_RE = re.compile(r"^\s*\|.*\|\s*$")


class MarkdownParser:
    def parse(self, document: Document) -> list[DocumentSection]:
        sections: list[DocumentSection] = []
        current_title = document.title
        current_level = 0
        buffer: list[str] = []

        def flush(kind: SectionKind = "text") -> None:
            nonlocal buffer
            content = "\n".join(buffer).strip()
            if content:
                sections.append(DocumentSection(title=current_title, level=current_level, content=content, kind=kind))
            buffer = []

        for block in _blocks(document.content):
            heading = HEADING_RE.match(block.splitlines()[0] if block else "")
            if heading:
                flush()
                current_level = len(heading.group(1))
                current_title = heading.group(2).strip()
                continue
            kind = _block_kind(block)
            if buffer and kind != _block_kind("\n".join(buffer)):
                flush(kind=_block_kind("\n".join(buffer)))
            buffer.append(block)
        flush(kind=_block_kind("\n".join(buffer)))
        if not sections and document.content.strip():
            sections.append(DocumentSection(title=document.title, level=0, content=document.content.strip(), kind="text"))
        return sections


def _blocks(text: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]


def _block_kind(block: str) -> SectionKind:
    lines = [line for line in block.splitlines() if line.strip()]
    if lines and all(LIST_RE.match(line) for line in lines):
        return "list"
    if lines and all(TABLE_RE.match(line) for line in lines):
        return "table"
    return "paragraph"
