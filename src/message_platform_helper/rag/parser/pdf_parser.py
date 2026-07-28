"""PDF text parser."""

from __future__ import annotations

import re

from ..models import Document, DocumentSection


class PdfParser:
    def parse(self, document: Document) -> list[DocumentSection]:
        sections = [
            DocumentSection(title=document.title, level=0, content=block, kind="paragraph")
            for block in _paragraphs(document.content)
        ]
        if not sections and document.content.strip():
            sections.append(DocumentSection(title=document.title, level=0, content=document.content.strip(), kind="text"))
        return sections


def _paragraphs(text: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]

