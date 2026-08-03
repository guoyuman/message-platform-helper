"""PDF text parser."""

from __future__ import annotations

import re

from ..models import Document, DocumentSection, SectionKind


PAGE_RE = re.compile(r"^\[Page\s+(\d+)\]$")
TABLE_RE = re.compile(r"^\[Table\s+([\d.]+)\]$")
IMAGE_RE = re.compile(r"^\[Image\s+([\d.]+):.+?\]$")


class PdfParser:
    def parse(self, document: Document) -> list[DocumentSection]:
        sections: list[DocumentSection] = []
        current_title = document.title
        current_page = 0
        for block in _paragraphs(document.content):
            lines = block.splitlines()
            page = PAGE_RE.match(lines[0] if lines else "")
            if page and len(lines) == 1:
                current_page = int(page.group(1))
                current_title = f"{document.title} Page {current_page}"
                continue
            if page:
                current_page = int(page.group(1))
                current_title = f"{document.title} Page {current_page}"
                block = "\n".join(lines[1:]).strip()
                if not block:
                    continue
            table = TABLE_RE.match(block.splitlines()[0] if block else "")
            image = IMAGE_RE.match(block)
            kind: SectionKind = "paragraph"
            title = current_title
            metadata = {"page": current_page} if current_page else {}
            if table:
                kind = "table"
                title = f"{current_title} Table {table.group(1)}"
                metadata["table"] = table.group(1)
            elif image:
                kind = "image"
                title = f"{current_title} Image {image.group(1)}"
                metadata["image"] = image.group(1)
            sections.append(DocumentSection(title=title, level=1 if current_page else 0, content=block, kind=kind, metadata=metadata))
        if not sections and document.content.strip():
            sections.append(DocumentSection(title=document.title, level=0, content=document.content.strip(), kind="text"))
        return sections


def _paragraphs(text: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
