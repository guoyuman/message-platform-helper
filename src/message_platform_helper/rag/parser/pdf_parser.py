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
        current_page = 0
        paragraph_blocks: list[str] = []

        def flush_paragraphs() -> None:
            if not paragraph_blocks:
                return
            metadata = {"page": current_page} if current_page else {}
            sections.append(
                DocumentSection(
                    title=document.title,
                    level=1 if current_page else 0,
                    content="\n\n".join(paragraph_blocks),
                    kind="paragraph",
                    metadata=metadata,
                )
            )
            paragraph_blocks.clear()

        for block in _paragraphs(document.content):
            lines = block.splitlines()
            page = PAGE_RE.match(lines[0] if lines else "")
            if page and len(lines) == 1:
                flush_paragraphs()
                current_page = int(page.group(1))
                continue
            if page:
                flush_paragraphs()
                current_page = int(page.group(1))
                block = "\n".join(lines[1:]).strip()
                if not block:
                    continue
            table = TABLE_RE.match(block.splitlines()[0] if block else "")
            image = IMAGE_RE.match(block)
            metadata = {"page": current_page} if current_page else {}
            if table:
                flush_paragraphs()
                metadata["table"] = table.group(1)
                sections.append(
                    DocumentSection(
                        title=f"{document.title} Table {table.group(1)}",
                        level=1 if current_page else 0,
                        content=block,
                        kind="table",
                        metadata=metadata,
                    )
                )
            elif image:
                flush_paragraphs()
                metadata["image"] = image.group(1)
                sections.append(
                    DocumentSection(
                        title=f"{document.title} Image {image.group(1)}",
                        level=1 if current_page else 0,
                        content=block,
                        kind="image",
                        metadata=metadata,
                    )
                )
            else:
                paragraph_blocks.append(block)
        flush_paragraphs()
        if not sections and document.content.strip():
            sections.append(DocumentSection(title=document.title, level=0, content=document.content.strip(), kind="text"))
        return sections


def _paragraphs(text: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
