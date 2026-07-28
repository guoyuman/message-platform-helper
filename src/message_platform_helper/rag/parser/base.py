"""Parser protocol and factory helpers."""

from __future__ import annotations

from typing import Protocol

from ..models import Document, DocumentSection


class DocumentParser(Protocol):
    def parse(self, document: Document) -> list[DocumentSection]:
        raise NotImplementedError


def parser_for(document: Document) -> DocumentParser:
    from .docx_parser import DocxParser
    from .markdown_parser import MarkdownParser
    from .pdf_parser import PdfParser

    parsers: dict[str, DocumentParser] = {
        "markdown": MarkdownParser(),
        "pdf": PdfParser(),
        "docx": DocxParser(),
    }
    return parsers.get(document.metadata.format, PdfParser())

