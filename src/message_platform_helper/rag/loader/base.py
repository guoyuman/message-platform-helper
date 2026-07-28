"""Base document loader abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Type

from ..models import Document


class UnsupportedDocumentFormat(ValueError):
    pass


class DocumentLoader(ABC):
    format: str = "unknown"
    extensions: tuple[str, ...] = ()

    @abstractmethod
    def load(self, source: str | Path, *, title: str | None = None, tags: list[str] | None = None) -> Document:
        raise NotImplementedError


class LoaderFactory:
    def __init__(self) -> None:
        self._loaders: dict[str, Type[DocumentLoader]] = {}

    def register(self, loader_cls: Type[DocumentLoader]) -> None:
        for extension in loader_cls.extensions:
            self._loaders[extension.lower()] = loader_cls

    def create(self, source: str | Path) -> DocumentLoader:
        path = Path(source)
        suffix = path.suffix.lower()
        loader_cls = self._loaders.get(suffix)
        if not loader_cls:
            raise UnsupportedDocumentFormat(f"Unsupported document format: {suffix or '<none>'}")
        return loader_cls()


def default_loader_factory() -> LoaderFactory:
    from .docx_loader import DocxLoader
    from .markdown_loader import MarkdownLoader
    from .pdf_loader import PdfLoader
    from .text_loader import TextLoader

    factory = LoaderFactory()
    for loader_cls in (TextLoader, MarkdownLoader, PdfLoader, DocxLoader):
        factory.register(loader_cls)
    return factory

