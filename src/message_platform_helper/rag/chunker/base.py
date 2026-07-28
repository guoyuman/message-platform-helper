"""Base chunk strategy."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Chunk, Document, DocumentSection


class ChunkStrategy(ABC):
    @abstractmethod
    def chunk(self, document: Document, sections: list[DocumentSection]) -> list[Chunk]:
        raise NotImplementedError

