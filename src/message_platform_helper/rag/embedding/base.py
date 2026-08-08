"""Embedding provider protocol."""

from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    dimensions: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        """Encode retrieval queries.

        Defaults to :meth:`embed`. Instruction-tuned models (e.g. BGE) must
        override this to prepend their query instruction; document/chunk side
        always goes through :meth:`embed` without any prefix.
        """
        return self.embed(texts)
