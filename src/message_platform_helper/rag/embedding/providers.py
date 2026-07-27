"""Configurable embedding providers."""

from __future__ import annotations

import hashlib
import os
from urllib.parse import urlsplit, urlunsplit

import requests
from dataclasses import dataclass
from typing import Any

from .base import EmbeddingProvider


@dataclass
class DeterministicEmbeddingProvider:
    """Offline embedding provider for tests and local dry-runs."""

    dimensions: int = 1536

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_hash_embedding(text, self.dimensions) for text in texts]


@dataclass
class OpenAIEmbeddingProvider:
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    dimensions: int = 1536
    timeout_seconds: int = 30

    def embed(self, texts: list[str]) -> list[list[float]]:
        key = self.api_key or os.environ.get("MESSAGE_HELPER_EMBEDDING_API_KEY", "")
        if not key:
            raise RuntimeError("Aliyun embedding provider requires DASHSCOPE_API_KEY or MESSAGE_HELPER_EMBEDDING_API_KEY.")

        url = _embedding_url(self.base_url or os.environ.get("MESSAGE_HELPER_EMBEDDING_BASE_URL", ""))
        payload = _embedding_payload(
            texts,
            self.model or os.environ.get("MESSAGE_HELPER_EMBEDDING_MODEL", "text-embedding-v2"),
            url,
        )
        try:
            response = requests.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                timeout=self.timeout_seconds,
                verify=True  # 验证 SSL 证书
            )
            response.raise_for_status()  # 检查 HTTP 错误
            data = response.json()
            embeddings = []
            for item in data["output"]["embeddings"]:
                embedding = item.get("embedding", [])
                embeddings.append([float(value) for value in embedding])

            return embeddings
        except requests.exceptions.SSLError as e:
            # SSL错误处理
            # 可以尝试不验证证书（不推荐生产环境）
            # response = requests.post(url, json=payload, headers=headers, verify=False)
            raise RuntimeError(f"SSL connection failed: {e}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Request failed: {e}")


@dataclass
class BGEEmbeddingProvider:
    model: str = "BAAI/bge-small-zh-v1.5"
    dimensions: int = 1536

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("BGE embedding requires sentence-transformers to be installed.") from exc
        model = SentenceTransformer(self.model)
        vectors = model.encode(texts, normalize_embeddings=True)
        return [_fit_dimensions([float(value) for value in vector], self.dimensions) for vector in vectors]


def build_embedding_provider(config: dict[str, Any] | None = None) -> EmbeddingProvider:
    embedding = dict((config or {}).get("embedding") or config or {})
    provider = str(embedding.get("provider") or os.environ.get("MESSAGE_HELPER_EMBEDDING_PROVIDER") or "deterministic").lower()
    model = str(embedding.get("model") or os.environ.get("MESSAGE_HELPER_EMBEDDING_MODEL") or "")
    dimensions = int(embedding.get("dimensions") or os.environ.get("MESSAGE_HELPER_EMBEDDING_DIMENSIONS") or 1536)
    print(f"Using embedding provider with model {model} and dimensions {dimensions}.")
    print(f"Using embedding provider {provider}.")
    if provider == "aliyuncs":
        return OpenAIEmbeddingProvider(
            model=model or "text-embedding-v3",
            api_key=str(embedding.get("api_key") or ""),
            base_url=str(embedding.get("base_url") or ""),
            dimensions=dimensions,
        )
    if provider == "bge":
        return BGEEmbeddingProvider(model=model or "BAAI/bge-small-zh-v1.5", dimensions=dimensions)

    return DeterministicEmbeddingProvider(dimensions=dimensions)


def _hash_embedding(text: str, dimensions: int) -> list[float]:
    values: list[float] = []
    seed = text.encode("utf-8")
    counter = 0
    while len(values) < dimensions:
        digest = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        for byte in digest:
            values.append((byte / 127.5) - 1.0)
            if len(values) == dimensions:
                break
        counter += 1
    norm = sum(value * value for value in values) ** 0.5 or 1.0
    return [value / norm for value in values]


def _embedding_url(base_url: str) -> str:
    url = _normalize_aliyun_maas_url(base_url.rstrip("/"))
    if not url:
        raise RuntimeError("MESSAGE_HELPER_EMBEDDING_BASE_URL is required for Aliyun embedding provider.")
    if url.endswith("/embeddings") or "/services/embeddings/" in url:
        return url
    return f"{url}/embeddings"


def _normalize_aliyun_maas_url(url: str) -> str:
    parts = urlsplit(url)
    host = parts.netloc
    if host.startswith("cn-beijing.ws-") and host.endswith(".maas.aliyuncs.com"):
        workspace = host.removeprefix("cn-beijing.").removesuffix(".maas.aliyuncs.com")
        host = f"{workspace}.cn-beijing.maas.aliyuncs.com"
        return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))
    return url


def _embedding_payload(texts: list[str], model: str, url: str) -> dict[str, Any]:
    if "/services/embeddings/" in url:
        return {"model": model, "input": {"texts": texts}}
    return {"model": model, "input": texts}


def _fit_dimensions(vector: list[float], dimensions: int) -> list[float]:
    if len(vector) == dimensions:
        return vector
    if len(vector) > dimensions:
        return vector[:dimensions]
    return [*vector, *([0.0] * (dimensions - len(vector)))]
