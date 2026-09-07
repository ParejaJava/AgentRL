"""按模型版本和文本内容隔离的 Embedding 缓存。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from app.application.catalog.models import FloatVector
from app.application.catalog.ports import EmbeddingEncoder


class VectorCache(Protocol):
    """同步检索编码器所需的最小向量缓存端口。"""

    def get(self, key: str) -> FloatVector | None: ...

    def set(self, key: str, vector: FloatVector) -> None: ...


class RedisVectorCache:
    """以 JSON 数组在 Redis 中缓存归一化向量。"""

    def __init__(self, redis_url: str, *, prefix: str, ttl_seconds: int) -> None:
        try:
            from redis import Redis
        except ImportError as exc:
            raise RuntimeError(
                "已启用 Redis Embedding 缓存；请运行 `uv sync --extra platform`"
            ) from exc
        self._redis = Redis.from_url(redis_url, decode_responses=True)
        self._prefix = prefix
        self._ttl = ttl_seconds

    def get(self, key: str) -> FloatVector | None:
        raw = self._redis.get(f"{self._prefix}:{key}")
        if raw is None:
            return None
        return np.asarray(json.loads(raw), dtype=np.float32)

    def set(self, key: str, vector: FloatVector) -> None:
        self._redis.setex(
            f"{self._prefix}:{key}",
            self._ttl,
            json.dumps(np.asarray(vector, dtype=np.float32).tolist()),
        )


class CachedEmbeddingEncoder:
    """只缓存确定性的 query/document 编码，不改变原编码器接口。"""

    def __init__(
        self,
        encoder: EmbeddingEncoder,
        cache: VectorCache,
        *,
        model_revision: str,
    ) -> None:
        self._encoder = encoder
        self._cache = cache
        self._revision = model_revision

    @property
    def dimension(self) -> int:
        return self._encoder.dimension

    def embed_queries(self, texts: Sequence[str]) -> FloatVector:
        return self._embed(texts, kind="query")

    def embed_documents(self, texts: Sequence[str]) -> FloatVector:
        return self._embed(texts, kind="document")

    def _embed(self, texts: Sequence[str], *, kind: str) -> FloatVector:
        vectors: list[FloatVector | None] = []
        misses: list[str] = []
        miss_positions: list[int] = []
        keys: list[str] = []
        for position, text in enumerate(texts):
            key = hashlib.sha256(
                f"{self._revision}\0{kind}\0{text}".encode()
            ).hexdigest()
            keys.append(key)
            cached = self._cache.get(key)
            vectors.append(cached)
            if cached is None:
                misses.append(text)
                miss_positions.append(position)
        if misses:
            encoded = (
                self._encoder.embed_queries(misses)
                if kind == "query"
                else self._encoder.embed_documents(misses)
            )
            for position, vector in zip(miss_positions, encoded, strict=True):
                vectors[position] = vector
                self._cache.set(keys[position], vector)
        return np.asarray(vectors, dtype=np.float32)
