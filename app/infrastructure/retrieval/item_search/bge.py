"""BGE embedding 与 reranker 的惰性加载适配器。"""

from collections.abc import Sequence
from threading import Lock
from typing import Any

import numpy as np

from app.application.catalog.models import FloatVector

from .fusion import normalize_matrix

_MODEL_CACHE: dict[tuple[object, ...], Any] = {}
_MODEL_CACHE_LOCK = Lock()
# Transformers fast tokenizer 与单 GPU 模型实例都不保证跨线程可重入。
_GLOBAL_INFERENCE_LOCK = Lock()


class BGEEmbeddingEncoder:
    """基于 FlagEmbedding 的 BGE-M3 稠密向量编码器。"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        *,
        dimension: int = 1024,
        batch_size: int = 16,
        max_length: int = 512,
        use_fp16: bool = False,
        device: str = "auto",
    ) -> None:
        if batch_size < 1:
            raise ValueError("embedding batch_size 必须大于 0")
        self.model_name = model_name
        self._dimension = dimension
        self._batch_size = batch_size
        self._max_length = max_length
        self._use_fp16 = use_fp16
        self._device = device
        self._model: Any | None = None

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_model(self) -> Any:
        """首次推理时加载模型，并把显存相关参数传给 FlagEmbedding。"""

        if self._model is not None:
            return self._model

        cache_key = (
            "embedding",
            self.model_name,
            self._use_fp16,
            self._device,
            self._batch_size,
            self._max_length,
        )
        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(cache_key)
            if cached is None:
                try:
                    from FlagEmbedding import BGEM3FlagModel
                except ImportError as exc:
                    raise RuntimeError(
                        "BGE support is not installed; run `uv sync --extra search`"
                    ) from exc
                cached = BGEM3FlagModel(
                    self.model_name,
                    use_fp16=self._use_fp16,
                    devices=None if self._device == "auto" else self._device,
                    batch_size=self._batch_size,
                    query_max_length=self._max_length,
                    passage_max_length=self._max_length,
                )
                _MODEL_CACHE[cache_key] = cached
            self._model = cached
        return self._model

    def _encode(self, texts: Sequence[str]) -> FloatVector:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        with _GLOBAL_INFERENCE_LOCK:
            output = self._get_model().encode(
                list(texts),
                batch_size=self._batch_size,
                max_length=self._max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
        vectors = normalize_matrix(np.asarray(output["dense_vecs"], dtype=np.float32))
        if vectors.shape != (len(texts), self.dimension):
            raise ValueError(
                "BGE output shape does not match the configured embedding dimension"
            )
        return vectors

    def embed_queries(self, texts: Sequence[str]) -> FloatVector:
        return self._encode(texts)

    def embed_documents(self, texts: Sequence[str]) -> FloatVector:
        return self._encode(texts)


class BGEReranker:
    """使用 BGE cross-encoder 返回归一化相关性分数。"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        *,
        use_fp16: bool = False,
        device: str = "auto",
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        if batch_size < 1:
            raise ValueError("reranker batch_size 必须大于 0")
        self.model_name = model_name
        self._use_fp16 = use_fp16
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._model: Any | None = None

    def _get_model(self) -> Any:
        """首次重排时加载模型，避免应用启动阶段占用显存。"""

        if self._model is not None:
            return self._model

        cache_key = (
            "reranker",
            self.model_name,
            self._use_fp16,
            self._device,
            self._batch_size,
            self._max_length,
        )
        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(cache_key)
            if cached is None:
                try:
                    from FlagEmbedding import FlagReranker
                except ImportError as exc:
                    raise RuntimeError(
                        "BGE support is not installed; run `uv sync --extra search`"
                    ) from exc
                cached = FlagReranker(
                    self.model_name,
                    use_fp16=self._use_fp16,
                    devices=None if self._device == "auto" else self._device,
                    batch_size=self._batch_size,
                    max_length=self._max_length,
                )
                _MODEL_CACHE[cache_key] = cached
            self._model = cached
        return self._model

    def score(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
        if not pairs:
            return []
        with _GLOBAL_INFERENCE_LOCK:
            scores = self._get_model().compute_score(
                [list(pair) for pair in pairs],
                batch_size=self._batch_size,
                max_length=self._max_length,
                normalize=True,
            )
        return np.atleast_1d(scores).astype(float).tolist()
