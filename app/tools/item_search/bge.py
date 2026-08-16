"""Lazy BGE embedding and reranker adapters.

Importing this module never loads or downloads a model. Model initialization only
happens on the first encode or rerank call.
"""

from collections.abc import Sequence
from threading import Lock
from typing import Any

import numpy as np

from .fusion import normalize_matrix
from .schemas import FloatVector


class BGEEmbeddingEncoder:
    """Dense BGE-M3 adapter backed by FlagEmbedding."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        *,
        dimension: int = 1024,
        batch_size: int = 16,
        max_length: int = 512,
        use_fp16: bool = False,
    ) -> None:
        self.model_name = model_name
        self._dimension = dimension
        self._batch_size = batch_size
        self._max_length = max_length
        self._use_fp16 = use_fp16
        self._model: Any | None = None
        self._load_lock = Lock()

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model

        with self._load_lock:
            if self._model is None:
                try:
                    from FlagEmbedding import BGEM3FlagModel
                except ImportError as exc:
                    raise RuntimeError(
                        "BGE support is not installed; run `uv sync --extra search`"
                    ) from exc
                self._model = BGEM3FlagModel(
                    self.model_name,
                    use_fp16=self._use_fp16,
                )
        return self._model

    def _encode(self, texts: Sequence[str]) -> FloatVector:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

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
    """BGE cross-encoder adapter that returns normalized relevance scores."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        *,
        use_fp16: bool = False,
    ) -> None:
        self.model_name = model_name
        self._use_fp16 = use_fp16
        self._model: Any | None = None
        self._load_lock = Lock()

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model

        with self._load_lock:
            if self._model is None:
                try:
                    from FlagEmbedding import FlagReranker
                except ImportError as exc:
                    raise RuntimeError(
                        "BGE support is not installed; run `uv sync --extra search`"
                    ) from exc
                self._model = FlagReranker(
                    self.model_name,
                    use_fp16=self._use_fp16,
                )
        return self._model

    def score(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
        if not pairs:
            return []
        scores = self._get_model().compute_score(
            [list(pair) for pair in pairs],
            normalize=True,
        )
        return np.atleast_1d(scores).astype(float).tolist()

