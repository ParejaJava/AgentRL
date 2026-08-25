"""商品召回与重排应用用例。"""

import math
from dataclasses import dataclass

from .config import ItemSearchConfig
from .fusion import fuse_request_vector
from .models import ItemSearchRequest, ItemSearchResponse, RankedItem
from .ports import (
    EmbeddingEncoder,
    IndexRegistry,
    NullUserSignalProvider,
    Reranker,
    UserSignalProvider,
)


@dataclass(frozen=True, slots=True)
class _Candidate:
    item_id: str
    retrieval_score: float
    rerank_score: float


class ItemSearchService:
    """在选定商品域内召回候选并执行二阶段重排。"""

    def __init__(
        self,
        *,
        encoder: EmbeddingEncoder,
        reranker: Reranker,
        indexes: IndexRegistry,
        user_signals: UserSignalProvider | None = None,
        config: ItemSearchConfig | None = None,
    ) -> None:
        self._encoder = encoder
        self._reranker = reranker
        self._indexes = indexes
        self._user_signals = user_signals or NullUserSignalProvider()
        self._config = config or ItemSearchConfig()
        if self._encoder.dimension != self._config.embedding_dimension:
            raise ValueError(
                "encoder dimension does not match ItemSearch configuration"
            )

    def search(self, request: ItemSearchRequest) -> ItemSearchResponse:
        """执行一次商品搜索用例。"""

        query = request.query.strip()
        if not query:
            raise ValueError("query cannot be empty")
        top_k = self._config.result_k if request.top_k is None else request.top_k
        if not 1 <= top_k <= self._config.max_result_k:
            raise ValueError(f"top_k must be between 1 and {self._config.max_result_k}")

        index = self._indexes.get(request.index_id)
        if index.dimension != self._encoder.dimension:
            raise ValueError("selected index dimension does not match the encoder")
        encoded = self._encoder.embed_queries([query])
        if len(encoded) != 1 or len(encoded[0]) != self._encoder.dimension:
            raise ValueError("query encoder returned an unexpected vector shape")

        user_signal = (
            self._user_signals.get(request.user_id, request.index_id)
            if request.user_id
            else None
        )
        request_vector = fuse_request_vector(
            encoded[0],
            user_signal.vector if user_signal else None,
            query_weight=self._config.query_weight,
        )
        hits = list(
            index.search(request_vector, min(self._config.recall_k, index.size))
        )
        if not hits:
            return ItemSearchResponse(
                query=query,
                index_id=request.index_id,
                recall_count=0,
                items=[],
            )

        products = [index.get_product(hit.item_id) for hit in hits]
        rerank_query = self._build_rerank_query(
            query,
            user_signal.summary if user_signal else "",
        )
        scores = list(
            self._reranker.score(
                [(rerank_query, product.to_search_text()) for product in products]
            )
        )
        if len(scores) != len(hits):
            raise ValueError("reranker score count does not match recalled candidates")
        if not all(math.isfinite(float(score)) for score in scores):
            raise ValueError("reranker returned a non-finite score")

        candidates = [
            _Candidate(hit.item_id, hit.score, float(score))
            for hit, score in zip(hits, scores, strict=True)
        ]
        candidates.sort(
            key=lambda candidate: (
                candidate.rerank_score,
                candidate.retrieval_score,
            ),
            reverse=True,
        )
        items = [
            RankedItem(
                rank=rank,
                product=index.get_product(candidate.item_id),
                retrieval_score=candidate.retrieval_score,
                rerank_score=candidate.rerank_score,
            )
            for rank, candidate in enumerate(candidates[:top_k], start=1)
        ]
        return ItemSearchResponse(
            query=query,
            index_id=request.index_id,
            recall_count=len(hits),
            items=items,
        )

    @staticmethod
    def _build_rerank_query(query: str, user_summary: str) -> str:
        """把可解释的买家偏好附加到重排查询。"""

        return (
            query
            if not user_summary
            else f"当前查询：{query}\n用户偏好：{user_summary}"
        )
