"""商品检索用例的命令和结果 DTO。"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

from app.domain.catalog import Product

FloatVector: TypeAlias = Sequence[float]


@dataclass(frozen=True, slots=True)
class UserSignal:
    """当前检索域内的买家偏好向量及可解释摘要。"""

    vector: FloatVector
    summary: str = ""


@dataclass(frozen=True, slots=True)
class ItemSearchRequest:
    """搜索商品用例的输入。"""

    query: str
    index_id: str
    user_id: str | None = None
    top_k: int | None = None


@dataclass(frozen=True, slots=True)
class RecallHit:
    """向量索引返回的候选商品。"""

    item_id: str
    score: float


@dataclass(frozen=True, slots=True)
class RankedItem:
    """重排后的商品结果。"""

    rank: int
    product: Product
    retrieval_score: float
    rerank_score: float

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "retrieval_score": self.retrieval_score,
            "rerank_score": self.rerank_score,
            "product": self.product.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ItemSearchResponse:
    """搜索商品用例的稳定输出。"""

    query: str
    index_id: str
    recall_count: int
    items: Sequence[RankedItem]

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "index_id": self.index_id,
            "recall_count": self.recall_count,
            "items": [item.to_dict() for item in self.items],
        }
