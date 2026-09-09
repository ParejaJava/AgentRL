"""商品检索用例的命令和结果 DTO。"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

from app.domain.catalog import Money, Product, ProductSearchSpec
from app.domain.shipping import ShippingQuote

FloatVector: TypeAlias = Sequence[float]
RecallStrategy: TypeAlias = Literal[
    "embedding_rerank",
    "embedding_only",
    "keyword_2gram",
    "bm25",
    "knn",
    "hybrid_rrf",
]


@dataclass(frozen=True, slots=True)
class UserSignal:
    """当前检索域内的买家偏好向量及可解释摘要。"""

    vector: FloatVector
    summary: str = ""


@dataclass(frozen=True, slots=True)
class ItemSearchCommand:
    """把纯业务搜索规格与索引、买家等运行路由组合起来。"""

    spec: ProductSearchSpec
    index_id: str
    buyer_id: str | None = None

    def __post_init__(self) -> None:
        """应用命令必须显式指定检索域，买家标识可选。"""

        if not self.index_id.strip():
            raise ValueError("index_id 不能为空")
        if self.buyer_id is not None and not self.buyer_id.strip():
            raise ValueError("buyer_id 不能为空字符串")


# 兼容已有导入名；新代码和文档使用 ItemSearchCommand。
ItemSearchRequest: TypeAlias = ItemSearchCommand


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
    landed_price: ShippingQuote | None = None
    pricing_unavailable_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        """输出商品卡，并把可选到手价直接内联在商品对象中。"""

        product_card = self.product.to_dict()
        try:
            product_card["primary_price"] = self.product.primary_sku().price.to_dict()
        except ValueError:
            product_card["primary_price"] = None
        if self.landed_price is not None:
            product_card["landed_price"] = self.landed_price.to_dict()
        elif self.pricing_unavailable_reason is not None:
            product_card["landed_price"] = {
                "status": "unavailable",
                "reason": self.pricing_unavailable_reason,
            }
        return {
            "rank": self.rank,
            "retrieval_score": self.retrieval_score,
            "rerank_score": self.rerank_score,
            "product": product_card,
        }


@dataclass(frozen=True, slots=True)
class FilteredItem:
    """被硬约束拦截的候选摘要，帮助 Agent 区分未召回和不满足条件。"""

    item_id: str
    title: str
    category: str
    reason: str
    converted_price: Money | None = None

    def to_dict(self) -> dict[str, object]:
        """只返回必要摘要，避免过滤轨迹膨胀上下文。"""

        result: dict[str, object] = {
            "item_id": self.item_id,
            "title": self.title,
            "category": self.category,
            "reason": self.reason,
        }
        if self.converted_price is not None:
            result["converted_price"] = self.converted_price.to_dict()
        return result


@dataclass(frozen=True, slots=True)
class ItemSearchResponse:
    """搜索商品用例的稳定输出。"""

    query: str
    index_id: str
    recall_count: int
    items: Sequence[RankedItem]
    recall_strategy: RecallStrategy = "embedding_rerank"
    filtered_out: Sequence[FilteredItem] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "index_id": self.index_id,
            "recall_count": self.recall_count,
            "recall_strategy": self.recall_strategy,
            "rerank_applied": self.recall_strategy == "embedding_rerank",
            "items": [item.to_dict() for item in self.items],
            "filtered_out": [item.to_dict() for item in self.filtered_out],
        }
