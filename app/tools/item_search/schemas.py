"""Typed inputs and outputs for ItemSearch."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

JsonScalar: TypeAlias = str | int | float | bool | None
FloatVector: TypeAlias = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class Product:
    """A product record stored next to its vector in a scoped item index."""

    item_id: str
    title: str
    category: str = ""
    brand: str = ""
    description: str = ""
    platform: str = ""
    attributes: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, JsonScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.item_id.strip():
            raise ValueError("item_id cannot be empty")
        if not self.title.strip():
            raise ValueError("title cannot be empty")

    def to_search_text(self) -> str:
        """Build the stable semantic text used by the item tower and reranker."""

        fields = [
            ("商品名", self.title),
            ("类目", self.category),
            ("品牌", self.brand),
            ("平台", self.platform),
            ("描述", self.description),
        ]
        lines = [f"{label}: {value}" for label, value in fields if value]
        if self.attributes:
            attributes = "；".join(
                f"{key}={value}" for key, value in sorted(self.attributes.items())
            )
            lines.append(f"属性: {attributes}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "title": self.title,
            "category": self.category,
            "brand": self.brand,
            "description": self.description,
            "platform": self.platform,
            "attributes": dict(self.attributes),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class UserSignal:
    """A user-tower vector and optional text summary for reranking."""

    vector: FloatVector
    summary: str = ""


@dataclass(frozen=True, slots=True)
class ItemSearchRequest:
    """Search request after a child Agent has selected an index domain."""

    query: str
    index_id: str
    user_id: str | None = None
    top_k: int | None = None


@dataclass(frozen=True, slots=True)
class RecallHit:
    """One item returned from ANN retrieval."""

    item_id: str
    score: float


@dataclass(frozen=True, slots=True)
class RankedItem:
    """One reranked item returned to the calling Agent."""

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
    """Structured response from recall and reranking."""

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

