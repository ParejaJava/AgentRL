"""与检索和 Agent 框架无关的品类知识模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


def _validate_confidence(value: float) -> None:
    """保证领域中的置信度始终可以直接比较和聚合。"""

    if not 0.0 <= value <= 1.0:
        raise ValueError("confidence 必须位于 0 到 1 之间")


class CategoryCardType(str, Enum):
    """知识摄取阶段允许生成的三类品类卡片。"""

    BESTSELLER = "bestseller"
    ATTRIBUTE = "attribute"
    PRICE_RANGE = "price_range"


class PriceTierName(str, Enum):
    """跨品类统一使用的价格档位名称。"""

    ENTRY = "entry"
    MAINSTREAM = "mainstream"
    PREMIUM = "premium"


@dataclass(frozen=True, slots=True)
class Bestseller:
    """一个品类中常见的热卖款型及其选择理由。"""

    name: str
    components: tuple[str, ...] = ()
    use_cases: tuple[str, ...] = ()
    selling_points: tuple[str, ...] = ()
    typical_attributes: dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("bestseller.name 不能为空")
        _validate_confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class AttributeDistribution:
    """关键属性的主流取值、选择口径与风险提示。"""

    attribute: str
    mainstream_values: tuple[str, ...]
    selection_guide: str
    caveats: tuple[str, ...] = ()
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if not self.attribute.strip():
            raise ValueError("attribute 不能为空")
        if not self.mainstream_values:
            raise ValueError("mainstream_values 不能为空")
        if not self.selection_guide.strip():
            raise ValueError("selection_guide 不能为空")
        _validate_confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class PriceTier:
    """一个可比较的价格档位；价格为空表示原文没有可靠数值。"""

    tier: PriceTierName
    currency: str
    representative_products: tuple[str, ...]
    description: str
    min_price: float | None = None
    max_price: float | None = None
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if not self.currency.strip():
            raise ValueError("currency 不能为空")
        if not self.description.strip():
            raise ValueError("price tier description 不能为空")
        if self.min_price is not None and self.min_price < 0:
            raise ValueError("min_price 不能小于 0")
        if self.max_price is not None and self.max_price < 0:
            raise ValueError("max_price 不能小于 0")
        if (
            self.min_price is not None
            and self.max_price is not None
            and self.min_price > self.max_price
        ):
            raise ValueError("min_price 不能大于 max_price")
        _validate_confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class CategoryCard:
    """由一份 Markdown 证据一次性提炼并长期复用的知识卡片。"""

    card_id: str
    category: str
    card_type: CategoryCardType
    summary: str
    raw_evidence: tuple[str, ...]
    last_updated: str
    confidence: float
    source_document: str
    source_hash: str
    applies_to_all_categories: bool = False
    bestsellers: tuple[Bestseller, ...] = ()
    attributes: tuple[AttributeDistribution, ...] = ()
    price_tiers: tuple[PriceTier, ...] = ()
    pitfalls: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("card_id", self.card_id),
            ("category", self.category),
            ("summary", self.summary),
            ("last_updated", self.last_updated),
            ("source_document", self.source_document),
            ("source_hash", self.source_hash),
        ):
            if not value.strip():
                raise ValueError(f"{name} 不能为空")
        if not 1 <= len(self.raw_evidence) <= 3:
            raise ValueError("raw_evidence 必须包含 1 到 3 段原始证据")
        if self.card_type is CategoryCardType.BESTSELLER and not self.bestsellers:
            raise ValueError("bestseller 卡片必须包含 bestsellers")
        if self.card_type is CategoryCardType.ATTRIBUTE and not self.attributes:
            raise ValueError("attribute 卡片必须包含 attributes")
        if self.card_type is CategoryCardType.PRICE_RANGE and not self.price_tiers:
            raise ValueError("price_range 卡片必须包含 price_tiers")
        _validate_confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class CategoryInsight:
    """对 Agent 稳定公开的结构化品类洞察，不包含 RAG 原文。"""

    category: str
    components: tuple[str, ...]
    bestsellers: tuple[Bestseller, ...]
    attributes: tuple[AttributeDistribution, ...]
    price_tiers: tuple[PriceTier, ...]
    pitfalls: tuple[str, ...]
    source_card_ids: tuple[str, ...]
    last_updated: str
    confidence: float

    def __post_init__(self) -> None:
        if not self.category.strip():
            raise ValueError("category 不能为空")
        if not self.source_card_ids:
            raise ValueError("source_card_ids 不能为空")
        _validate_confidence(self.confidence)

    def to_dict(self) -> dict[str, object]:
        """输出工具可直接序列化的字典，并刻意排除 raw_evidence。"""

        return {
            "category": self.category,
            "components": list(self.components),
            "bestsellers": [
                {
                    "name": item.name,
                    "components": list(item.components),
                    "use_cases": list(item.use_cases),
                    "selling_points": list(item.selling_points),
                    "typical_attributes": dict(item.typical_attributes),
                    "confidence": item.confidence,
                }
                for item in self.bestsellers
            ],
            "attributes": [
                {
                    "attribute": item.attribute,
                    "mainstream_values": list(item.mainstream_values),
                    "selection_guide": item.selection_guide,
                    "caveats": list(item.caveats),
                    "confidence": item.confidence,
                }
                for item in self.attributes
            ],
            "price_tiers": [
                {
                    "tier": item.tier.value,
                    "min_price": item.min_price,
                    "max_price": item.max_price,
                    "currency": item.currency,
                    "representative_products": list(item.representative_products),
                    "description": item.description,
                    "confidence": item.confidence,
                }
                for item in self.price_tiers
            ],
            "pitfalls": list(self.pitfalls),
            "source_card_ids": list(self.source_card_ids),
            "last_updated": self.last_updated,
            "confidence": self.confidence,
        }
