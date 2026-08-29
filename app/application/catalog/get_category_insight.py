"""检索并聚合结构化品类知识卡片。"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Sequence
from typing import TypeVar

from app.domain.catalog import (
    AttributeDistribution,
    Bestseller,
    CategoryCard,
    CategoryCardType,
    CategoryInsight,
    PriceTier,
)

from .category_insight_config import CategoryInsightConfig
from .category_insight_models import (
    CategoryInsightRequest,
    CategoryInsightResult,
    RetrievedCategoryCard,
)
from .category_insight_ports import CategoryKnowledgeRetriever

_T = TypeVar("_T")


class CategoryInsightNotFound(LookupError):
    """知识库中没有达到最低置信度的品类卡片。"""


def _unique_strings(values: Iterable[str]) -> tuple[str, ...]:
    """去除空白和重复文本，同时保持知识卡片的原始顺序。"""

    return tuple(dict.fromkeys(value.strip() for value in values if value.strip()))


class GetCategoryInsight:
    """按 quick/deep 策略查询并确定性聚合品类常识。"""

    def __init__(
        self,
        retriever: CategoryKnowledgeRetriever,
        config: CategoryInsightConfig | None = None,
    ) -> None:
        self._retriever = retriever
        self._config = config or CategoryInsightConfig()

    def execute(self, request: CategoryInsightRequest) -> CategoryInsight:
        """返回压缩后的结构化知识，绝不把 raw_evidence 传给 Agent。"""

        return self.execute_with_metadata(request).insight

    def execute_with_metadata(
        self,
        request: CategoryInsightRequest,
    ) -> CategoryInsightResult:
        """返回领域洞察，并附带本次召回实际采用的降级层级。"""

        category = request.category.strip()
        if not category:
            raise ValueError("category 不能为空")
        if request.depth not in {"quick", "deep"}:
            raise ValueError("depth 只能是 quick 或 deep")

        limit = (
            self._config.quick_recall_k
            if request.depth == "quick"
            else self._config.deep_recall_k
        )
        retrieved = [
            item
            for item in self._retriever.search(category, limit)
            if item.card.confidence >= self._config.min_confidence
        ]
        if not retrieved:
            raise CategoryInsightNotFound(f"没有找到品类“{category}”的可靠知识")

        cards = [item.card for item in retrieved]
        bestsellers = self._collect_bestsellers(cards)
        attributes = (
            self._collect_attributes(cards) if request.depth == "deep" else ()
        )
        price_tiers = self._collect_price_tiers(cards)
        relevant_cards = [
            item for item in retrieved if not item.card.applies_to_all_categories
        ]
        canonical_category = (
            relevant_cards[0].card.category if relevant_cards else category
        )
        return CategoryInsightResult(
            insight=CategoryInsight(
                category=canonical_category,
                components=_unique_strings(
                    component
                    for item in bestsellers
                    for component in item.components
                ),
                bestsellers=bestsellers,
                attributes=attributes,
                price_tiers=price_tiers,
                pitfalls=_unique_strings(
                    pitfall for card in cards for pitfall in card.pitfalls
                ),
                source_card_ids=tuple(item.card.card_id for item in retrieved),
                last_updated=max(card.last_updated for card in cards),
                confidence=self._weighted_confidence(retrieved),
            ),
            recall_strategy=retrieved[0].recall_strategy,
        )

    def _collect_bestsellers(
        self,
        cards: Sequence[CategoryCard],
    ) -> tuple[Bestseller, ...]:
        """按款型名称去重，并限制工具返回体积。"""

        values = (
            item
            for card in cards
            if card.card_type is CategoryCardType.BESTSELLER
            for item in card.bestsellers
        )
        return self._deduplicate(values, key=lambda item: item.name)[
            : self._config.max_bestsellers
        ]

    def _collect_attributes(
        self,
        cards: Sequence[CategoryCard],
    ) -> tuple[AttributeDistribution, ...]:
        """deep 模式下按属性名去重并返回详细选择口径。"""

        values = (
            item
            for card in cards
            if card.card_type is CategoryCardType.ATTRIBUTE
            for item in card.attributes
        )
        return self._deduplicate(values, key=lambda item: item.attribute)[
            : self._config.max_attributes
        ]

    def _collect_price_tiers(
        self,
        cards: Sequence[CategoryCard],
    ) -> tuple[PriceTier, ...]:
        """保留不同代表商品的价格档，避免把不相干区间错误合并。"""

        values = (
            item
            for card in cards
            if card.card_type is CategoryCardType.PRICE_RANGE
            for item in card.price_tiers
        )
        return self._deduplicate(
            values,
            key=lambda item: (
                item.tier.value,
                item.currency,
                item.representative_products,
                item.description,
            ),
        )[: self._config.max_price_tiers]

    @staticmethod
    def _deduplicate(
        values: Iterable[_T],
        *,
        key: Callable[[_T], Hashable],
    ) -> tuple[_T, ...]:
        """根据调用方提供的业务键稳定去重。"""

        selected: dict[Hashable, _T] = {}
        for value in values:
            normalized = key(value)
            if isinstance(normalized, str):
                normalized = normalized.casefold().strip()
            selected.setdefault(normalized, value)
        return tuple(selected.values())

    @staticmethod
    def _weighted_confidence(cards: Sequence[RetrievedCategoryCard]) -> float:
        """使用检索相关度加权卡片置信度，并保留四位小数。"""

        weights = [max(item.score, 0.01) for item in cards]
        total = sum(weights)
        confidence = sum(
            item.card.confidence * weight
            for item, weight in zip(cards, weights, strict=True)
        ) / total
        return round(confidence, 4)
