"""结构化 LLM 输出与 JSONL 存储使用的 Pydantic Schema。"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.domain.catalog import (
    AttributeDistribution,
    Bestseller,
    CategoryCard,
    CategoryCardType,
    PriceTier,
    PriceTierName,
)


class ExtractedBestseller(BaseModel):
    """LLM 从原始知识文档提取的热卖款型。"""

    name: str
    components: list[str] = Field(default_factory=list)
    use_cases: list[str] = Field(default_factory=list)
    selling_points: list[str] = Field(default_factory=list)
    typical_attributes: dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractedAttributeDistribution(BaseModel):
    """LLM 提取的属性分布与选购口径。"""

    attribute: str
    mainstream_values: list[str] = Field(min_length=1)
    selection_guide: str
    caveats: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractedPriceTier(BaseModel):
    """LLM 提取的统一价格档位。"""

    tier: Literal["entry", "mainstream", "premium"]
    min_price: float | None = Field(default=None, ge=0.0)
    max_price: float | None = Field(default=None, ge=0.0)
    currency: str
    representative_products: list[str] = Field(default_factory=list)
    description: str
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractedCategoryCard(BaseModel):
    """一张尚未补充来源信息和稳定 ID 的知识卡片。"""

    category: str
    card_type: Literal["bestseller", "attribute", "price_range"]
    summary: str
    raw_evidence: list[str] = Field(min_length=1, max_length=3)
    confidence: float = Field(ge=0.0, le=1.0)
    applies_to_all_categories: bool = False
    bestsellers: list[ExtractedBestseller] = Field(default_factory=list)
    attributes: list[ExtractedAttributeDistribution] = Field(default_factory=list)
    price_tiers: list[ExtractedPriceTier] = Field(default_factory=list)
    pitfalls: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_typed_payload(self) -> ExtractedCategoryCard:
        """卡片类型必须具有对应的非空业务载荷。"""

        payloads = {
            "bestseller": self.bestsellers,
            "attribute": self.attributes,
            "price_range": self.price_tiers,
        }
        if not payloads[self.card_type]:
            raise ValueError(f"{self.card_type} 卡片缺少对应业务载荷")
        return self


class StructuredCategoryDocument(BaseModel):
    """一次结构化模型调用必须返回的完整文档结果。"""

    cards: list[ExtractedCategoryCard] = Field(min_length=1)


def extracted_card_to_domain(
    extracted: ExtractedCategoryCard,
    *,
    source_document: str,
    source_hash: str,
    last_updated: str,
) -> CategoryCard:
    """补齐稳定元数据，并转换成框架无关的领域对象。"""

    identity = (
        f"{source_document}:{extracted.category}:"
        f"{extracted.card_type}:{extracted.summary}"
    )
    card_id = f"category-{hashlib.sha256(identity.encode()).hexdigest()[:20]}"
    return CategoryCard(
        card_id=card_id,
        category=extracted.category,
        card_type=CategoryCardType(extracted.card_type),
        summary=extracted.summary,
        raw_evidence=tuple(extracted.raw_evidence),
        last_updated=last_updated,
        confidence=extracted.confidence,
        source_document=source_document,
        source_hash=source_hash,
        applies_to_all_categories=extracted.applies_to_all_categories,
        bestsellers=tuple(
            Bestseller(
                name=item.name,
                components=tuple(item.components),
                use_cases=tuple(item.use_cases),
                selling_points=tuple(item.selling_points),
                typical_attributes=dict(item.typical_attributes),
                confidence=item.confidence,
            )
            for item in extracted.bestsellers
        ),
        attributes=tuple(
            AttributeDistribution(
                attribute=item.attribute,
                mainstream_values=tuple(item.mainstream_values),
                selection_guide=item.selection_guide,
                caveats=tuple(item.caveats),
                confidence=item.confidence,
            )
            for item in extracted.attributes
        ),
        price_tiers=tuple(
            PriceTier(
                tier=PriceTierName(item.tier),
                min_price=item.min_price,
                max_price=item.max_price,
                currency=item.currency,
                representative_products=tuple(item.representative_products),
                description=item.description,
                confidence=item.confidence,
            )
            for item in extracted.price_tiers
        ),
        pitfalls=tuple(extracted.pitfalls),
    )


def category_card_to_dict(card: CategoryCard) -> dict[str, object]:
    """把领域卡片完整序列化到离线 JSONL，保留原始证据。"""

    return {
        "card_id": card.card_id,
        "category": card.category,
        "card_type": card.card_type.value,
        "summary": card.summary,
        "raw_evidence": list(card.raw_evidence),
        "last_updated": card.last_updated,
        "confidence": card.confidence,
        "source_document": card.source_document,
        "source_hash": card.source_hash,
        "applies_to_all_categories": card.applies_to_all_categories,
        "bestsellers": [
            {
                "name": item.name,
                "components": list(item.components),
                "use_cases": list(item.use_cases),
                "selling_points": list(item.selling_points),
                "typical_attributes": dict(item.typical_attributes),
                "confidence": item.confidence,
            }
            for item in card.bestsellers
        ],
        "attributes": [
            {
                "attribute": item.attribute,
                "mainstream_values": list(item.mainstream_values),
                "selection_guide": item.selection_guide,
                "caveats": list(item.caveats),
                "confidence": item.confidence,
            }
            for item in card.attributes
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
            for item in card.price_tiers
        ],
        "pitfalls": list(card.pitfalls),
    }


def category_card_from_dict(raw: dict[str, object]) -> CategoryCard:
    """校验 JSONL 数据并恢复领域卡片。"""

    extracted = ExtractedCategoryCard.model_validate(raw)
    return replace(
        extracted_card_to_domain(
            extracted,
            source_document=str(raw["source_document"]),
            source_hash=str(raw["source_hash"]),
            last_updated=str(raw["last_updated"]),
        ),
        card_id=str(raw["card_id"]),
    )
