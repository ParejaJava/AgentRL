"""Business regressions exercised through real filtering and decorated tools."""

import asyncio
import json
from dataclasses import replace
from decimal import Decimal

import pytest

from app.application.catalog.config import ItemSearchConfig
from app.application.catalog.models import ItemSearchCommand
from app.application.catalog.search_catalog import ItemSearchService
from app.domain.catalog import Money, ProductSearchSpec
from app.domain.shipping import ShippingQuote
from app.infrastructure.langchain.tools.product_search import create_item_search_tool
from training.integration_smoke import FixtureRegistry, UnusedDenseBackend


def service(pricing=None):
    registry = FixtureRegistry()
    registry.index.products = {"CUP-OK": registry.index.products["CUP-OK"]}
    dense = UnusedDenseBackend()
    return ItemSearchService(
        encoder=dense,
        reranker=dense,
        indexes=registry,
        pricing=pricing,
        config=ItemSearchConfig(embedding_dimension=2, retrieval_mode="lexical"),
    ), registry


def search(runtime, **values):
    return runtime.search(
        ItemSearchCommand(
            spec=ProductSearchSpec(
                normalized_query="测试水杯", target_currency="EUR", **values
            ),
            index_id="executor-fixture",
        )
    )


def test_budget_without_currency_requests_clarification() -> None:
    runtime, _ = service()
    tool = create_item_search_tool(runtime, index_id="executor-fixture")
    result = json.loads(
        asyncio.run(
            tool.ainvoke({"normalized_query": "测试水杯", "price_max_major": 50})
        )
    )
    assert result["status"] == "needs_clarification"
    assert result["code"] == "currency_required"


def test_subtotal_and_quantity_are_not_candidate_count() -> None:
    runtime, _ = service()
    assert len(search(runtime, quantity=3, price_max_major=50).items) == 1
    result = search(runtime, quantity=3, price_max_major=50, price_basis="subtotal")
    assert not result.items
    assert result.filtered_out[0].reason == "over_price_cap"
    assert result.filtered_out[0].converted_price.major == 60
    assert search(runtime, quantity=11).filtered_out[0].reason == "insufficient_stock"


def test_brand_conflicts_are_rejected_and_exclusions_are_hard() -> None:
    with pytest.raises(ValueError, match="冲突"):
        ProductSearchSpec(
            normalized_query="杯", required_brand="ACME", excluded_brands=("acme",)
        )
    runtime, registry = service()
    registry.index.products["CUP-OK"] = replace(
        registry.index.products["CUP-OK"], brand="ACME"
    )
    assert not search(runtime, excluded_brands=("acme",)).items
    assert not search(runtime, required_brand="Other").items
    assert len(search(runtime, required_brand="acme").items) == 1


def test_landed_budget_requires_destination_and_price_evidence() -> None:
    with pytest.raises(ValueError, match="配送"):
        ProductSearchSpec(
            normalized_query="杯", price_basis="landed", price_max_major=50
        )
    runtime, _ = service()
    result = search(runtime, ship_to="DE", price_basis="landed", price_max_major=50)
    assert not result.items
    assert result.filtered_out[0].reason == "pricing_unavailable"


def test_landed_budget_uses_the_same_quote_as_returned_card() -> None:
    class Pricing:
        calls = 0

        def supported_destinations(self):
            return ("DE",)

        def convert(self, money, target_currency):
            assert money.currency == target_currency
            return money

        def quote(self, *, unit_price, category, ship_to, quantity, target_currency):
            self.calls += 1
            return ShippingQuote(
                ship_to,
                quantity,
                Money(unit_price.amount_minor * quantity, target_currency),
                Money(1500, target_currency),
                Money(0, target_currency),
                Decimal(0),
                False,
                "fixture",
                "fixture",
                "2026-09-10",
            )

    pricing = Pricing()
    runtime, _ = service(pricing)
    result = search(
        runtime, quantity=2, ship_to="DE", price_basis="landed", price_max_major=60
    )
    assert len(result.items) == 1
    assert result.items[0].landed_price.landed_total.major == 55
    assert pricing.calls == 1
    assert not search(
        runtime, quantity=2, ship_to="DE", price_basis="landed", price_max_major=45
    ).items
