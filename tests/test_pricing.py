"""验证金额、汇率、运费和关税规则的精确边界。"""

from decimal import Decimal

import pytest

from app.domain.catalog import Money, ProductSearchSpec
from app.domain.pricing import ExchangeRateSnapshot
from app.infrastructure.pricing import create_static_pricing_provider


def test_money_uses_currency_specific_minor_units_and_half_up_rounding() -> None:
    """CNY 使用分、JPY 使用整数日元，且金额不会受二进制浮点误差影响。"""

    cny = Money.from_major_units("12.345", "cny")
    jpy = Money.from_major_units("1000.4", "jpy")

    assert cny.amount_minor == 1235
    assert cny.to_major_units() == Decimal("12.35")
    assert jpy.amount_minor == 1000
    assert jpy.to_major_units() == Decimal(1000)
    assert Money.from_major_units(10, "CNY").multiply(3).major == 30


def test_exchange_rate_converts_via_major_units_for_jpy() -> None:
    """币种精度不同时不能直接拿两边最小单位相乘。"""

    snapshot = ExchangeRateSnapshot(
        rates_to_cny={"CNY": Decimal(1), "JPY": Decimal("0.048")},
        version="test-v1",
        as_of="2026-08-01",
        source="test",
    )

    converted = snapshot.convert(Money.from_major_units(1000, "JPY"), "CNY")

    assert converted.amount_minor == 4800
    assert converted.major == 48


def test_tariff_quote_handles_threshold_and_quantity() -> None:
    """商品小计和续件运费都随数量变化，关税只计算免税额以上部分。"""

    pricing = create_static_pricing_provider()

    exempt = pricing.quote(
        unit_price=Money.from_major_units(100, "USD"),
        category="旅行装备",
        ship_to="CN",
        quantity=1,
        target_currency="CNY",
    )
    taxed = pricing.quote(
        unit_price=Money.from_major_units(1000, "USD"),
        category="旅行装备",
        ship_to="CN",
        quantity=2,
        target_currency="CNY",
    )

    assert exempt.subtotal.major == 710
    assert exempt.freight.major == 25
    assert exempt.tariff.major == 0
    assert exempt.landed_total.major == 735
    assert exempt.de_minimis_applied is True
    assert taxed.subtotal.major == 14200
    assert taxed.freight.major == 40
    assert taxed.tariff.major == 828
    assert taxed.landed_total.major == 15068
    assert taxed.de_minimis_applied is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"normalized_query": "  "}, "normalized_query"),
        ({"normalized_query": "商品", "top_k": 0}, "top_k"),
        ({"normalized_query": "商品", "ship_to": "CHINA"}, "ship_to"),
        ({"normalized_query": "商品", "target_currency": "ABC"}, "币种"),
        ({"normalized_query": "商品", "price_max_major": -1}, "price_max"),
        ({"normalized_query": "商品", "price_max_major": float("nan")}, "price_max"),
        ({"normalized_query": "商品", "quantity": 0}, "quantity"),
    ],
)
def test_product_search_spec_rejects_invalid_business_parameters(
    kwargs: dict[str, object],
    message: str,
) -> None:
    """业务值对象必须独立于 LangChain/Pydantic 维护最终不变量。"""

    with pytest.raises((TypeError, ValueError), match=message):
        ProductSearchSpec(**kwargs)
