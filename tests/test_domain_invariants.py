"""覆盖金额、检索规格、关税规则和订单意向的领域边界。"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from app.domain.catalog import Money, ProductSearchSpec
from app.domain.order import Address, Order, OrderLine, OrderStatus
from app.domain.pricing import ExchangeRateSnapshot
from app.domain.shipping import TariffPolicy, TariffRuleSet


def _rules(**overrides: object) -> TariffRuleSet:
    """构造一个最小可用规则快照，并允许测试覆盖单个字段。"""

    values: dict[str, object] = {
        "tariff_rates": {"CN": {"*": Decimal("0.1"), "旅行装备": Decimal("0.2")}},
        "de_minimis_cny_minor": {"CN": 1000},
        "base_freight_cny_minor": {"CN": 100},
        "version": "v1",
        "continuation_ratio": Decimal("0.5"),
    }
    values.update(overrides)
    return TariffRuleSet(**values)  # type: ignore[arg-type]


def _rates() -> ExchangeRateSnapshot:
    """返回只含 CNY/USD 的确定性汇率快照。"""

    return ExchangeRateSnapshot(
        rates_to_cny={"CNY": Decimal(1), "USD": Decimal(7)},
        version="rates-v1",
        as_of="2026-09-01",
        source="test",
    )


@pytest.mark.parametrize("amount", [-1, True, 1.5, "1"])
def test_money_rejects_invalid_minor_units(amount: object) -> None:
    """最小货币单位只接受非负且非布尔的整数。"""

    with pytest.raises(ValueError, match="amount_minor"):
        Money(amount, "CNY")  # type: ignore[arg-type]


@pytest.mark.parametrize("major", [True, "not-a-number", "Infinity", -0.01])
def test_money_rejects_invalid_major_units(major: object) -> None:
    """主单位转换拒绝布尔、非法数字、无穷值和负数。"""

    with pytest.raises((TypeError, ValueError)):
        Money.from_major_units(major, "CNY")  # type: ignore[arg-type]


def test_money_arithmetic_requires_compatible_operands() -> None:
    """金额加法不隐式换汇，数量乘法拒绝不安全类型。"""

    value = Money(125, " cny ")
    assert value.add(Money(75, "CNY")) == Money(200, "CNY")
    assert value.multiply(0) == Money(0, "CNY")
    assert value.to_dict() == {"amount_minor": 125, "major": 1.25, "currency": "CNY"}
    with pytest.raises(ValueError, match="币种不一致"):
        value.add(Money(1, "USD"))
    for quantity in (-1, True, 1.5):
        with pytest.raises(ValueError, match="quantity"):
            value.multiply(quantity)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"normalized_query": "x" * 501},
        {"normalized_query": "商品", "top_k": True},
        {"normalized_query": "商品", "top_k": "5"},
        {"normalized_query": "商品", "top_k": 51},
        {"normalized_query": "商品", "quantity": True},
        {"normalized_query": "商品", "quantity": "1"},
        {"normalized_query": "商品", "category": "x" * 101},
        {"normalized_query": "商品", "ship_to": "1N"},
        {"normalized_query": "商品", "ship_to": "中国"},
        {"normalized_query": "商品", "locale": "  "},
        {"normalized_query": "商品", "price_max_major": True},
        {"normalized_query": "商品", "price_max_major": float("inf")},
    ],
)
def test_product_search_spec_rejects_remaining_invalid_branches(
    kwargs: dict[str, object],
) -> None:
    """工具入参进入应用层后仍由领域规格执行最终校验。"""

    with pytest.raises((TypeError, ValueError)):
        ProductSearchSpec(**kwargs)  # type: ignore[arg-type]


def test_product_search_spec_normalizes_optional_fields_and_budget() -> None:
    """规格统一查询、品类、国家、语言和币种，并精确生成预算。"""

    spec = ProductSearchSpec(
        normalized_query=" 旅行箱 ",
        category=" 旅行装备 ",
        ship_to=" cn ",
        locale=" zh-CN ",
        target_currency=" cny ",
        price_max_major=99.995,
    )
    assert (spec.normalized_query, spec.category, spec.ship_to) == (
        "旅行箱",
        "旅行装备",
        "CN",
    )
    assert spec.locale == "zh-CN"
    assert spec.price_cap == Money(10000, "CNY")
    assert ProductSearchSpec("商品", category="   ").price_cap is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"tariff_rates": {"CHINA": {"*": Decimal("0.1")}}},
        {"tariff_rates": {"C1": {"*": Decimal("0.1")}}},
        {
            "tariff_rates": {
                "cn": {"*": Decimal("0.1")},
                "CN": {"*": Decimal("0.2")},
            },
            "de_minimis_cny_minor": {"CN": 1},
            "base_freight_cny_minor": {"CN": 1},
        },
        {"tariff_rates": {"CN": {"旅行装备": Decimal("0.1")}}},
        {"tariff_rates": {"CN": {"*": Decimal("-0.1")}}},
        {"tariff_rates": {"CN": {"*": Decimal("Infinity")}}},
        {"de_minimis_cny_minor": {"US": 1}},
        {"base_freight_cny_minor": {"US": 1}},
        {"tariff_rates": {}, "de_minimis_cny_minor": {}, "base_freight_cny_minor": {}},
        {"version": "  "},
        {"de_minimis_cny_minor": {"CN": -1}},
        {"de_minimis_cny_minor": {"CN": True}},
        {"base_freight_cny_minor": {"CN": -1}},
        {"base_freight_cny_minor": {"CN": True}},
        {"continuation_ratio": Decimal("-0.1")},
        {"continuation_ratio": Decimal("Infinity")},
    ],
)
def test_tariff_rule_set_rejects_invalid_snapshots(
    overrides: dict[str, object],
) -> None:
    """外部规则配置必须完整、有限且覆盖相同目的地。"""

    with pytest.raises(ValueError):
        _rules(**overrides)


def test_tariff_policy_supports_specific_and_fallback_rates() -> None:
    """报价覆盖免税、品类税率、兜底税率和续件运费。"""

    policy = TariffPolicy(_rules())
    assert policy.supported_destinations() == ("CN",)
    exempt = policy.quote(
        unit_price=Money(100, "CNY"),
        category="旅行装备",
        ship_to=" cn ",
        quantity=1,
        target_currency="CNY",
        rates=_rates(),
    )
    taxed = policy.quote(
        unit_price=Money(1000, "CNY"),
        category="其他",
        ship_to="CN",
        quantity=2,
        target_currency="CNY",
        rates=_rates(),
    )
    assert exempt.de_minimis_applied is True
    assert taxed.freight == Money(150, "CNY")
    assert taxed.tariff == Money(100, "CNY")
    assert taxed.de_minimis_applied is False
    with pytest.raises(ValueError, match="暂不支持目的地"):
        policy.quote(
            unit_price=Money(1, "CNY"),
            category="其他",
            ship_to="US",
            quantity=1,
            target_currency="CNY",
            rates=_rates(),
        )
    for quantity in (0, True, 1.5):
        with pytest.raises(ValueError, match="quantity"):
            policy.quote(
                unit_price=Money(1, "CNY"),
                category="其他",
                ship_to="CN",
                quantity=quantity,  # type: ignore[arg-type]
                target_currency="CNY",
                rates=_rates(),
            )


def _line(item_id: str = "item-1", currency: str = "CNY") -> OrderLine:
    """构造订单领域测试使用的合法订单行。"""

    return OrderLine(item_id, f"{item_id}-sku", item_id, Money(100, currency), 2)


def _order(**overrides: object) -> Order:
    """构造未支付、未占库存的最小订单意向。"""

    values: dict[str, object] = {
        "order_id": "order-1",
        "buyer_id": "buyer-1",
        "index_id": "products",
        "shipping_address": Address(" 张三 ", " cn ", " 上海 ", " 测试路 1 号 "),
        "lines": (_line(),),
        "idempotency_key": "request-1",
    }
    values.update(overrides)
    return Order(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["recipient_name", "city", "address_line"])
def test_address_rejects_missing_required_fields(field: str) -> None:
    """地址必要字段为空时不能创建订单意向。"""

    values = {
        "recipient_name": "张三",
        "country": "CN",
        "city": "上海",
        "address_line": "测试路 1 号",
    }
    values[field] = " "
    with pytest.raises(ValueError, match=field):
        Address(**values)
    with pytest.raises(ValueError, match="country"):
        Address("张三", "中国", "上海", "测试路 1 号")


@pytest.mark.parametrize(
    ("args", "error"),
    [
        (("", "sku", "title", Money(1, "CNY"), 1), ValueError),
        (("item", "", "title", Money(1, "CNY"), 1), ValueError),
        (("item", "sku", "title", Money(1, "CNY"), True), TypeError),
        (("item", "sku", "title", Money(1, "CNY"), 0), ValueError),
    ],
)
def test_order_line_rejects_invalid_identity_and_quantity(
    args: tuple[object, ...],
    error: type[Exception],
) -> None:
    """订单行必须具备稳定商品/SKU身份和正整数数量。"""

    with pytest.raises(error):
        OrderLine(*args)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "overrides",
    [
        {"order_id": " "},
        {"buyer_id": " "},
        {"index_id": " "},
        {"idempotency_key": " "},
        {"lines": ()},
        {"lines": (_line("one"), _line("two", "USD"))},
        {"status": OrderStatus.CANCELLED},
        {
            "status": OrderStatus.CANCELLED,
            "cancelled_at": datetime.now(UTC),
            "cancel_reason": "",
        },
    ],
)
def test_order_rejects_invalid_aggregate_state(overrides: dict[str, object]) -> None:
    """订单聚合维护身份、订单行、币种和取消状态一致性。"""

    with pytest.raises(ValueError):
        _order(**overrides)


def test_order_total_cancel_and_serialization_cover_both_states() -> None:
    """订单支持多行汇总、一次取消和稳定的边界序列化。"""

    order = _order(lines=(_line("one"), _line("two")))
    assert order.total_amount() == Money(400, "CNY")
    before = order.to_dict()
    assert before["cancelled_at"] is None
    assert before["inventory_reserved"] is False
    assert before["payment_status"] == "not_supported"
    with pytest.raises(ValueError, match="reason"):
        order.cancel(" ")
    order.cancel(" 用户改变主意 ")
    assert order.to_dict()["cancelled_at"] is not None
    with pytest.raises(ValueError, match="CONFIRMED"):
        order.cancel("再次取消")
