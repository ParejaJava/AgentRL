"""MVP 使用的版本化静态汇率、关税和运费快照。"""

from dataclasses import dataclass
from decimal import Decimal

from app.domain.catalog import Money
from app.domain.pricing import ExchangeRateSnapshot
from app.domain.shipping import ShippingQuote, TariffPolicy, TariffRuleSet

_RATES_TO_CNY = {
    "CNY": Decimal(1),
    "USD": Decimal("7.10"),
    "EUR": Decimal("7.80"),
    "GBP": Decimal("9.10"),
    "JPY": Decimal("0.048"),
    "HKD": Decimal("0.91"),
    "AUD": Decimal("4.70"),
    "CAD": Decimal("5.20"),
    "SGD": Decimal("5.30"),
}

_TARIFF_RATES = {
    "CN": {
        "数码配件": Decimal("0.13"),
        "旅行装备": Decimal("0.09"),
        "户外运动": Decimal("0.09"),
        "家居生活": Decimal("0.09"),
        "*": Decimal("0.09"),
    },
    "US": {
        "数码配件": Decimal(0),
        "旅行装备": Decimal("0.075"),
        "户外运动": Decimal("0.075"),
        "家居生活": Decimal("0.05"),
        "*": Decimal("0.06"),
    },
    "EU": {"*": Decimal("0.12")},
    "JP": {"*": Decimal("0.08")},
    "SG": {"*": Decimal("0.07")},
}

_DE_MINIMIS_CNY_MINOR = {
    "CN": 500_000,
    "US": 568_000,
    "EU": 117_000,
    "JP": 50_000,
    "SG": 212_000,
}

_BASE_FREIGHT_CNY_MINOR = {
    "CN": 2_500,
    "US": 6_500,
    "EU": 7_500,
    "JP": 4_500,
    "SG": 4_000,
}


@dataclass(frozen=True, slots=True)
class StaticPricingProvider:
    """用静态快照实现计价端口，便于以后替换为远程报价服务。"""

    rates: ExchangeRateSnapshot
    tariff_policy: TariffPolicy

    def supported_destinations(self) -> tuple[str, ...]:
        """返回当前关税与运费规则支持的目的地。"""

        return self.tariff_policy.supported_destinations()

    def convert(self, money: Money, target_currency: str) -> Money:
        """使用当前版本汇率快照转换金额。"""

        return self.rates.convert(money, target_currency)

    def quote(
        self,
        *,
        unit_price: Money,
        category: str,
        ship_to: str,
        quantity: int,
        target_currency: str,
    ) -> ShippingQuote:
        """使用同一汇率和规则快照计算估算到手价。"""

        return self.tariff_policy.quote(
            unit_price=unit_price,
            category=category,
            ship_to=ship_to,
            quantity=quantity,
            target_currency=target_currency,
            rates=self.rates,
        )


def create_static_pricing_provider() -> StaticPricingProvider:
    """装配默认静态汇率和 `static-2026-08` 关税运费快照。"""

    rates = ExchangeRateSnapshot(
        rates_to_cny=_RATES_TO_CNY,
        version="static-2026-08",
        as_of="2026-08-01",
        source="repository-static-snapshot",
    )
    rules = TariffRuleSet(
        tariff_rates=_TARIFF_RATES,
        de_minimis_cny_minor=_DE_MINIMIS_CNY_MINOR,
        base_freight_cny_minor=_BASE_FREIGHT_CNY_MINOR,
        version="static-2026-08",
    )
    return StaticPricingProvider(rates=rates, tariff_policy=TariffPolicy(rules))
