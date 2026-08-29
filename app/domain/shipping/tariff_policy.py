"""可注入规则快照的纯关税与运费计算策略。"""

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from types import MappingProxyType

from app.domain.catalog import Money
from app.domain.pricing import ExchangeRateSnapshot

from .models import ShippingQuote


@dataclass(frozen=True, slots=True)
class TariffRuleSet:
    """一个版本化的目的国关税、免税额和基础运费规则快照。"""

    tariff_rates: Mapping[str, Mapping[str, Decimal]]
    de_minimis_cny_minor: Mapping[str, int]
    base_freight_cny_minor: Mapping[str, int]
    version: str
    continuation_ratio: Decimal = Decimal("0.6")

    def __post_init__(self) -> None:
        """三个规则表必须覆盖相同目的地，费率和金额不能为负。"""

        normalized_rates: dict[str, Mapping[str, Decimal]] = {}
        for destination, category_rates in self.tariff_rates.items():
            normalized_destination = destination.strip().upper()
            if len(normalized_destination) != 2 or not normalized_destination.isalpha():
                raise ValueError("关税目的地必须是两位字母代码")
            if normalized_destination in normalized_rates:
                raise ValueError(f"关税目的地重复：{normalized_destination}")
            parsed = {
                category.strip(): Decimal(str(rate))
                for category, rate in category_rates.items()
            }
            if (
                "*" not in parsed
                or any(not rate.is_finite() or rate < 0 for rate in parsed.values())
            ):
                raise ValueError("每个目的地必须提供有限、非负的 * 兜底税率")
            normalized_rates[normalized_destination] = MappingProxyType(parsed)

        normalized_thresholds = {
            destination.strip().upper(): value
            for destination, value in self.de_minimis_cny_minor.items()
        }
        normalized_freight = {
            destination.strip().upper(): value
            for destination, value in self.base_freight_cny_minor.items()
        }
        destinations = set(normalized_rates)
        if destinations != set(normalized_thresholds) or destinations != set(
            normalized_freight
        ):
            raise ValueError("关税、免税额和运费规则必须覆盖相同目的地")
        version = self.version.strip()
        if not destinations or not version:
            raise ValueError("TariffRuleSet 必须包含目的地和版本")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in normalized_thresholds.values()
        ):
            raise ValueError("免税额度不能为负")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in normalized_freight.values()
        ):
            raise ValueError("基础运费不能为负")
        continuation_ratio = Decimal(str(self.continuation_ratio))
        if not continuation_ratio.is_finite() or continuation_ratio < 0:
            raise ValueError("续件运费比例不能为负")
        object.__setattr__(self, "tariff_rates", MappingProxyType(normalized_rates))
        object.__setattr__(
            self,
            "de_minimis_cny_minor",
            MappingProxyType(normalized_thresholds),
        )
        object.__setattr__(
            self,
            "base_freight_cny_minor",
            MappingProxyType(normalized_freight),
        )
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "continuation_ratio", continuation_ratio)


@dataclass(frozen=True, slots=True)
class TariffPolicy:
    """根据规则和汇率快照计算估算到手价。"""

    rules: TariffRuleSet

    def supported_destinations(self) -> tuple[str, ...]:
        """返回当前规则快照支持的目的地。"""

        return tuple(sorted(self.rules.tariff_rates))

    def quote(
        self,
        *,
        unit_price: Money,
        category: str,
        ship_to: str,
        quantity: int,
        target_currency: str,
        rates: ExchangeRateSnapshot,
    ) -> ShippingQuote:
        """计算目标币种下的商品小计、运费、关税及到手总价。"""

        destination = ship_to.strip().upper()
        if destination not in self.rules.tariff_rates:
            supported = ", ".join(self.supported_destinations())
            raise ValueError(f"暂不支持目的地 {destination}；支持：{supported}")
        if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity < 1:
            raise ValueError("quantity 必须是正整数")

        # 商品小计必须随数量增长；参考脚本只增加运费而漏乘了商品小计。
        subtotal_original = unit_price.multiply(quantity)
        subtotal_target = rates.convert(subtotal_original, target_currency)

        base_freight = self.rules.base_freight_cny_minor[destination]
        freight_factor = Decimal(1) + self.rules.continuation_ratio * Decimal(
            quantity - 1
        )
        freight_minor = int(
            (Decimal(base_freight) * freight_factor).quantize(
                Decimal(1),
                rounding=ROUND_HALF_UP,
            )
        )
        freight_target = rates.convert(
            Money(freight_minor, "CNY"),
            target_currency,
        )

        subtotal_cny = rates.convert(subtotal_original, "CNY")
        threshold = self.rules.de_minimis_cny_minor[destination]
        taxable_minor = max(0, subtotal_cny.amount_minor - threshold)
        category_rates = self.rules.tariff_rates[destination]
        tariff_rate = category_rates.get(category, category_rates["*"])
        tariff_minor = int(
            (Decimal(taxable_minor) * tariff_rate).quantize(
                Decimal(1),
                rounding=ROUND_HALF_UP,
            )
        )
        tariff_target = rates.convert(
            Money(tariff_minor, "CNY"),
            target_currency,
        )
        return ShippingQuote(
            ship_to=destination,
            quantity=quantity,
            subtotal=subtotal_target,
            freight=freight_target,
            tariff=tariff_target,
            tariff_rate=tariff_rate,
            de_minimis_applied=taxable_minor == 0,
            tariff_rule_version=self.rules.version,
            exchange_rate_version=rates.version,
            exchange_rate_as_of=rates.as_of,
        )
