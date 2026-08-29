"""不可变汇率快照及精确币种转换规则。"""

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from types import MappingProxyType

from app.domain.catalog.money import Money, normalize_currency


@dataclass(frozen=True, slots=True)
class ExchangeRateSnapshot:
    """以 CNY 为中间价的、带版本和时间信息的汇率快照。"""

    rates_to_cny: Mapping[str, Decimal]
    version: str
    as_of: str
    source: str

    def __post_init__(self) -> None:
        """汇率必须为正有限数，并统一使用规范币种代码。"""

        normalized: dict[str, Decimal] = {}
        for currency, raw_rate in self.rates_to_cny.items():
            normalized_currency = normalize_currency(currency)
            try:
                rate = Decimal(str(raw_rate))
            except InvalidOperation as exc:
                raise ValueError(f"{normalized_currency} 汇率不是合法数字") from exc
            if not rate.is_finite() or rate <= 0:
                raise ValueError(f"{normalized_currency} 汇率必须是有限正数")
            normalized[normalized_currency] = rate
        if normalized.get("CNY") != Decimal(1):
            raise ValueError("CNY 中间价必须为 1")
        if not self.version.strip() or not self.as_of.strip() or not self.source.strip():
            raise ValueError("汇率快照必须包含 version、as_of 和 source")
        object.__setattr__(self, "rates_to_cny", MappingProxyType(normalized))

    def rate(self, from_currency: str, to_currency: str) -> Decimal:
        """返回一单位源币种可兑换的目标币种数量。"""

        source = normalize_currency(from_currency)
        target = normalize_currency(to_currency)
        try:
            return self.rates_to_cny[source] / self.rates_to_cny[target]
        except KeyError as exc:
            raise ValueError(f"汇率快照不支持币种：{exc.args[0]}") from exc

    def convert(self, money: Money, to_currency: str) -> Money:
        """以主单位做跨币种计算，再按目标币种最小单位精确舍入。"""

        target = normalize_currency(to_currency)
        if target == money.currency:
            return money
        converted_major = money.to_major_units() * self.rate(
            money.currency,
            target,
        )
        return Money.from_major_units(converted_major, target)
