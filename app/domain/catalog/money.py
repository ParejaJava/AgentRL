"""跨境商品价格使用的不可变金额值对象。"""

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Final

CURRENCY_EXPONENTS: Final[dict[str, int]] = {
    "USD": 2,
    "EUR": 2,
    "GBP": 2,
    "JPY": 0,
    "CNY": 2,
    "HKD": 2,
    "AUD": 2,
    "CAD": 2,
    "SGD": 2,
}
SUPPORTED_CURRENCIES: Final[tuple[str, ...]] = tuple(CURRENCY_EXPONENTS)


def normalize_currency(currency: str) -> str:
    """规范化并校验受支持的 ISO-4217 币种代码。"""

    normalized = currency.strip().upper()
    if normalized not in CURRENCY_EXPONENTS:
        raise ValueError(f"不支持的币种：{normalized or currency!r}")
    return normalized


@dataclass(frozen=True, slots=True)
class Money:
    """以最小货币单位保存金额，避免价格计算中的浮点误差。"""

    amount_minor: int
    currency: str

    def __post_init__(self) -> None:
        """金额必须为非负整数，布尔值不能冒充整数。"""

        if (
            not isinstance(self.amount_minor, int)
            or isinstance(self.amount_minor, bool)
            or self.amount_minor < 0
        ):
            raise ValueError("Money.amount_minor 必须是非负整数")
        object.__setattr__(self, "currency", normalize_currency(self.currency))

    @classmethod
    def from_major_units(
        cls,
        major: float | str | Decimal,
        currency: str,
    ) -> "Money":
        """把用户可读的主单位金额按币种精度转换为最小单位。"""

        if isinstance(major, bool):
            raise TypeError("Money.major 不能是布尔值")
        normalized_currency = normalize_currency(currency)
        try:
            decimal_major = Decimal(str(major))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError("Money.major 必须是合法数字") from exc
        if not decimal_major.is_finite() or decimal_major < 0:
            raise ValueError("Money.major 必须是有限的非负数")
        scale = Decimal(10) ** CURRENCY_EXPONENTS[normalized_currency]
        amount_minor = int(
            (decimal_major * scale).quantize(Decimal(1), rounding=ROUND_HALF_UP)
        )
        return cls(amount_minor=amount_minor, currency=normalized_currency)

    @property
    def major(self) -> float:
        """兼容商品展示代码，返回浮点主单位；计算逻辑不得使用此属性。"""

        return float(self.to_major_units())

    def to_major_units(self) -> Decimal:
        """返回精确的 Decimal 主单位金额。"""

        scale = Decimal(10) ** CURRENCY_EXPONENTS[self.currency]
        return Decimal(self.amount_minor) / scale

    def add(self, other: "Money") -> "Money":
        """相加同币种金额，不允许隐式跨币种运算。"""

        if self.currency != other.currency:
            raise ValueError(
                f"Money 币种不一致：{self.currency} vs {other.currency}"
            )
        return Money(self.amount_minor + other.amount_minor, self.currency)

    def multiply(self, quantity: int) -> "Money":
        """按非负整数数量计算小计。"""

        if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity < 0:
            raise ValueError("Money.multiply.quantity 必须是非负整数")
        return Money(self.amount_minor * quantity, self.currency)

    def to_dict(self) -> dict[str, object]:
        """输出索引持久化使用的精确金额及展示主单位。"""

        return {
            "amount_minor": self.amount_minor,
            "major": self.major,
            "currency": self.currency,
        }
