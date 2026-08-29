"""跨境到手价报价值对象。"""

from dataclasses import dataclass
from decimal import Decimal

from app.domain.catalog import Money


@dataclass(frozen=True, slots=True)
class ShippingQuote:
    """同一目标币种下的商品小计、运费、关税和到手总价。"""

    ship_to: str
    quantity: int
    subtotal: Money
    freight: Money
    tariff: Money
    tariff_rate: Decimal
    de_minimis_applied: bool
    tariff_rule_version: str
    exchange_rate_version: str
    exchange_rate_as_of: str

    @property
    def landed_total(self) -> Money:
        """计算商品小计、运费和关税之和。"""

        return self.subtotal.add(self.freight).add(self.tariff)

    def to_dict(self) -> dict[str, object]:
        """输出可以直接内联进商品卡的估算到手价。"""

        return {
            "status": "estimated",
            "ship_to": self.ship_to,
            "quantity": self.quantity,
            "subtotal_major": self.subtotal.major,
            "freight_major": self.freight.major,
            "tariff_major": self.tariff.major,
            "tariff_rate": float(self.tariff_rate),
            "de_minimis_applied": self.de_minimis_applied,
            "landed_total_major": self.landed_total.major,
            "currency": self.subtotal.currency,
            "tariff_rule_version": self.tariff_rule_version,
            "exchange_rate_version": self.exchange_rate_version,
            "exchange_rate_as_of": self.exchange_rate_as_of,
        }
