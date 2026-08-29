"""商品搜索用例依赖的跨境计价端口。"""

from typing import Protocol

from app.domain.catalog import Money
from app.domain.shipping import ShippingQuote


class PricingProvider(Protocol):
    """提供同一快照口径的币种转换和估算到手价。"""

    def supported_destinations(self) -> tuple[str, ...]:
        """返回当前规则快照可以报价的目的地代码。"""

        ...

    def convert(self, money: Money, target_currency: str) -> Money:
        """把金额按当前汇率快照转换为目标币种。"""

        ...

    def quote(
        self,
        *,
        unit_price: Money,
        category: str,
        ship_to: str,
        quantity: int,
        target_currency: str,
    ) -> ShippingQuote:
        """按当前汇率、运费和关税规则返回估算到手价。"""

        ...
