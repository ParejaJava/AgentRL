"""与具体检索基础设施无关的商品搜索业务规格。"""

import math
from dataclasses import dataclass

from .money import Money, normalize_currency


@dataclass(frozen=True, slots=True)
class ProductSearchSpec:
    """封装语义查询与结构化硬约束，并集中维护业务不变量。"""

    normalized_query: str
    category: str | None = None
    ship_to: str | None = None
    locale: str = "zh-CN"
    top_k: int = 5
    target_currency: str = "CNY"
    price_max_major: float | None = None
    quantity: int = 1

    def __post_init__(self) -> None:
        """规范化查询槽位，并拒绝不可能安全执行的搜索条件。"""

        query = self.normalized_query.strip()
        if not query:
            raise ValueError("normalized_query 不能为空")
        if len(query) > 500:
            raise ValueError("normalized_query 不能超过 500 个字符")
        if not isinstance(self.top_k, int) or isinstance(self.top_k, bool):
            raise TypeError("top_k 必须是整数")
        if not 1 <= self.top_k <= 50:
            raise ValueError("top_k 必须位于 1 到 50 之间")
        if not isinstance(self.quantity, int) or isinstance(self.quantity, bool):
            raise TypeError("quantity 必须是整数")
        if self.quantity < 1:
            raise ValueError("quantity 必须大于 0")

        category = self.category.strip() if self.category else None
        if category and len(category) > 100:
            raise ValueError("category 不能超过 100 个字符")
        ship_to = self.ship_to.strip().upper() if self.ship_to else None
        if ship_to and (
            len(ship_to) != 2
            or not ship_to.isascii()
            or not ship_to.isalpha()
        ):
            raise ValueError("ship_to 必须是两位字母国家或地区代码")
        locale = self.locale.strip()
        if not locale:
            raise ValueError("locale 不能为空")
        target_currency = normalize_currency(self.target_currency)
        if self.price_max_major is not None:
            if (
                isinstance(self.price_max_major, bool)
                or not math.isfinite(self.price_max_major)
                or self.price_max_major < 0
            ):
                raise ValueError("price_max_major 必须是有限的非负数")
            # 立即走 Money 转换，确保金额能按目标币种精度安全表示。
            Money.from_major_units(self.price_max_major, target_currency)

        object.__setattr__(self, "normalized_query", query)
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "ship_to", ship_to)
        object.__setattr__(self, "locale", locale)
        object.__setattr__(self, "target_currency", target_currency)

    @property
    def price_cap(self) -> Money | None:
        """把可选预算转换成目标币种的精确金额值对象。"""

        if self.price_max_major is None:
            return None
        return Money.from_major_units(
            self.price_max_major,
            self.target_currency,
        )
