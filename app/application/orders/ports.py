"""订单用例使用的商品读取和订单仓储端口。"""

from typing import Protocol

from app.domain.catalog import Product
from app.domain.order import Order


class ProductReader(Protocol):
    """从指定检索域读取权威商品快照。"""

    def get(self, index_id: str, item_id: str) -> Product | None: ...


class OrderRepository(Protocol):
    """持久化订单意向单并提供幂等创建。"""

    async def create_or_get(self, order: Order) -> Order: ...

    async def get(self, order_id: str) -> Order | None: ...

    async def save(self, order: Order) -> None: ...
