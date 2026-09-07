"""订单意向单应用命令。"""

from dataclasses import dataclass

from app.domain.order import Address


@dataclass(frozen=True, slots=True)
class OrderItemCommand:
    """用户确认的一条商品SKU与数量。"""

    item_id: str
    sku_id: str
    quantity: int = 1

    def __post_init__(self) -> None:
        if not self.item_id.strip() or not self.sku_id.strip():
            raise ValueError("item_id 和 sku_id 不能为空")
        if not isinstance(self.quantity, int) or isinstance(self.quantity, bool):
            raise TypeError("quantity 必须是整数")
        if self.quantity < 1:
            raise ValueError("quantity 必须大于 0")


@dataclass(frozen=True, slots=True)
class PlaceOrderCommand:
    """创建订单意向单所需的可信应用命令。"""

    buyer_id: str
    index_id: str
    items: tuple[OrderItemCommand, ...]
    shipping_address: Address
    idempotency_key: str

    def __post_init__(self) -> None:
        if not self.buyer_id.strip() or not self.index_id.strip():
            raise ValueError("buyer_id 和 index_id 不能为空")
        if not self.items:
            raise ValueError("items 不能为空")
        if not self.idempotency_key.strip():
            raise ValueError("idempotency_key 不能为空")
