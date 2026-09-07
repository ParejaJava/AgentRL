"""订单意向单创建、查询和取消应用服务。"""

from __future__ import annotations

from uuid import uuid4

from app.domain.order import Order, OrderLine

from .models import PlaceOrderCommand
from .ports import OrderRepository, ProductReader


class OrderService:
    """在只读商品快照之上创建可审计、幂等的订单意向单。"""

    def __init__(self, products: ProductReader, orders: OrderRepository) -> None:
        self._products = products
        self._orders = orders

    async def place(self, command: PlaceOrderCommand) -> Order:
        """校验商品、配送和库存快照后创建已确认意向单。"""

        lines: list[OrderLine] = []
        for item in command.items:
            product = self._products.get(command.index_id, item.item_id)
            if product is None:
                raise ValueError(f"商品不存在：{item.item_id}")
            if command.shipping_address.country not in product.ships_to:
                raise ValueError(
                    f"商品 {item.item_id} 不支持配送到 "
                    f"{command.shipping_address.country}"
                )
            sku = next((sku for sku in product.skus if sku.sku_id == item.sku_id), None)
            if sku is None:
                raise ValueError(f"SKU 不存在：{item.item_id}/{item.sku_id}")
            if sku.stock < item.quantity:
                raise ValueError(f"SKU 库存快照不足：{item.item_id}/{item.sku_id}")
            lines.append(
                OrderLine(
                    item_id=product.item_id,
                    sku_id=sku.sku_id,
                    title=f"{product.title}（{sku.spec}）",
                    unit_price=sku.price,
                    quantity=item.quantity,
                )
            )

        order = Order(
            order_id=f"GBX-{uuid4().hex[:12].upper()}",
            buyer_id=command.buyer_id,
            index_id=command.index_id,
            shipping_address=command.shipping_address,
            lines=tuple(lines),
            idempotency_key=command.idempotency_key,
        )
        return await self._orders.create_or_get(order)

    async def query(self, order_id: str, buyer_id: str) -> Order:
        """查询属于当前买家的订单，防止跨买家读取。"""

        order = await self._orders.get(order_id.strip())
        if order is None or order.buyer_id != buyer_id.strip():
            raise ValueError(f"订单不存在：{order_id}")
        return order

    async def cancel(self, order_id: str, buyer_id: str, reason: str) -> Order:
        """取消属于当前买家的意向单并持久化状态。"""

        order = await self.query(order_id, buyer_id)
        order.cancel(reason)
        await self._orders.save(order)
        return order
