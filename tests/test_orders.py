"""订单意向领域、仓储与 Agent 工具边界测试。"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.application.orders import OrderItemCommand, OrderService, PlaceOrderCommand
from app.application.runtime import AgentExecutionContext, ShoppingContextSnapshot
from app.domain.catalog import Money, Product, Sku
from app.domain.order import Address, OrderStatus
from app.infrastructure.context import reset_context, set_context
from app.infrastructure.langchain.tools import create_order_tools
from app.infrastructure.orders import SQLiteOrderRepository


class _Products:
    def __init__(self, product: Product) -> None:
        self._product = product

    def get(self, index_id: str, item_id: str) -> Product | None:
        if index_id == "products" and item_id == self._product.item_id:
            return self._product
        return None


def _product() -> Product:
    return Product(
        item_id="item-1",
        title="旅行收纳袋",
        ships_to=("CN",),
        skus=(Sku("sku-1", "蓝色", Money(8900, "CNY"), stock=3),),
        primary_sku_id="sku-1",
    )


def _address() -> Address:
    return Address("张三", "CN", "上海", "测试路 1 号")


def test_order_repository_is_idempotent_and_persists_cancellation(tmp_path) -> None:
    service = OrderService(
        _Products(_product()),
        SQLiteOrderRepository(tmp_path / "orders.db"),
    )
    command = PlaceOrderCommand(
        buyer_id="buyer-1",
        index_id="products",
        items=(OrderItemCommand("item-1", "sku-1", 2),),
        shipping_address=_address(),
        idempotency_key="run-1",
    )

    async def exercise() -> None:
        first = await service.place(command)
        repeated = await service.place(command)
        assert repeated.order_id == first.order_id
        assert first.total_amount() == Money(17800, "CNY")
        cancelled = await service.cancel(first.order_id, "buyer-1", "改变主意")
        assert cancelled.status is OrderStatus.CANCELLED
        loaded = await service.query(first.order_id, "buyer-1")
        assert loaded.cancel_reason == "改变主意"
        with pytest.raises(ValueError, match="订单不存在"):
            await service.query(first.order_id, "buyer-2")

    asyncio.run(exercise())


def test_order_tool_requires_confirmation_and_uses_trusted_buyer(tmp_path) -> None:
    service = OrderService(
        _Products(_product()),
        SQLiteOrderRepository(tmp_path / "orders.db"),
    )
    create_order, query_order, _ = create_order_tools(service, index_id="products")
    token = set_context(
        AgentExecutionContext(
            thread_id="thread-1",
            run_id="run-1",
            shopping=ShoppingContextSnapshot("shopping-1", "buyer-1"),
        )
    )
    payload = {
        "items": [{"item_id": "item-1", "sku_id": "sku-1", "quantity": 1}],
        "shipping_address": {
            "recipient_name": "张三",
            "country": "CN",
            "city": "上海",
            "address_line": "测试路 1 号",
        },
    }

    async def exercise() -> None:
        refused = json.loads(
            await create_order.ainvoke({**payload, "user_confirmed": False})
        )
        assert refused["created"] is False
        created = json.loads(
            await create_order.ainvoke({**payload, "user_confirmed": True})
        )
        order = created["order"]
        assert order["buyer_id"] == "buyer-1"
        assert order["inventory_reserved"] is False
        assert order["payment_status"] == "not_supported"
        queried = json.loads(await query_order.ainvoke({"order_id": order["order_id"]}))
        assert queried["order_id"] == order["order_id"]

    try:
        asyncio.run(exercise())
    finally:
        reset_context(token)
