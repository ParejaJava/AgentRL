"""将订单意向用例适配为 LangChain 工具。"""

from __future__ import annotations

import json
from typing import Annotated

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, ConfigDict, Field

from app.application.orders import OrderItemCommand, OrderService, PlaceOrderCommand
from app.domain.order import Address
from app.infrastructure.context import ShoppingContext, require_context


class OrderItemInput(BaseModel):
    """模型可提交的订单行；价格从服务端商品快照读取。"""

    model_config = ConfigDict(extra="forbid")
    item_id: str = Field(min_length=1, max_length=128)
    sku_id: str = Field(min_length=1, max_length=128)
    quantity: int = Field(default=1, ge=1, le=99)


class ShippingAddressInput(BaseModel):
    """创建订单意向单所需的结构化收货地址。"""

    model_config = ConfigDict(extra="forbid")
    recipient_name: str = Field(min_length=1, max_length=100)
    country: str = Field(min_length=2, max_length=2)
    city: str = Field(min_length=1, max_length=100)
    address_line: str = Field(min_length=1, max_length=300)
    state: str = Field(default="", max_length=100)
    postal_code: str = Field(default="", max_length=32)
    phone: str = Field(default="", max_length=32)


def create_order_tools(
    service: OrderService,
    *,
    index_id: str,
) -> tuple[BaseTool, BaseTool, BaseTool]:
    """创建订单意向单的创建、查询与取消工具。"""

    @tool
    async def create_order_intent(
        items: list[OrderItemInput],
        shipping_address: ShippingAddressInput,
        user_confirmed: Annotated[
            bool,
            Field(description="用户是否在当前对话中明确确认了商品、数量和地址"),
        ],
    ) -> str:
        """在用户明确确认后创建购买意向单。

        items 是已确认的商品、SKU 和数量；shipping_address 是完整地址；
        user_confirmed 只有在用户本轮明确确认时才能为 true。本工具不支付也不锁库存。
        """

        if not user_confirmed:
            return json.dumps(
                {"status": "confirmation_required", "created": False},
                ensure_ascii=False,
            )
        execution = require_context()
        context = execution.require_shopping()
        run_scope = execution.run_id or execution.thread_id
        order = await service.place(
            PlaceOrderCommand(
                buyer_id=context.buyer_id,
                index_id=index_id,
                items=tuple(
                    OrderItemCommand(
                        item_id=item.item_id,
                        sku_id=item.sku_id,
                        quantity=item.quantity,
                    )
                    for item in items
                ),
                shipping_address=Address(**shipping_address.model_dump()),
                idempotency_key=f"agent-run:{run_scope}:create-order",
            )
        )
        return json.dumps(
            {"status": "ok", "created": True, "order": order.to_dict()},
            ensure_ascii=False,
        )

    @tool
    async def query_order(
        order_id: Annotated[str, Field(min_length=1, max_length=64)],
    ) -> str:
        """查询当前买家的订单意向状态；order_id 是创建工具返回的编号。"""

        context = ShoppingContext.require_current()
        order = await service.query(order_id, context.buyer_id)
        return json.dumps(order.to_dict(), ensure_ascii=False)

    @tool
    async def cancel_order(
        order_id: Annotated[str, Field(min_length=1, max_length=64)],
        reason: Annotated[str, Field(min_length=1, max_length=300)],
        user_confirmed: Annotated[
            bool,
            Field(description="用户是否明确确认取消该订单意向"),
        ],
    ) -> str:
        """在用户明确确认后取消自己的订单意向单。

        order_id 是订单编号，reason 是用户给出的取消原因；未确认时不得执行状态变更。
        """

        if not user_confirmed:
            return json.dumps(
                {"status": "confirmation_required", "cancelled": False},
                ensure_ascii=False,
            )
        context = ShoppingContext.require_current()
        order = await service.cancel(order_id, context.buyer_id, reason)
        return json.dumps(
            {"status": "ok", "cancelled": True, "order": order.to_dict()},
            ensure_ascii=False,
        )

    return create_order_intent, query_order, cancel_order
