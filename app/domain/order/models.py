"""不包含支付和真实库存预占的订单意向单聚合。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum

from app.domain.catalog import Money


class OrderStatus(str, Enum):
    """当前 MVP 支持的意向单状态。"""

    CONFIRMED = "CONFIRMED"
    CANCELLED = "CANCELLED"


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class Address:
    """创建跨境订单意向单所需的收货地址快照。"""

    recipient_name: str
    country: str
    city: str
    address_line: str
    state: str = ""
    postal_code: str = ""
    phone: str = ""

    def __post_init__(self) -> None:
        """校验必要字段并统一国家/地区代码。"""

        for field_name in ("recipient_name", "city", "address_line"):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"Address.{field_name} 不能为空")
            object.__setattr__(self, field_name, value)
        country = self.country.strip().upper()
        if len(country) != 2 or not country.isascii() or not country.isalpha():
            raise ValueError("Address.country 必须是两位国家或地区代码")
        object.__setattr__(self, "country", country)

    def to_dict(self) -> dict[str, str]:
        """返回可持久化的结构化地址，不拼接或丢弃字段。"""

        return {
            "recipient_name": self.recipient_name,
            "country": self.country,
            "state": self.state,
            "city": self.city,
            "address_line": self.address_line,
            "postal_code": self.postal_code,
            "phone": self.phone,
        }


@dataclass(frozen=True, slots=True)
class OrderLine:
    """保存创建意向单时的商品和价格快照。"""

    item_id: str
    sku_id: str
    title: str
    unit_price: Money
    quantity: int

    def __post_init__(self) -> None:
        """订单行必须指向稳定商品/SKU，并使用正整数数量。"""

        if not self.item_id.strip() or not self.sku_id.strip():
            raise ValueError("OrderLine.item_id 和 sku_id 不能为空")
        if not isinstance(self.quantity, int) or isinstance(self.quantity, bool):
            raise TypeError("OrderLine.quantity 必须是整数")
        if self.quantity < 1:
            raise ValueError("OrderLine.quantity 必须大于 0")

    def subtotal(self) -> Money:
        """使用 Money 最小单位计算订单行小计。"""

        return self.unit_price.multiply(self.quantity)

    def to_dict(self) -> dict[str, object]:
        """输出不依赖商品当前价格的订单行快照。"""

        return {
            "item_id": self.item_id,
            "sku_id": self.sku_id,
            "title": self.title,
            "unit_price": self.unit_price.to_dict(),
            "quantity": self.quantity,
            "subtotal": self.subtotal().to_dict(),
        }


@dataclass(slots=True)
class Order:
    """已获用户确认但尚未支付、未预占真实库存的订单意向单。"""

    order_id: str
    buyer_id: str
    index_id: str
    shipping_address: Address
    lines: tuple[OrderLine, ...]
    idempotency_key: str
    status: OrderStatus = OrderStatus.CONFIRMED
    created_at: datetime = field(default_factory=_utc_now)
    cancelled_at: datetime | None = None
    cancel_reason: str | None = None

    def __post_init__(self) -> None:
        """维护订单身份、币种和状态字段的不变量。"""

        if not self.order_id.strip() or not self.buyer_id.strip():
            raise ValueError("Order.order_id 和 buyer_id 不能为空")
        if not self.index_id.strip() or not self.idempotency_key.strip():
            raise ValueError("Order.index_id 和 idempotency_key 不能为空")
        if not self.lines:
            raise ValueError("订单至少需要一条订单行")
        currencies = {line.unit_price.currency for line in self.lines}
        if len(currencies) != 1:
            raise ValueError("订单行币种必须一致")
        if self.status is OrderStatus.CANCELLED and (
            self.cancelled_at is None or not self.cancel_reason
        ):
            raise ValueError("已取消订单必须包含取消时间和原因")

    def total_amount(self) -> Money:
        """汇总所有同币种订单行。"""

        total = self.lines[0].subtotal()
        for line in self.lines[1:]:
            total = total.add(line.subtotal())
        return total

    def cancel(self, reason: str) -> None:
        """取消已确认意向单；重复取消会被拒绝。"""

        normalized_reason = reason.strip()
        if self.status is not OrderStatus.CONFIRMED:
            raise ValueError("只有 CONFIRMED 订单可以取消")
        if not normalized_reason:
            raise ValueError("取消订单必须提供 reason")
        self.status = OrderStatus.CANCELLED
        self.cancelled_at = _utc_now()
        self.cancel_reason = normalized_reason

    def to_dict(self) -> dict[str, object]:
        """返回对工具和REST稳定的意向单快照。"""

        return {
            "order_id": self.order_id,
            "order_type": "purchase_intent",
            "buyer_id": self.buyer_id,
            "index_id": self.index_id,
            "idempotency_key": self.idempotency_key,
            "status": self.status.value,
            "shipping_address": self.shipping_address.to_dict(),
            "lines": [line.to_dict() for line in self.lines],
            "total_amount": self.total_amount().to_dict(),
            "inventory_reserved": False,
            "payment_status": "not_supported",
            "created_at": self.created_at.isoformat(),
            "cancelled_at": (
                self.cancelled_at.isoformat() if self.cancelled_at else None
            ),
            "cancel_reason": self.cancel_reason,
        }
