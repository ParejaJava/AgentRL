"""SQLite 订单意向单仓储。"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

from app.application.orders import OrderRepository
from app.domain.catalog import Money
from app.domain.order import Address, Order, OrderLine, OrderStatus


class SQLiteOrderRepository(OrderRepository):
    """以事务和唯一键保证同一买家的订单创建幂等。"""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS order_intents (
                    order_id TEXT PRIMARY KEY,
                    buyer_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    UNIQUE (buyer_id, idempotency_key)
                )
                """
            )

    async def create_or_get(self, order: Order) -> Order:
        """原子创建订单；相同幂等键重复提交时返回首次结果。"""

        return await asyncio.to_thread(self._create_or_get_sync, order)

    def _create_or_get_sync(self, order: Order) -> Order:
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT payload FROM order_intents
                WHERE buyer_id = ? AND idempotency_key = ?
                """,
                (order.buyer_id, order.idempotency_key),
            ).fetchone()
            if existing is not None:
                persisted = _deserialize_order(str(existing["payload"]))
                if _request_shape(persisted) != _request_shape(order):
                    raise ValueError("相同幂等键不能创建内容不同的订单")
                return persisted
            payload = _serialize_order(order)
            connection.execute(
                """
                INSERT INTO order_intents
                    (order_id, buyer_id, idempotency_key, status, payload)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    order.order_id,
                    order.buyer_id,
                    order.idempotency_key,
                    order.status.value,
                    payload,
                ),
            )
            return order

    async def get(self, order_id: str) -> Order | None:
        """按订单编号读取意向单。"""

        return await asyncio.to_thread(self._get_sync, order_id)

    def _get_sync(self, order_id: str) -> Order | None:
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT payload FROM order_intents WHERE order_id = ?",
                (order_id,),
            ).fetchone()
        return None if row is None else _deserialize_order(str(row["payload"]))

    async def save(self, order: Order) -> None:
        """保存已发生状态迁移的订单。"""

        await asyncio.to_thread(self._save_sync, order)

    def _save_sync(self, order: Order) -> None:
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """
                UPDATE order_intents SET status = ?, payload = ?
                WHERE order_id = ? AND buyer_id = ?
                """,
                (
                    order.status.value,
                    _serialize_order(order),
                    order.order_id,
                    order.buyer_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"订单不存在：{order.order_id}")


def _serialize_order(order: Order) -> str:
    return json.dumps(order.to_dict(), ensure_ascii=False, sort_keys=True)


def _deserialize_order(payload: str) -> Order:
    raw = json.loads(payload)
    address = Address(**raw["shipping_address"])
    lines = tuple(
        OrderLine(
            item_id=line["item_id"],
            sku_id=line["sku_id"],
            title=line["title"],
            unit_price=Money(
                amount_minor=int(line["unit_price"]["amount_minor"]),
                currency=line["unit_price"]["currency"],
            ),
            quantity=int(line["quantity"]),
        )
        for line in raw["lines"]
    )
    return Order(
        order_id=raw["order_id"],
        buyer_id=raw["buyer_id"],
        index_id=raw["index_id"],
        shipping_address=address,
        lines=lines,
        idempotency_key=raw.get("idempotency_key", "legacy"),
        status=OrderStatus(raw["status"]),
        created_at=datetime.fromisoformat(raw["created_at"]),
        cancelled_at=(
            datetime.fromisoformat(raw["cancelled_at"])
            if raw.get("cancelled_at")
            else None
        ),
        cancel_reason=raw.get("cancel_reason"),
    )


def _request_shape(order: Order) -> tuple[object, ...]:
    """比较幂等请求的业务字段，不比较服务端生成字段。"""

    return (
        order.index_id,
        tuple(
            (line.item_id, line.sku_id, line.quantity, line.unit_price)
            for line in order.lines
        ),
        tuple(sorted(order.shipping_address.to_dict().items())),
    )
