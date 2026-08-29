"""单进程部署使用的异步会话事件总线。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from app.application.events import TradeEvent


@dataclass(slots=True)
class InMemoryTradeEventBus:
    """按 shopping_session_id 分区，并为每个订阅者维护独立队列。"""

    max_queue_size: int = 256
    _subscribers: dict[str, list[asyncio.Queue[TradeEvent]]] = field(
        default_factory=dict,
        init=False,
    )

    def subscribe(self, shopping_session_id: str) -> asyncio.Queue[TradeEvent]:
        """创建有界订阅队列；调用方结束时必须 unsubscribe。"""

        if not shopping_session_id.strip():
            raise ValueError("shopping_session_id 不能为空")
        queue: asyncio.Queue[TradeEvent] = asyncio.Queue(self.max_queue_size)
        self._subscribers.setdefault(shopping_session_id, []).append(queue)
        return queue

    def unsubscribe(
        self,
        shopping_session_id: str,
        queue: asyncio.Queue[TradeEvent],
    ) -> None:
        """注销一个订阅者，并在会话无订阅者时清理分区。"""

        queues = self._subscribers.get(shopping_session_id)
        if queues is None:
            return
        if queue in queues:
            queues.remove(queue)
        if not queues:
            self._subscribers.pop(shopping_session_id, None)

    async def publish(self, event: TradeEvent) -> None:
        """向当前进程内同一购物会话的所有订阅者投递事件。"""

        queues = tuple(self._subscribers.get(event.shopping_session_id, ()))
        for queue in queues:
            # UI 过程事件允许丢弃最旧项，避免慢 WebSocket 阻塞 AgentLoop。
            if queue.full():
                queue.get_nowait()
                queue.task_done()
            queue.put_nowait(event)
