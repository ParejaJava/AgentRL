"""Redis Pub/Sub 跨进程事件背板。"""

from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from uuid import uuid4

from app.application.events import TradeEvent

from .in_memory import InMemoryTradeEventBus


class RedisTradeEventBus:
    """本地 fan-out 与 Redis Pub/Sub 组合的会话事件总线。"""

    def __init__(self, redis_url: str, *, channel_prefix: str = "globex:events") -> None:
        try:
            from redis.asyncio import Redis
        except ImportError as exc:
            raise RuntimeError(
                "已启用 Redis；请运行 `uv sync --extra platform`"
            ) from exc
        self._redis = Redis.from_url(redis_url, decode_responses=True)
        self._prefix = channel_prefix.rstrip(":")
        self._origin = uuid4().hex
        self._local = InMemoryTradeEventBus()
        self._listener: asyncio.Task[None] | None = None
        self._pubsub = None

    def subscribe(self, shopping_session_id: str) -> asyncio.Queue[TradeEvent]:
        """创建本地订阅，并按需启动 Redis 模式订阅监听器。"""

        queue = self._local.subscribe(shopping_session_id)
        if self._listener is None or self._listener.done():
            self._listener = asyncio.create_task(self._listen())
        return queue

    def unsubscribe(
        self,
        shopping_session_id: str,
        queue: asyncio.Queue[TradeEvent],
    ) -> None:
        """移除当前进程订阅者。"""

        self._local.unsubscribe(shopping_session_id, queue)

    async def publish(self, event: TradeEvent) -> None:
        """先投递本地订阅者，再广播到其他 API/Worker 进程。"""

        await self._local.publish(event)
        envelope = json.dumps(
            {"origin": self._origin, "event": event.to_dict()},
            ensure_ascii=False,
        )
        await self._redis.publish(f"{self._prefix}:{event.shopping_session_id}", envelope)

    async def ping(self) -> bool:
        """检查 Redis 连接是否可用。"""

        return bool(await self._redis.ping())

    async def close(self) -> None:
        """停止监听并关闭 Redis 连接。"""

        if self._listener is not None:
            self._listener.cancel()
            with suppress(asyncio.CancelledError):
                await self._listener
        if self._pubsub is not None:
            await self._pubsub.aclose()
        await self._redis.aclose()

    async def _listen(self) -> None:
        pubsub = self._redis.pubsub()
        self._pubsub = pubsub
        await pubsub.psubscribe(f"{self._prefix}:*")
        try:
            async for message in pubsub.listen():
                if message.get("type") != "pmessage":
                    continue
                try:
                    envelope = json.loads(str(message["data"]))
                    if envelope.get("origin") == self._origin:
                        continue
                    event = TradeEvent.from_dict(envelope["event"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                await self._local.publish(event)
        finally:
            await pubsub.punsubscribe(f"{self._prefix}:*")
