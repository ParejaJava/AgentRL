"""应用事件发布端口。"""

from typing import Protocol

from app.application.events import TradeEvent


class EventPublisher(Protocol):
    """把应用事件交给进程内或跨进程消息适配器。"""

    async def publish(self, event: TradeEvent) -> None: ...
