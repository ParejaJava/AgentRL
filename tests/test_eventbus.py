"""验证应用事件与进程内 EventBus 的会话隔离。"""

import asyncio

from app.application.events import TradeEvent, TradeEventType
from app.infrastructure.eventbus import InMemoryTradeEventBus


def test_eventbus_delivers_only_to_matching_session() -> None:
    """不同 shopping_session_id 的订阅者不能收到彼此事件。"""

    async def scenario() -> None:
        bus = InMemoryTradeEventBus()
        first = bus.subscribe("shopping-1")
        second = bus.subscribe("shopping-2")
        event = TradeEvent(
            shopping_session_id="shopping-1",
            type=TradeEventType.TOKEN_DELTA,
            payload={"content": "你好"},
        )

        await bus.publish(event)

        assert await first.get() == event
        assert second.empty()
        bus.unsubscribe("shopping-1", first)
        bus.unsubscribe("shopping-2", second)

    asyncio.run(scenario())


def test_event_round_trip() -> None:
    """事件通过字典传输后仍保留分区键和类型。"""

    event = TradeEvent(
        shopping_session_id="shopping-1",
        type=TradeEventType.FINAL_RESULT,
        payload={"content": "完成"},
    )

    assert TradeEvent.from_dict(event.to_dict()) == event
