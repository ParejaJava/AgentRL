"""验证应用用例管理 Run 生命周期并发布统一事件。"""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from app.application.agents import RunAgent, RunAgentCommand
from app.application.runtime import AgentExecutionContext, ShoppingContextSnapshot
from app.infrastructure.eventbus import InMemoryTradeEventBus


class FakeRuntime:
    """不依赖 LangGraph 的 Runtime 端口测试替身。"""

    def __init__(self) -> None:
        self.contexts: list[AgentExecutionContext] = []

    async def stream(
        self,
        message: str,
        context: AgentExecutionContext,
    ) -> AsyncIterator[dict[str, Any]]:
        self.contexts.append(context)
        yield {"type": "text_message_content", "content": f"收到：{message}"}


def test_run_agent_publishes_session_scoped_events() -> None:
    """同一轮事件必须共享 shopping_session_id、thread_id 和 run_id。"""

    async def scenario() -> None:
        runtime = FakeRuntime()
        bus = InMemoryTradeEventBus()
        queue = bus.subscribe("shopping-1")
        use_case = RunAgent(runtime, bus)
        command = RunAgentCommand(
            message="搜索旅行包",
            shopping=ShoppingContextSnapshot("shopping-1", "buyer-1"),
            thread_id="thread-1",
        )

        emitted = [event async for event in use_case.execute(command)]
        published = [await queue.get() for _ in emitted]

        assert [event["type"] for event in emitted] == [
            "run.started",
            "token.delta",
            "final.result",
            "run.finished",
        ]
        assert all(
            event.shopping_session_id == "shopping-1" for event in published
        )
        run_ids = {event.payload["run_id"] for event in published}
        assert len(run_ids) == 1
        assert runtime.contexts[0].thread_id == "thread-1"
        assert runtime.contexts[0].shopping == command.shopping

    asyncio.run(scenario())
