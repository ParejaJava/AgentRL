"""启动 Agent Run 并发布会话事件的应用用例。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4

from app.application.events import TradeEvent, TradeEventType
from app.application.ports.event_publisher import EventPublisher
from app.application.runtime import (
    AgentExecutionContext,
    AgentId,
    AgentRun,
    RunId,
    ShoppingContextSnapshot,
    ThreadId,
)


@dataclass(frozen=True, slots=True)
class RunAgentCommand:
    """从任意接口层提交给 Agent 平台的标准运行命令。"""

    message: str
    shopping: ShoppingContextSnapshot
    thread_id: str
    session_dir: str | None = None


class AgentRuntimePort(Protocol):
    """Application 驱动具体 Agent Runtime 所需的最小端口。"""

    def stream(
        self,
        message: str,
        context: AgentExecutionContext,
    ) -> AsyncIterator[dict[str, Any]]: ...


class RunAgent:
    """管理运行生命周期，并把 Runtime 输出转换成稳定应用事件。"""

    def __init__(self, runtime: AgentRuntimePort, events: EventPublisher) -> None:
        self._runtime = runtime
        self._events = events

    async def execute(self, command: RunAgentCommand) -> AsyncIterator[dict[str, Any]]:
        """执行一轮主 Agent，并同时向调用方和 EventBus 发布事件。"""

        message = command.message.strip()
        if not message:
            raise ValueError("message 不能为空")

        run_id = str(uuid4())
        run = AgentRun(
            run_id=RunId(run_id),
            thread_id=ThreadId(command.thread_id),
            agent_id=AgentId("main_agent"),
        )
        context = AgentExecutionContext(
            thread_id=command.thread_id,
            shopping=command.shopping,
            run_id=run_id,
            session_dir=command.session_dir,
        )
        run.start()
        yield await self._publish(
            command.shopping.shopping_session_id,
            TradeEventType.RUN_STARTED,
            {"run_id": run_id, "thread_id": command.thread_id},
        )

        answer_parts: list[str] = []
        try:
            async for runtime_event in self._runtime.stream(message, context):
                content = str(runtime_event.get("content", ""))
                if not content:
                    continue
                answer_parts.append(content)
                yield await self._publish(
                    command.shopping.shopping_session_id,
                    TradeEventType.TOKEN_DELTA,
                    {"run_id": run_id, "content": content},
                )
        except Exception as exc:  # noqa: BLE001 - 用例边界统一发布失败事件。
            run.fail(str(exc))
            yield await self._publish(
                command.shopping.shopping_session_id,
                TradeEventType.ERROR,
                {"run_id": run_id, "message": str(exc)},
            )
            return

        final_answer = "".join(answer_parts)
        yield await self._publish(
            command.shopping.shopping_session_id,
            TradeEventType.FINAL_RESULT,
            {"run_id": run_id, "content": final_answer},
        )
        run.complete()
        yield await self._publish(
            command.shopping.shopping_session_id,
            TradeEventType.RUN_FINISHED,
            {"run_id": run_id, "thread_id": command.thread_id},
        )

    async def _publish(
        self,
        shopping_session_id: str,
        event_type: TradeEventType,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """构造一次事件，只发布一次，并返回同一份传输数据。"""

        event = TradeEvent(
            shopping_session_id=shopping_session_id,
            type=event_type,
            payload=payload,
        )
        await self._events.publish(event)
        return event.to_dict()
