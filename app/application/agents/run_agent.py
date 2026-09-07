"""启动 Agent Run 并发布会话事件的应用用例。"""

from __future__ import annotations

import json
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
    BudgetTier,
    RunId,
    ShoppingContextSnapshot,
    ThreadId,
    bind_token_budget,
    current_token_budget,
    reset_token_budget,
)
from app.application.safety import SafetyPolicy, UnsafePromptError

from .drift import DriftDetector


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


@dataclass(frozen=True, slots=True)
class AgentCacheLookup:
    """应用层理解的缓存命中结果。"""

    reply: str | None
    eligible: bool
    similarity: float | None = None


class AgentResponseCache(Protocol):
    """RunAgent 使用的安全响应缓存端口。"""

    async def lookup(
        self,
        message: str,
        shopping: ShoppingContextSnapshot,
        thread_id: str,
    ) -> AgentCacheLookup: ...

    async def remember(
        self,
        message: str,
        reply: str,
        shopping: ShoppingContextSnapshot,
        *,
        eligible: bool,
    ) -> None: ...


class RunAgent:
    """管理运行生命周期，并把 Runtime 输出转换成稳定应用事件。"""

    def __init__(
        self,
        runtime: AgentRuntimePort,
        events: EventPublisher,
        safety: SafetyPolicy | None = None,
        response_cache: AgentResponseCache | None = None,
        token_budget_total: int = 0,
        drift_detector: DriftDetector | None = None,
    ) -> None:
        self._runtime = runtime
        self._events = events
        self._safety = safety or SafetyPolicy()
        self._response_cache = response_cache
        self._token_budget_total = max(0, token_budget_total)
        self._drift_detector = drift_detector

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
        if self._drift_detector is not None:
            self._drift_detector.start_turn(
                command.shopping.shopping_session_id,
                message,
            )

        try:
            self._safety.validate_input(message)
        except UnsafePromptError as exc:
            run.fail(str(exc))
            yield await self._publish(
                command.shopping.shopping_session_id,
                TradeEventType.SAFETY_BLOCKED,
                {"run_id": run_id, "message": str(exc)},
            )
            return

        cache_eligible = False
        if self._response_cache is not None:
            cached = await self._response_cache.lookup(
                message,
                command.shopping,
                command.thread_id,
            )
            cache_eligible = bool(cached.eligible)
            if cached.reply:
                yield await self._publish(
                    command.shopping.shopping_session_id,
                    TradeEventType.CACHE_HIT,
                    {
                        "run_id": run_id,
                        "similarity": cached.similarity,
                    },
                )
                yield await self._publish(
                    command.shopping.shopping_session_id,
                    TradeEventType.FINAL_RESULT,
                    {"run_id": run_id, "content": cached.reply},
                )
                run.complete()
                yield await self._publish(
                    command.shopping.shopping_session_id,
                    TradeEventType.RUN_FINISHED,
                    {"run_id": run_id, "thread_id": command.thread_id},
                )
                return

        answer_parts: list[str] = []
        budget_token = bind_token_budget(self._token_budget_total)
        budget_snapshot: tuple[int, int, BudgetTier] | None = None
        try:
            async for runtime_event in self._runtime.stream(message, context):
                runtime_type = str(runtime_event.get("type", ""))
                if runtime_type != "text_message_content":
                    if (
                        runtime_type == "tool_result"
                        and self._drift_detector is not None
                    ):
                        tool_content = str(runtime_event.get("content", ""))
                        self._drift_detector.observe_action(
                            command.shopping.shopping_session_id,
                            (
                                f"{runtime_event.get('tool_name', 'tool')}: "
                                f"{tool_content}"
                            ),
                            result_empty=_tool_result_is_empty(tool_content),
                        )
                    event_type = _runtime_event_type(runtime_event)
                    if event_type is not None:
                        yield await self._publish(
                            command.shopping.shopping_session_id,
                            event_type,
                            {"run_id": run_id, **runtime_event},
                        )
                    continue
                content = self._safety.sanitize_output(
                    str(runtime_event.get("content", ""))
                )
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
        finally:
            budget = current_token_budget()
            if budget is not None:
                budget_snapshot = (budget.used, budget.total, budget.tier)
            reset_token_budget(budget_token)

        if budget_snapshot is not None and budget_snapshot[2] is BudgetTier.FALLBACK:
            yield await self._publish(
                command.shopping.shopping_session_id,
                TradeEventType.BUDGET_EXHAUSTED,
                {
                    "run_id": run_id,
                    "used_tokens": budget_snapshot[0],
                    "token_limit": budget_snapshot[1],
                },
            )

        if self._drift_detector is not None:
            drift = self._drift_detector.check(
                command.shopping.shopping_session_id
            )
            if drift.drifted:
                yield await self._publish(
                    command.shopping.shopping_session_id,
                    TradeEventType.DRIFT_DETECTED,
                    {"run_id": run_id, "reasons": list(drift.reasons)},
                )

        final_answer = "".join(answer_parts)
        if self._response_cache is not None:
            await self._response_cache.remember(
                message,
                final_answer,
                command.shopping,
                eligible=cache_eligible,
            )
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


def _runtime_event_type(event: dict[str, Any]) -> TradeEventType | None:
    """把基础设施运行事件映射成对外稳定协议。"""

    event_type = str(event.get("type", ""))
    if event_type == "tool_invoke":
        if event.get("tool_name") == "fork_sub_agents":
            return TradeEventType.AGENT_DISPATCH
        return TradeEventType.TOOL_INVOKE
    if event_type == "tool_result":
        return TradeEventType.TOOL_RESULT
    if event_type == "context_compressed":
        return TradeEventType.CONTEXT_COMPRESSED
    return None


def _tool_result_is_empty(content: str) -> bool:
    """从工具稳定 JSON 契约识别空召回，不让 DriftDetector 依赖具体工具类。"""

    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(parsed, dict):
        return False
    collections = [
        parsed[field]
        for field in ("items", "results", "bestsellers", "attributes", "price_tiers")
        if isinstance(parsed.get(field), list)
    ]
    if collections:
        return all(not value for value in collections)
    return parsed.get("status") in {"empty", "not_found", "no_results"}
