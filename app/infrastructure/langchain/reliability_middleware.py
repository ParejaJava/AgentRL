"""模型网关与工具调用的运行时韧性中间件。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.types import Command

from app.application.runtime import BudgetTier, current_token_budget
from app.infrastructure.context import require_context
from app.infrastructure.evidence_budget import (
    EvidenceBudgetExceeded,
    EvidenceUsageBudget,
)
from app.infrastructure.resilience import SharedCircuitBreaker

__all__ = [
    "EvidenceBudgetExceeded",
    "ModelAttemptBudgetMiddleware",
    "ModelGatewayMiddleware",
    "ToolHarnessMiddleware",
    "ToolResilienceConfig",
    "ToolResilienceMiddleware",
]


@dataclass(frozen=True, slots=True)
class ToolResilienceConfig:
    """工具超时、熔断和循环保护参数。"""

    default_timeout_seconds: float = 30.0
    timeout_by_tool: Mapping[str, float] | None = None
    failure_threshold: int = 3
    recovery_seconds: float = 30.0
    repeated_call_limit: int = 3


class ModelGatewayMiddleware(AgentMiddleware):
    """在模型边界统一执行并发、请求间隔、重试与备用模型回退。"""

    def __init__(
        self,
        *,
        max_concurrency: int,
        min_interval_seconds: float,
        max_retries: int,
        max_total_requests: int = 0,
        max_observed_tokens: int = 0,
        evidence_budget: EvidenceUsageBudget | None = None,
        fallback_model: BaseChatModel | None = None,
        lite_model: BaseChatModel | None = None,
        semaphore: asyncio.Semaphore | None = None,
        external_attempt_accounting: bool = False,
    ) -> None:
        self._semaphore = (
            semaphore if semaphore is not None else asyncio.Semaphore(max_concurrency)
        )
        self._max_retries = max(0, max_retries)
        self._evidence_budget = evidence_budget or EvidenceUsageBudget(
            max_total_requests=max_total_requests,
            max_observed_tokens=max_observed_tokens,
            min_interval_seconds=min_interval_seconds,
        )
        self._fallback = fallback_model
        self._lite = lite_model
        self._external_attempt_accounting = external_attempt_accounting

    def usage_snapshot(self) -> dict[str, int]:
        """返回进程内模型网关计数，供受控评测和运行诊断使用。"""

        return self._evidence_budget.snapshot()

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """只对可重试错误退避；主模型耗尽后最多调用一次备用模型。"""

        async with self._semaphore:
            budget = current_token_budget()
            tier = budget.tier if budget is not None else BudgetTier.MAIN
            if tier is BudgetTier.FALLBACK:
                return ModelResponse(
                    result=[
                        AIMessage(
                            content=(
                                "本次任务的模型 Token 预算已耗尽，已停止继续推理。"
                                "请缩小任务范围或提高 TOKEN_BUDGET_TOTAL。"
                            )
                        )
                    ]
                )
            selected = request
            if tier in {BudgetTier.LITE, BudgetTier.MINIMAL} and self._lite:
                selected = selected.override(model=self._lite)
            if tier is BudgetTier.MINIMAL:
                current_system = selected.system_message
                prefix = current_system.text if current_system is not None else ""
                selected = selected.override(
                    system_message=SystemMessage(
                        content=(
                            f"{prefix}\n\n当前预算接近上限：只给结论与最多三条依据，"
                            "不要展开推理，不调用非必要工具。"
                        ).strip()
                    )
                )
            last_error: Exception | None = None
            for attempt in range(self._max_retries + 1):
                await self._reserve_request()
                try:
                    response = await handler(selected)
                    await self._record_response(response, "model")
                    return response
                except Exception as exc:  # noqa: BLE001 - 需兼容多供应商异常。
                    last_error = exc
                    if attempt >= self._max_retries or not _is_retryable(exc):
                        break
                    await asyncio.sleep((2**attempt) * 0.25 + random.uniform(0, 0.1))
            if self._fallback is not None:
                await self._reserve_request()
                response = await handler(selected.override(model=self._fallback))
                await self._record_response(response, "fallback_model")
                return response
            assert last_error is not None
            raise last_error

    async def _reserve_request(self) -> None:
        """原子检查证据硬上限、执行请求间隔并预留一个模型请求。"""

        if not self._external_attempt_accounting:
            await self._evidence_budget.reserve_request()

    async def _record_response(self, response: ModelResponse, source: str) -> None:
        if not self._external_attempt_accounting:
            input_tokens, output_tokens = _response_usage(response)
            await self._evidence_budget.record_usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            _charge_budget(response, source)


class ModelAttemptBudgetMiddleware(AgentMiddleware):
    """Count each physical attempt, including retries inside context governance.

    Install after governance and pair with a gateway configured for external
    attempt accounting; the gateway still controls concurrency and model routing.
    """

    def __init__(self, budget: EvidenceUsageBudget) -> None:
        self._budget = budget

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        await self._budget.reserve_request()
        response = await handler(request)
        input_tokens, output_tokens = _response_usage(response)
        await self._budget.record_usage(
            input_tokens=input_tokens, output_tokens=output_tokens
        )
        _charge_budget(response, "model_attempt")
        return response


class ToolResilienceMiddleware(AgentMiddleware):
    """防止慢工具、连续失败工具和模型工具调用循环拖垮 AgentLoop。"""

    def __init__(
        self,
        config: ToolResilienceConfig | None = None,
        shared_breaker: SharedCircuitBreaker | None = None,
    ) -> None:
        self._config = config or ToolResilienceConfig()
        self._failures: dict[str, int] = defaultdict(int)
        self._opened_at: dict[str, float] = {}
        self._recent: dict[str, deque[str]] = defaultdict(
            lambda: deque(maxlen=self._config.repeated_call_limit)
        )
        self._lock = asyncio.Lock()
        self._shared_breaker = shared_breaker

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """以结构化错误返回超时、熔断和重复调用，不伪造工具成功。"""

        tool_name = request.tool.name if request.tool is not None else "unknown"
        signature = _tool_signature(tool_name, request.tool_call)
        thread_id = require_context().thread_id
        async with self._lock:
            opened = self._opened_at.get(tool_name)
            if (
                opened is not None
                and time.monotonic() - opened < self._config.recovery_seconds
            ):
                return _tool_error(request, tool_name, "circuit_open", "工具熔断中")
            recent = self._recent[thread_id]
            if list(recent).count(signature) >= self._config.repeated_call_limit - 1:
                return _tool_error(
                    request,
                    tool_name,
                    "repeated_call_blocked",
                    "相同参数的工具调用重复次数过多",
                )
            recent.append(signature)

        if self._shared_breaker is not None:
            allowed = await self._shared_breaker.allow(tool_name)
            if not allowed:
                return _tool_error(
                    request,
                    tool_name,
                    "shared_circuit_open",
                    "工具在其他服务实例中已进入熔断冷却期",
                )

        timeout = (self._config.timeout_by_tool or {}).get(
            tool_name,
            self._config.default_timeout_seconds,
        )
        try:
            result = await asyncio.wait_for(handler(request), timeout=timeout)
        except TimeoutError:
            await self._record_failure(tool_name)
            return _tool_error(request, tool_name, "timeout", f"工具超过 {timeout:g}s")
        except Exception:
            await self._record_failure(tool_name)
            raise
        async with self._lock:
            self._failures[tool_name] = 0
            self._opened_at.pop(tool_name, None)
        if self._shared_breaker is not None:
            await self._shared_breaker.record_success(tool_name)
        return result

    async def _record_failure(self, tool_name: str) -> None:
        async with self._lock:
            self._failures[tool_name] += 1
            if self._failures[tool_name] >= self._config.failure_threshold:
                self._opened_at[tool_name] = time.monotonic()
        if self._shared_breaker is not None:
            await self._shared_breaker.record_failure(tool_name)


class ToolHarnessMiddleware(AgentMiddleware):
    """在工具边界执行顺序、Schema 和工具内容注入检查。"""

    _PREREQUISITES: ClassVar[dict[str, tuple[str, ...]]] = {
        "create_order_intent": ("item_search",),
        "cancel_order": ("query_order",),
    }
    _REQUIRED_FIELDS: ClassVar[dict[str, tuple[str, ...]]] = {
        "item_search": ("recall_strategy", "items"),
        "category_insight": ("status", "category"),
        "create_order_intent": ("status", "order"),
        "query_order": ("order_id", "status"),
        "cancel_order": ("status", "order"),
        "web_search": ("query", "results"),
    }
    _CONTENT_INJECTION = re.compile(
        r"(?is)(ignore\s+(?:all\s+)?previous\s+instructions|"
        r"忽略.{0,20}(?:之前|以上).{0,20}(?:指令|提示词)|"
        r"reveal.{0,30}(?:system prompt|api key))"
    )

    def __init__(self) -> None:
        self._called: dict[str, list[str]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """写工具前检查证据链，执行后校验并净化模型可见结果。"""

        tool_name = request.tool.name if request.tool is not None else "unknown"
        context = require_context()
        scope = (
            context.shopping.shopping_session_id
            if context.shopping is not None
            else context.thread_id
        )
        async with self._lock:
            history = self._called[scope]
            missing = [
                name
                for name in self._PREREQUISITES.get(tool_name, ())
                if name not in history
            ]
            # 进程重启后中间件历史可能为空；此时降级为告警，避免误杀已由
            # LangGraph Checkpointer 恢复的合法链路。已有本轮证据但顺序错误才硬拒绝。
            prerequisite_warning = ""
            if missing and history:
                return _tool_error(
                    request,
                    tool_name,
                    "prerequisite_missing",
                    f"执行 {tool_name} 前必须先执行 {missing[0]} 建立可信业务事实",
                )
            if missing:
                prerequisite_warning = (
                    f"未观察到前置工具 {missing[0]}；可能来自恢复会话，"
                    "只能依赖工具自身的领域校验"
                )
            history.append(tool_name)

        result = await handler(request)
        if not isinstance(result, ToolMessage):
            return result
        original = result.text
        cleaned, injection_count = self._CONTENT_INJECTION.subn(
            "[FILTERED_TOOL_INSTRUCTION]",
            original,
        )
        warnings: list[str] = [prerequisite_warning] if prerequisite_warning else []
        if injection_count:
            warnings.append("工具结果中的疑似提示词注入已过滤")
        required = self._REQUIRED_FIELDS.get(tool_name)
        if required and result.status != "error":
            try:
                parsed = json.loads(cleaned)
            except (json.JSONDecodeError, TypeError):
                warnings.append("工具返回不是合法 JSON，不得据此编造数据")
            else:
                if not isinstance(parsed, dict):
                    warnings.append("工具返回不是 JSON 对象，不得据此编造数据")
                    parsed = {}
                business_status = parsed.get("status")
                expected_error = business_status in {
                    "confirmation_required",
                    "needs_clarification",
                    "unavailable",
                    "invalid_request",
                    "not_found",
                    "error",
                }
                missing_fields = (
                    []
                    if expected_error
                    else [field for field in required if field not in parsed]
                )
                if missing_fields:
                    warnings.append(
                        f"工具返回缺少字段：{', '.join(missing_fields)}，不得编造"
                    )
                if warnings and isinstance(parsed, dict):
                    parsed["harness_warnings"] = warnings
                    cleaned = json.dumps(parsed, ensure_ascii=False)
                    warnings = []
        if warnings:
            cleaned = json.dumps(
                {
                    "status": "error",
                    "error_code": "harness_validation_failed",
                    "raw_preview": cleaned[:500],
                    "warnings": warnings,
                },
                ensure_ascii=False,
            )
        return result.model_copy(update={"content": cleaned})


def _tool_signature(tool_name: str, tool_call: dict[str, Any]) -> str:
    raw = json.dumps(tool_call.get("args", {}), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(f"{tool_name}:{raw}".encode()).hexdigest()


def _tool_error(
    request: ToolCallRequest,
    tool_name: str,
    code: str,
    message: str,
) -> ToolMessage:
    return ToolMessage(
        content=json.dumps(
            {"status": "error", "error_code": code, "message": message},
            ensure_ascii=False,
        ),
        tool_call_id=str(request.tool_call.get("id", "unknown")),
        name=tool_name,
        status="error",
    )


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status in {408, 409, 429, 500, 502, 503, 504}:
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in text
        for marker in ("timeout", "temporarily", "connection", "rate limit")
    )


def _response_usage(response: ModelResponse) -> tuple[int, int]:
    """兼容不同模型供应商的 usage 字段，汇总输入与输出 Token。"""

    input_tokens = 0
    output_tokens = 0
    for message in response.result:
        usage = getattr(message, "usage_metadata", None)
        if not isinstance(usage, Mapping):
            continue
        message_input = int(usage.get("input_tokens", 0) or 0)
        message_output = int(usage.get("output_tokens", 0) or 0)
        if not message_input and not message_output and usage.get("total_tokens"):
            # 无法拆分时保守记入输入侧，确保总预算仍然有效。
            message_input = int(usage["total_tokens"] or 0)
        input_tokens += message_input
        output_tokens += message_output
    return input_tokens, output_tokens


def _response_token_count(response: ModelResponse) -> int:
    """返回一次响应的输入与输出 Token 总和。"""

    return sum(_response_usage(response))


def _charge_budget(response: ModelResponse, source: str) -> None:
    """把一次模型响应的真实用量记入当前意图账本。"""

    budget = current_token_budget()
    if budget is None:
        return
    total = _response_token_count(response)
    budget.charge(source, total)
