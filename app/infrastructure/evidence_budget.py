"""受控证据运行共享的模型调用与 Token 硬预算。"""

from __future__ import annotations

import asyncio
import time
from typing import Any


class EvidenceBudgetExceeded(RuntimeError):
    """证据运行达到请求数或已观测 Token 上限。"""


class EvidenceUsageBudget:
    """让 Agent 推理与上下文压缩共享同一份并发安全计数器。"""

    def __init__(
        self,
        *,
        max_total_requests: int = 0,
        max_observed_tokens: int = 0,
        min_interval_seconds: float = 0.0,
    ) -> None:
        self._max_total_requests = max(0, max_total_requests)
        self._max_observed_tokens = max(0, max_observed_tokens)
        self._min_interval_seconds = max(0.0, min_interval_seconds)
        self._lock = asyncio.Lock()
        self._last_started = 0.0
        self._started_requests = 0
        self._completed_requests = 0
        self._observed_input_tokens = 0
        self._observed_output_tokens = 0

    async def reserve_request(self) -> None:
        """在发出外部请求之前原子检查上限并预留一次请求。"""

        async with self._lock:
            observed = self._observed_input_tokens + self._observed_output_tokens
            if (
                self._max_total_requests
                and self._started_requests >= self._max_total_requests
            ):
                raise EvidenceBudgetExceeded("model request limit reached")
            if self._max_observed_tokens and observed >= self._max_observed_tokens:
                raise EvidenceBudgetExceeded("observed token limit reached")
            delay = self._min_interval_seconds - (
                time.monotonic() - self._last_started
            )
            if delay > 0:
                await asyncio.sleep(delay)
            self._last_started = time.monotonic()
            self._started_requests += 1

    async def record_usage(self, *, input_tokens: int, output_tokens: int) -> None:
        """记录一次成功响应；供应商缺失 usage 时仍记录完成次数。"""

        async with self._lock:
            self._completed_requests += 1
            self._observed_input_tokens += max(0, input_tokens)
            self._observed_output_tokens += max(0, output_tokens)

    def snapshot(self) -> dict[str, int]:
        """返回只读计数快照，字段与证据报告模型保持一致。"""

        input_tokens = self._observed_input_tokens
        output_tokens = self._observed_output_tokens
        return {
            "started_requests": self._started_requests,
            "completed_requests": self._completed_requests,
            "observed_tokens": input_tokens + output_tokens,
            "observed_input_tokens": input_tokens,
            "observed_output_tokens": output_tokens,
        }


def message_usage(message: Any) -> tuple[int, int]:
    """从不同 ChatModel 的消息对象中提取输入与输出 Token。"""

    usage = getattr(message, "usage_metadata", None) or {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    if input_tokens or output_tokens:
        return input_tokens, output_tokens
    response_metadata = getattr(message, "response_metadata", None) or {}
    token_usage = response_metadata.get("token_usage", {})
    return (
        int(token_usage.get("prompt_tokens", 0) or 0),
        int(token_usage.get("completion_tokens", 0) or 0),
    )
