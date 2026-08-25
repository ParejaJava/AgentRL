"""热路径使用的近似 token 统计工具。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import BaseTool


@dataclass(frozen=True, slots=True)
class ContextTokenUsage:
    """描述 L0/L1/L2 的近似 token 占用。"""

    l0_tokens: int
    l1_tokens: int
    l2_tokens: int
    total_tokens: int
    usage_ratio: float


def count_messages(messages: Sequence[BaseMessage]) -> int:
    """使用 LangChain 的轻量算法估算消息 token。"""

    return int(count_tokens_approximately(messages)) if messages else 0


def count_text(text: str) -> int:
    """估算普通文本 token 数。"""

    return count_messages([{"role": "user", "content": text}])


def count_context(
    *,
    system_message: BaseMessage | None,
    tools: Sequence[BaseTool | dict[str, Any]],
    baseline_messages: Sequence[BaseMessage],
    active_messages: Sequence[BaseMessage],
    context_window_tokens: int,
) -> ContextTokenUsage:
    """分别统计静态根、epoch 基线和动态工作集。"""

    l0_messages = [system_message] if system_message is not None else []
    l0_tokens = int(count_tokens_approximately(l0_messages, tools=list(tools)))
    l1_tokens = count_messages(baseline_messages)
    l2_tokens = count_messages(active_messages)
    total_tokens = l0_tokens + l1_tokens + l2_tokens
    return ContextTokenUsage(
        l0_tokens=l0_tokens,
        l1_tokens=l1_tokens,
        l2_tokens=l2_tokens,
        total_tokens=total_tokens,
        usage_ratio=total_tokens / context_window_tokens,
    )
