"""根据 L0/L1/L2 构造单次模型调用的临时 Prompt Projection。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage

from .breakpoint import CacheBreakpointManager
from .messages import IndexedMessage, canonical_json, validate_tool_pairs
from .schemas import EpochBaseline, TaskDelta, TaskState


@dataclass(frozen=True, slots=True)
class PromptProjection:
    """保存构造完成的模型消息和各层边界。"""

    messages: list[BaseMessage]
    baseline_message_id: str


class PromptProjectionBuilder:
    """将稳定基线、结构化增量、工作记忆和 D1 消息组装成 Prompt。"""

    def __init__(self, breakpoint_manager: CacheBreakpointManager) -> None:
        self._breakpoint_manager = breakpoint_manager

    def build(
        self,
        *,
        epoch: int,
        baseline: EpochBaseline,
        task_state: TaskState,
        task_delta: TaskDelta,
        working_memory: dict[str, dict[str, Any]],
        indexed_messages: Sequence[IndexedMessage],
        compressed_event_ids: set[str],
        governed_messages: Sequence[BaseMessage] | None = None,
    ) -> PromptProjection:
        """生成不修改 LangGraph 原始 messages 的临时投影视图。"""

        baseline_message = self._breakpoint_manager.baseline_message(baseline, epoch)
        dynamic_context = SystemMessage(
            id=f"active-context-{epoch}",
            content=(
                "<active_context>\n"
                + canonical_json(
                    {
                        "task_state": task_state.model_dump(mode="json"),
                        "task_delta": task_delta.model_dump(mode="json"),
                        "working_memory": list(working_memory.values()),
                    }
                )
                + "\n</active_context>"
            ),
        )

        source_messages = (
            list(governed_messages)
            if governed_messages
            else [item.message for item in indexed_messages]
        )
        active_messages = [
            message
            for item, message in zip(indexed_messages, source_messages, strict=True)
            if item.event_id not in compressed_event_ids
        ]
        projection = [baseline_message, dynamic_context, *active_messages]
        if not validate_tool_pairs(projection):
            raise ValueError("Prompt Projection 中 Tool Call/ToolMessage 配对损坏")
        return PromptProjection(
            messages=projection,
            baseline_message_id=str(baseline_message.id),
        )
