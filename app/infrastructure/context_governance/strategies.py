"""确定性消息清理、候选选择和结构化状态合并策略。"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from .messages import IndexedMessage, message_text
from .schemas import CompressionDelta, TaskState, ToolResultEnvelope
from .token_counter import count_messages, count_text


def select_summary_candidates(
    indexed: Sequence[IndexedMessage],
    *,
    protected_event_ids: set[str],
    compressed_event_ids: set[str],
    max_tokens: int,
) -> list[IndexedMessage]:
    """从最旧历史开始选择完整、闭合且允许压缩的消息组。"""

    by_position = {item.position: item for item in indexed}
    tool_results: dict[str, IndexedMessage] = {
        item.message.tool_call_id: item
        for item in indexed
        if isinstance(item.message, ToolMessage)
    }
    selected: list[IndexedMessage] = []
    selected_ids: set[str] = set()
    consumed_tokens = 0

    for item in indexed:
        if item.event_id in protected_event_ids | compressed_event_ids | selected_ids:
            continue

        group = [item]
        if isinstance(item.message, AIMessage) and item.message.tool_calls:
            result_messages = []
            for call in item.message.tool_calls:
                call_id = str(call.get("id", ""))
                result = tool_results.get(call_id)
                if result is None:
                    result_messages = []
                    break
                result_messages.append(result)
            if not result_messages:
                continue
            group.extend(result_messages)
        elif isinstance(item.message, ToolMessage):
            # ToolMessage 只能随声明它的 AIMessage 一起进入候选集合。
            announcing = next(
                (
                    candidate
                    for candidate in by_position.values()
                    if isinstance(candidate.message, AIMessage)
                    and any(
                        str(call.get("id", "")) == item.message.tool_call_id
                        for call in candidate.message.tool_calls
                    )
                ),
                None,
            )
            if announcing is None or announcing.event_id not in selected_ids:
                continue

        group_ids = {candidate.event_id for candidate in group}
        if group_ids & protected_event_ids:
            continue
        group_tokens = count_messages([candidate.message for candidate in group])
        if selected and consumed_tokens + group_tokens > max_tokens:
            break
        selected.extend(
            candidate for candidate in group if candidate.event_id not in selected_ids
        )
        selected_ids.update(group_ids)
        consumed_tokens += group_tokens

    return sorted(selected, key=lambda item: item.position)


def clear_old_tool_results(
    indexed: Sequence[IndexedMessage],
    *,
    protected_event_ids: set[str],
) -> list[BaseMessage]:
    """把旧 ToolMessage 原文替换成可从 Event Log 恢复的轻量信封。"""

    governed: list[BaseMessage] = []
    for item in indexed:
        message = item.message
        if (
            isinstance(message, ToolMessage)
            and item.event_id not in protected_event_ids
        ):
            original_tokens = count_text(message_text(message))
            envelope = ToolResultEnvelope(
                status=message.status,
                summary="旧工具结果已从活动上下文清理，可从 Event Log 恢复。",
                preview=None,
                artifact_ref=f"event:{item.event_id}",
                original_token_count=original_tokens,
                visible_token_count=0,
            )
            content = json.dumps(envelope.model_dump(mode="json"), ensure_ascii=False)
            envelope.visible_token_count = count_text(content)
            governed.append(
                message.model_copy(
                    update={
                        "content": json.dumps(
                            envelope.model_dump(mode="json"), ensure_ascii=False
                        )
                    }
                )
            )
        else:
            governed.append(message)
    return governed


def apply_compression_delta(
    *,
    task_state: TaskState,
    working_memory: dict[str, dict[str, Any]],
    delta: CompressionDelta,
    allowed_cold_event_ids: set[str],
    protected_event_ids: set[str],
) -> tuple[TaskState, dict[str, dict[str, Any]], list[str]]:
    """校验并原子应用压缩模型返回的 D2/D3/D4 增量。"""

    requested_cold = set(delta.cold_event_ids)
    if requested_cold - allowed_cold_event_ids:
        raise ValueError("压缩模型返回了候选集合之外的 cold_event_ids")
    if requested_cold & protected_event_ids:
        raise ValueError("压缩模型尝试归档 D1 保护事件")

    allowed_task_fields = set(TaskState.model_fields)
    if set(delta.task_state_patch) - allowed_task_fields:
        raise ValueError("task_state_patch 包含未知字段")
    merged_task = task_state.model_copy(update=delta.task_state_patch)
    merged_task = TaskState.model_validate(merged_task.model_dump())

    merged_memory = dict(working_memory)
    for memory_id in delta.working_memory_delete_ids:
        merged_memory.pop(memory_id, None)
    for item in delta.working_memory_upserts:
        merged_memory[item.id] = item.model_dump(mode="json")

    return merged_task, merged_memory, sorted(requested_cold)
