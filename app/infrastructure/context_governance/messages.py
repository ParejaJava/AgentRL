"""LangChain 消息的规范化、分组和保护规则。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.messages.base import message_to_dict


def canonical_json(value: Any) -> str:
    """生成稳定 JSON，供哈希、事件记录和缓存校验使用。"""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def message_payload(message: BaseMessage) -> dict[str, Any]:
    """把 LangChain 消息转换为可稳定序列化的字典。"""

    return message_to_dict(message)


def message_event_id(
    thread_id: str,
    position: int,
    message: BaseMessage,
) -> str:
    """根据线程、位置和内容生成可重复计算的事件 ID。"""

    payload = canonical_json(message_payload(message))
    digest = hashlib.sha256(f"{thread_id}:{position}:{payload}".encode()).hexdigest()
    return f"msg-{digest[:24]}"


def message_text(message: BaseMessage) -> str:
    """安全提取消息中的文本内容。"""

    try:
        return message.text
    except (AttributeError, ValueError, TypeError):
        content = message.content
        if isinstance(content, str):
            return content
        return canonical_json(content)


@dataclass(frozen=True, slots=True)
class IndexedMessage:
    """将消息与位置和事件 ID 绑定。"""

    position: int
    event_id: str
    message: BaseMessage


def index_messages(
    thread_id: str,
    messages: Sequence[BaseMessage],
) -> list[IndexedMessage]:
    """为当前消息列表生成稳定索引。"""

    return [
        IndexedMessage(
            position=position,
            event_id=message_event_id(thread_id, position, message),
            message=message,
        )
        for position, message in enumerate(messages)
    ]


def collect_protected_event_ids(
    indexed: Sequence[IndexedMessage],
    *,
    hot_message_count: int,
) -> set[str]:
    """识别 D1 热上下文，并扩展保护完整工具调用组。"""

    if not indexed:
        return set()

    protected_positions = set(
        range(max(0, len(indexed) - hot_message_count), len(indexed))
    )

    # 当前用户请求永远属于 D1；它之后只有最近消息和未闭合工具组必须保留。
    # 这样单次长 AgentLoop 中较早、已经闭合的工具循环仍能被压缩。
    last_user_position = max(
        (item.position for item in indexed if isinstance(item.message, HumanMessage)),
        default=len(indexed) - 1,
    )
    protected_positions.add(last_user_position)

    tool_call_to_ai_position: dict[str, int] = {}
    tool_call_to_result_position: dict[str, int] = {}
    for item in indexed:
        if isinstance(item.message, AIMessage):
            for call in item.message.tool_calls:
                call_id = str(call.get("id", ""))
                if call_id:
                    tool_call_to_ai_position[call_id] = item.position
        elif isinstance(item.message, ToolMessage):
            tool_call_to_result_position[item.message.tool_call_id] = item.position

    # 如果工具组任意一侧处于热区，或者调用尚未闭合，则整组进入保护集合。
    for call_id, ai_position in tool_call_to_ai_position.items():
        result_position = tool_call_to_result_position.get(call_id)
        is_open = result_position is None
        touches_hot_context = ai_position in protected_positions or (
            result_position is not None and result_position in protected_positions
        )
        if is_open or touches_hot_context:
            protected_positions.add(ai_position)
            if result_position is not None:
                protected_positions.add(result_position)

    return {item.event_id for item in indexed if item.position in protected_positions}


def validate_tool_pairs(messages: Sequence[BaseMessage]) -> bool:
    """校验所有 ToolMessage 都能找到对应的 AI tool call。"""

    announced: set[str] = set()
    for message in messages:
        if isinstance(message, AIMessage):
            announced.update(
                str(call.get("id", "")) for call in message.tool_calls if call.get("id")
            )
        elif isinstance(message, ToolMessage) and message.tool_call_id not in announced:
            return False
    return True
