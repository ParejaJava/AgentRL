"""Qwen OpenAI-compatible 显式上下文缓存适配器。"""

from __future__ import annotations

from langchain_core.messages import BaseMessage, SystemMessage

from .base import DecoratedPrompt, ImplicitCacheAdapter


def _with_cache_marker(message: BaseMessage) -> BaseMessage:
    """复制消息并在最后一个文本 block 上添加 ephemeral 标记。"""

    if isinstance(message.content, str):
        content = [
            {
                "type": "text",
                "text": message.content,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    else:
        content = [
            dict(block) if isinstance(block, dict) else block
            for block in message.content
        ]
        for block in reversed(content):
            if isinstance(block, dict) and block.get("type") in {"text", "input_text"}:
                block["cache_control"] = {"type": "ephemeral"}
                break
    return message.model_copy(update={"content": content})


class QwenExplicitCacheAdapter(ImplicitCacheAdapter):
    """在 L0 和 L1 末端添加 Qwen `cache_control` 标记。"""

    def decorate_request(
        self,
        *,
        system_message: SystemMessage | None,
        messages: list[BaseMessage],
        model_settings: dict[str, object],
        baseline_message_id: str,
    ) -> DecoratedPrompt:
        """只修改临时请求副本，不污染 Session State。"""

        marked_system = (
            _with_cache_marker(system_message) if system_message is not None else None
        )
        marked_messages = [
            _with_cache_marker(message)
            if str(message.id) == baseline_message_id
            else message
            for message in messages
        ]
        return DecoratedPrompt(marked_system, marked_messages, dict(model_settings))
