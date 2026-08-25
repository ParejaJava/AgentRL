"""模型供应商缓存适配器的公共接口。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage


@dataclass(frozen=True, slots=True)
class DecoratedPrompt:
    """保存添加 provider 缓存信息后的模型请求字段。"""

    system_message: SystemMessage | None
    messages: list[BaseMessage]
    model_settings: dict[str, object]


@dataclass(frozen=True, slots=True)
class ProviderUsage:
    """统一不同供应商返回的 token/cache 指标。"""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0


class CacheProviderAdapter(Protocol):
    """隔离不同模型供应商的 cache marker 和 usage 格式。"""

    def decorate_request(
        self,
        *,
        system_message: SystemMessage | None,
        messages: list[BaseMessage],
        model_settings: dict[str, object],
        baseline_message_id: str,
    ) -> DecoratedPrompt:
        """向临时 Prompt Projection 添加 provider-specific 标记。"""

    def extract_metrics(self, message: AIMessage) -> ProviderUsage:
        """从模型消息中提取统一的 token 和缓存指标。"""


class ImplicitCacheAdapter:
    """依赖公共前缀稳定性和供应商隐式缓存，不修改请求内容。"""

    def decorate_request(
        self,
        *,
        system_message: SystemMessage | None,
        messages: list[BaseMessage],
        model_settings: dict[str, object],
        baseline_message_id: str,
    ) -> DecoratedPrompt:
        """原样返回请求，稳定性由 Prompt Projection 保证。"""

        del baseline_message_id
        return DecoratedPrompt(system_message, messages, dict(model_settings))

    def extract_metrics(self, message: AIMessage) -> ProviderUsage:
        """兼容 LangChain usage_metadata 和 OpenAI response_metadata。"""

        usage = message.usage_metadata or {}
        input_details = usage.get("input_token_details") or {}
        response_usage = message.response_metadata.get("token_usage") or {}
        prompt_details = response_usage.get("prompt_tokens_details") or {}
        return ProviderUsage(
            input_tokens=int(
                usage.get("input_tokens", response_usage.get("prompt_tokens", 0)) or 0
            ),
            output_tokens=int(
                usage.get("output_tokens", response_usage.get("completion_tokens", 0))
                or 0
            ),
            cached_input_tokens=int(
                input_details.get("cache_read", prompt_details.get("cached_tokens", 0))
                or 0
            ),
            cache_write_tokens=int(input_details.get("cache_creation", 0) or 0),
        )
