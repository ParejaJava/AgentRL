"""OpenAI-compatible ChatModel 基础设施适配器。"""

from typing import Any

from langchain_openai import ChatOpenAI

from .settings import Settings


def _chat_model_kwargs(
    settings: Settings,
    *,
    model_name: str,
    temperature: float,
    max_tokens: int | None = None,
    disable_thinking: bool = False,
) -> dict[str, Any]:
    """生成兼容普通模型与只允许默认温度的推理模型参数。"""

    kwargs: dict[str, Any] = {
        "model": model_name,
        "api_key": settings.llm_api_key,
        "base_url": settings.llm_base_url,
    }
    normalized = model_name.lower()
    reasoning_prefixes = ("gpt-5", "o1", "o3", "o4", "kimi-k2")
    if not normalized.startswith(reasoning_prefixes):
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if disable_thinking and normalized.startswith("kimi-k2"):
        # Kimi K2.5/K2.6 官方 OpenAI 兼容接口使用此字段切换 Instant 模式。
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    return kwargs


def create_chat_model(
    settings: Settings,
    *,
    model_name: str | None = None,
    disable_thinking: bool = False,
) -> ChatOpenAI:
    """创建聊天模型；受控实验可关闭 Kimi 思考模式以节省预算。"""

    resolved_model = model_name or settings.llm_model_name
    return ChatOpenAI(
        **_chat_model_kwargs(
            settings,
            model_name=resolved_model,
            temperature=settings.llm_temperature,
            disable_thinking=disable_thinking,
        )
    )


def create_compression_model(settings: Settings) -> ChatOpenAI:
    """创建零温度、受输出预算约束的上下文压缩模型。"""

    return ChatOpenAI(
        **_chat_model_kwargs(
            settings,
            model_name=settings.compression_llm_model,
            temperature=0,
            max_tokens=settings.compression_llm_max_tokens,
            disable_thinking=True,
        )
    )


def create_category_structuring_model(settings: Settings) -> ChatOpenAI:
    """创建只供离线品类知识摄取使用的零温度结构化模型。"""

    return ChatOpenAI(
        **_chat_model_kwargs(
            settings,
            model_name=settings.category_structuring_model,
            temperature=0,
            max_tokens=settings.category_structuring_max_tokens,
            disable_thinking=True,
        )
    )


def get_llm() -> ChatOpenAI:
    """兼容入口：从当前环境创建聊天模型。"""

    return create_chat_model(Settings.from_env())


def get_compression_llm() -> ChatOpenAI:
    """兼容入口：从当前环境创建压缩模型。"""

    return create_compression_model(Settings.from_env())
