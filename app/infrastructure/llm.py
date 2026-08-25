"""OpenAI-compatible ChatModel 基础设施适配器。"""

from langchain_openai import ChatOpenAI

from .settings import Settings


def create_chat_model(settings: Settings) -> ChatOpenAI:
    """创建主、子 Agent 共用的聊天模型。"""

    return ChatOpenAI(
        model=settings.llm_model_name,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
    )


def create_compression_model(settings: Settings) -> ChatOpenAI:
    """创建零温度、受输出预算约束的上下文压缩模型。"""

    return ChatOpenAI(
        model=settings.compression_llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0,
        max_tokens=settings.compression_llm_max_tokens,
    )


def get_llm() -> ChatOpenAI:
    """兼容入口：从当前环境创建聊天模型。"""

    return create_chat_model(Settings.from_env())


def get_compression_llm() -> ChatOpenAI:
    """兼容入口：从当前环境创建压缩模型。"""

    return create_compression_model(Settings.from_env())
