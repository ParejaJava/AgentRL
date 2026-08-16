"""项目统一使用的大语言模型客户端工厂。"""

import os

from langchain_openai import ChatOpenAI


def get_llm() -> ChatOpenAI:
    """根据环境变量创建供所有 Agent 循环使用的 ChatOpenAI 客户端。"""

    # 使用统一工厂可避免每个 Agent 分别维护模型名称、密钥和服务地址。
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL_NAME", "qwen-max"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=1,
    )
