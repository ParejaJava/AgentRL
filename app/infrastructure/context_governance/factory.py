"""为主、子 AgentLoop 创建一致的上下文治理 Middleware。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from .compressor import ContextCompressor
from .config import GovernanceConfig
from .middleware import (
    CacheMetricsMiddleware,
    ContextGovernanceMiddleware,
    ToolResultMiddleware,
)


def create_context_middleware(
    *,
    model: BaseChatModel,
    tools: Sequence[BaseTool | dict[str, Any]],
    system_prompt: str,
    agent_id: str,
    config: GovernanceConfig,
    compressor: ContextCompressor | None = None,
) -> list[AgentMiddleware]:
    """创建工具侧防线、上下文治理和缓存指标 Middleware。"""

    model_name = str(
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or type(model).__name__
    )
    return [
        ToolResultMiddleware(config=config, agent_id=agent_id),
        ContextGovernanceMiddleware(
            compressor=compressor,
            config=config,
            agent_id=agent_id,
            static_system_prompt=system_prompt,
            static_tools=tools,
            static_model_name=model_name,
        ),
        CacheMetricsMiddleware(),
    ]
