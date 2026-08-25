"""在 Composition Root 使用的强类型环境配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from app.infrastructure.context_governance.config import GovernanceConfig


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    """应用启动时一次性读取的配置快照。"""

    llm_model_name: str
    llm_api_key: str | None
    llm_base_url: str | None
    llm_temperature: float
    compression_llm_model: str
    compression_llm_max_tokens: int
    context_llm_enabled: bool
    sub_agent_max_concurrency: int
    item_index_root: Path
    governance: GovernanceConfig

    @classmethod
    def from_env(cls) -> Settings:
        """加载 `.env` 并校验平台启动配置。"""

        load_dotenv()
        model_name = os.getenv("LLM_MODEL_NAME", "qwen-max")
        settings = cls(
            llm_model_name=model_name,
            llm_api_key=os.getenv("LLM_API_KEY"),
            llm_base_url=os.getenv("LLM_BASE_URL"),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            compression_llm_model=os.getenv("COMPRESSION_LLM_MODEL", model_name),
            compression_llm_max_tokens=int(
                os.getenv("COMPRESSION_LLM_MAX_TOKENS", "2048")
            ),
            context_llm_enabled=_env_bool("CONTEXT_LLM_ENABLED", True),
            sub_agent_max_concurrency=int(os.getenv("SUB_AGENT_MAX_CONCURRENCY", "50")),
            item_index_root=Path(os.getenv("ITEM_INDEX_ROOT", "data/indexes")),
            governance=GovernanceConfig.from_env(),
        )
        if settings.sub_agent_max_concurrency < 1:
            raise ValueError("SUB_AGENT_MAX_CONCURRENCY 必须大于 0")
        if settings.compression_llm_max_tokens < 1:
            raise ValueError("COMPRESSION_LLM_MAX_TOKENS 必须大于 0")
        return settings
