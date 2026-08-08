"""Language-model initialization helpers."""

from dataclasses import dataclass
from os import getenv


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Runtime model configuration loaded from environment variables."""

    model: str = "gpt-4.1-mini"
    api_key: str | None = None
    base_url: str | None = None


def load_llm_config() -> LLMConfig:
    """Load model configuration without creating a provider-specific client."""

    return LLMConfig(
        model=getenv("LLM_MODEL", "gpt-4.1-mini"),
        api_key=getenv("OPENAI_API_KEY"),
        base_url=getenv("OPENAI_BASE_URL"),
    )

