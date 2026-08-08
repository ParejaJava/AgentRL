"""Main Agent loop and execution entry point."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from uuid import uuid4

from .llm import LLMConfig, load_llm_config


@dataclass(slots=True)
class MainAgent:
    """Minimal main-agent facade ready for model and tool integration."""

    llm: LLMConfig = field(default_factory=load_llm_config)

    async def run_agent(self, message: str) -> AsyncIterator[dict[str, str]]:
        """Run one task and stream normalized AG-UI-style events."""

        run_id = str(uuid4())
        yield {"type": "run_started", "run_id": run_id}
        # Replace this echo with the actual LLM/tool loop.
        yield {"type": "text_message_content", "content": message}
        yield {"type": "run_finished", "run_id": run_id}

