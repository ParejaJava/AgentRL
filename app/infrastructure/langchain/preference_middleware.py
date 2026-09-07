"""在每次模型调用前把当前买家的长期偏好投影为动态工作记忆。"""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware

from app.application.memory import PreferenceSelector, PreferenceStore
from app.infrastructure.context_governance.runtime import resolve_request_context
from app.infrastructure.context_governance.schemas import (
    SessionAgentState,
    WorkingMemoryItem,
)

logger = logging.getLogger(__name__)

_MEMORY_PREFIX = "buyer-preference:"


class PreferenceMemoryMiddleware(AgentMiddleware):
    """读取跨会话偏好，并将其放入可更新的 L2 工作记忆。"""

    state_schema = SessionAgentState

    def __init__(
        self,
        store: PreferenceStore,
        selector: PreferenceSelector,
    ) -> None:
        self._store = store
        self._selector = selector

    async def abefore_model(
        self,
        state: SessionAgentState,
        runtime: Any,
    ) -> dict[str, Any] | None:
        """按可信 buyer_id 刷新偏好项；读取失败时不阻断 AgentLoop。"""

        context = resolve_request_context(runtime)
        if context.shopping is None:
            return None
        try:
            preferences = await self._store.list_by_buyer(
                context.shopping.buyer_id
            )
        except Exception:
            logger.warning("读取买家长期偏好失败，本轮按无偏好继续", exc_info=True)
            return None

        working_memory = {
            key: value
            for key, value in dict(state.get("working_memory", {})).items()
            if not key.startswith(_MEMORY_PREFIX)
        }
        for preference in self._selector.select(preferences):
            memory_id = f"{_MEMORY_PREFIX}{preference.preference_id}"
            working_memory[memory_id] = WorkingMemoryItem(
                id=memory_id,
                kind="preference",
                content=f"[{preference.kind.value}] {preference.statement}",
            ).model_dump(mode="json")
        return {"working_memory": working_memory}
