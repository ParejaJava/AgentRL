"""LangFuse v3/v4 LangChain Callback 工厂。"""

from __future__ import annotations

from typing import Any

from app.application.runtime import AgentExecutionContext


class LangfuseCallbacks:
    """按运行创建 CallbackHandler；未配置时返回空列表并保持零副作用。"""

    def __init__(self, *, enabled: bool) -> None:
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def create(self, context: AgentExecutionContext, agent_id: str) -> list[Any]:
        """延迟导入 SDK，避免基础安装强依赖 LangFuse。"""

        if not self._enabled:
            return []
        try:
            from langfuse.langchain import CallbackHandler
        except ImportError as exc:
            raise RuntimeError(
                "已启用 LangFuse；请运行 `uv sync --extra platform`"
            ) from exc
        del context, agent_id
        return [CallbackHandler()]

    def metadata(
        self,
        context: AgentExecutionContext,
        agent_id: str,
    ) -> dict[str, object]:
        """生成 LangFuse 官方识别的 trace 属性和平台诊断字段。"""

        if not self._enabled:
            return {}
        shopping = context.shopping
        return {
            "langfuse_user_id": shopping.buyer_id if shopping else "system",
            "langfuse_session_id": (
                shopping.shopping_session_id if shopping else context.thread_id
            ),
            "langfuse_tags": ["globex-agent", agent_id],
            "run_id": context.run_id or "",
            "thread_id": context.thread_id,
            "agent_id": agent_id,
        }
