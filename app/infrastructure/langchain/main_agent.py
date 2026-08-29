"""主 AgentLoop、工具装配和流式执行入口。"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any
from uuid import uuid4

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from app.application.runtime import AgentExecutionContext
from app.infrastructure.context import (
    current_execution_context,
    reset_context,
    set_context,
)
from app.infrastructure.context_governance.compressor import ContextCompressor
from app.infrastructure.context_governance.config import GovernanceConfig
from app.infrastructure.context_governance.factory import create_context_middleware
from app.infrastructure.context_governance.schemas import SessionAgentState

from .prompts import SYSTEM_PROMPT
from .sub_agents import ForkedAgentLoop, create_fork_tool


class MainAgent:
    """把 LangChain `create_agent` 适配为应用层 Runtime 端口。"""

    def __init__(
        self,
        *,
        model: BaseChatModel,
        tools: Sequence[BaseTool],
        governance_config: GovernanceConfig,
        compressor: ContextCompressor | None = None,
        checkpointer: BaseCheckpointSaver[Any] | None = None,
        sub_agent_max_concurrency: int = 50,
    ) -> None:
        self._model = model
        self._config = governance_config
        self._checkpointer = checkpointer or InMemorySaver()
        child_tools = list(tools)

        child_middleware = create_context_middleware(
            model=self._model,
            tools=child_tools,
            system_prompt=SYSTEM_PROMPT,
            agent_id="forked_sub_agent",
            config=self._config,
            compressor=compressor,
        )
        child_loop = ForkedAgentLoop.create(
            model=self._model,
            tools=child_tools,
            system_prompt=SYSTEM_PROMPT,
            max_concurrency=sub_agent_max_concurrency,
            middleware=child_middleware,
        )
        main_tools = [*child_tools, create_fork_tool(child_loop)]
        main_middleware = create_context_middleware(
            model=self._model,
            tools=main_tools,
            system_prompt=SYSTEM_PROMPT,
            agent_id="main_agent",
            config=self._config,
            compressor=compressor,
        )
        self._agent = create_agent(
            model=self._model,
            tools=main_tools,
            system_prompt=SYSTEM_PROMPT,
            middleware=main_middleware,
            state_schema=SessionAgentState,
            context_schema=AgentExecutionContext,
            checkpointer=self._checkpointer,
            name="main_agent",
        )

    async def stream(
        self,
        message: str,
        context: AgentExecutionContext,
    ) -> AsyncIterator[dict[str, str]]:
        """执行 LangGraph 循环，只输出框架归一化后的内容片段。"""

        config = {"configurable": {"thread_id": context.thread_id}}
        token = set_context(context)
        try:
            async for update in self._agent.astream(
                {"messages": [("user", message)]},
                config=config,
                context=context,
                stream_mode="updates",
            ):
                if not isinstance(update, dict):
                    continue
                for node_update in update.values():
                    if not isinstance(node_update, dict):
                        continue
                    for output_message in node_update.get("messages", []):
                        if not isinstance(output_message, AIMessage):
                            continue
                        content = output_message.text
                        if content:
                            yield {
                                "type": "text_message_content",
                                "content": content,
                            }
        finally:
            reset_context(token)

    async def run_agent(self, message: str) -> AsyncIterator[dict[str, str]]:
        """兼容旧调用入口；新代码应通过应用层 `RunAgent.execute` 调用。"""

        run_id = str(uuid4())
        context = current_execution_context.get() or AgentExecutionContext(
            thread_id=f"main-{uuid4().hex[:12]}",
            run_id=run_id,
        )
        yield {"type": "run_started", "run_id": run_id}
        try:
            async for event in self.stream(message, context):
                yield event
        except Exception as exc:  # noqa: BLE001 - 仅为旧入口保持事件兼容。
            yield {"type": "run_error", "message": str(exc), "run_id": run_id}
            return
        yield {"type": "run_finished", "run_id": run_id}
