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

from app.domain.runtime import AgentId, AgentRun, RunId, ThreadId
from app.infrastructure.context import RequestContext, current_context
from app.infrastructure.context_governance.compressor import ContextCompressor
from app.infrastructure.context_governance.config import GovernanceConfig
from app.infrastructure.context_governance.factory import create_context_middleware
from app.infrastructure.context_governance.schemas import SessionAgentState

from .prompts import SYSTEM_PROMPT
from .sub_agents import ForkedAgentLoop, create_fork_tool


class MainAgent:
    """使用 LangChain `create_agent` 运行主循环并输出前端事件。"""

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
            context_schema=RequestContext,
            checkpointer=self._checkpointer,
            name="main_agent",
        )

    async def run_agent(self, message: str) -> AsyncIterator[dict[str, str]]:
        """运行一次主 AgentLoop，并流式输出归一化事件。"""

        run_id = str(uuid4())
        request_context = current_context.get() or RequestContext(
            thread_id=f"main-{uuid4().hex[:12]}"
        )
        run = AgentRun(
            run_id=RunId(run_id),
            thread_id=ThreadId(request_context.thread_id),
            agent_id=AgentId("main_agent"),
        )
        run.start()
        config = {"configurable": {"thread_id": request_context.thread_id}}
        yield {"type": "run_started", "run_id": run_id}
        try:
            async for update in self._agent.astream(
                {"messages": [("user", message)]},
                config=config,
                context=request_context,
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
        except Exception as exc:  # noqa: BLE001 - API 边界必须转换运行时错误事件。
            run.fail(str(exc))
            yield {"type": "run_error", "message": str(exc), "run_id": run_id}
            return
        run.complete()
        yield {"type": "run_finished", "run_id": run_id}
