"""主 AgentLoop、工具装配和流式执行入口。"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any
from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from app.application.runtime import AgentExecutionContext
from app.application.tasking import TaskBoardService, TaskDispatchService
from app.infrastructure.context import (
    current_execution_context,
    reset_context,
    set_context,
)
from app.infrastructure.context_governance.compressor import ContextCompressor
from app.infrastructure.context_governance.config import GovernanceConfig
from app.infrastructure.context_governance.factory import create_context_middleware
from app.infrastructure.context_governance.schemas import SessionAgentState

from .prompts import MAIN_SYSTEM_PROMPT, SUB_AGENT_SYSTEM_PROMPT
from .sub_agents import ForkedAgentLoop, ForkedTaskExecutor, create_fork_tool
from .sub_agents.fork import ObservabilityCallbacks


class MainAgent:
    """把 LangChain `create_agent` 适配为应用层 Runtime 端口。"""

    def __init__(
        self,
        *,
        model: BaseChatModel,
        tools: Sequence[BaseTool],
        governance_config: GovernanceConfig,
        main_only_tools: Sequence[BaseTool] = (),
        task_service: TaskBoardService | None = None,
        compressor: ContextCompressor | None = None,
        checkpointer: BaseCheckpointSaver[Any] | None = None,
        sub_agent_max_concurrency: int = 10,
        shared_middleware: Sequence[AgentMiddleware] = (),
        observability: ObservabilityCallbacks | None = None,
        main_system_prompt: str = MAIN_SYSTEM_PROMPT,
        sub_agent_system_prompt: str = SUB_AGENT_SYSTEM_PROMPT,
    ) -> None:
        self._model = model
        self._config = governance_config
        self._checkpointer = checkpointer or InMemorySaver()
        self._observability = observability
        child_tools = list(tools)

        child_middleware = [
            *shared_middleware,
            *create_context_middleware(
                model=self._model,
                tools=child_tools,
                system_prompt=sub_agent_system_prompt,
                agent_id="forked_sub_agent",
                config=self._config,
                compressor=compressor,
            ),
        ]
        child_loop = ForkedAgentLoop.create(
            model=self._model,
            tools=child_tools,
            system_prompt=sub_agent_system_prompt,
            max_concurrency=sub_agent_max_concurrency,
            middleware=child_middleware,
            observability=observability,
            checkpointer=self._checkpointer,
        )
        task_dispatcher = (
            TaskDispatchService(task_service, ForkedTaskExecutor(child_loop))
            if task_service is not None
            else None
        )
        # 任务控制面只属于主 Agent；子 Agent 只复用业务执行工具。
        main_tools = [
            *child_tools,
            *main_only_tools,
            create_fork_tool(child_loop, task_dispatcher),
        ]
        self._child_tool_names = tuple(tool.name for tool in child_tools)
        self._main_tool_names = tuple(tool.name for tool in main_tools)
        main_middleware = [
            *shared_middleware,
            *create_context_middleware(
                model=self._model,
                tools=main_tools,
                system_prompt=main_system_prompt,
                agent_id="main_agent",
                config=self._config,
                compressor=compressor,
            ),
        ]
        self._agent = create_agent(
            model=self._model,
            tools=main_tools,
            system_prompt=main_system_prompt,
            middleware=main_middleware,
            state_schema=SessionAgentState,
            context_schema=AgentExecutionContext,
            checkpointer=self._checkpointer,
            name="main_agent",
        )

    @property
    def child_tool_names(self) -> tuple[str, ...]:
        """返回 fork 子 Agent 可见的工具名，用于装配检查和诊断。"""

        return self._child_tool_names

    @property
    def main_tool_names(self) -> tuple[str, ...]:
        """返回主 Agent 可见的完整工具名。"""

        return self._main_tool_names

    async def state_snapshot(self, thread_id: str) -> dict[str, Any]:
        """读取指定 thread 的 LangGraph 状态，供诊断和离线证据使用。"""

        snapshot = await self._agent.aget_state(
            {"configurable": {"thread_id": thread_id}}
        )
        return dict(snapshot.values)

    async def stream(
        self,
        message: str,
        context: AgentExecutionContext,
    ) -> AsyncIterator[dict[str, Any]]:
        """执行 LangGraph 循环，只输出框架归一化后的内容片段。"""

        config: dict[str, Any] = {
            "configurable": {"thread_id": context.thread_id},
        }
        if self._observability is not None:
            config["callbacks"] = self._observability.create(context, "main_agent")
            config["metadata"] = self._observability.metadata(context, "main_agent")
            config["run_name"] = "globex-main-agent"
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
                        if isinstance(output_message, AIMessage):
                            for tool_call in output_message.tool_calls:
                                yield {
                                    "type": "tool_invoke",
                                    "tool_name": str(tool_call.get("name", "")),
                                    "tool_call_id": str(tool_call.get("id", "")),
                                    "arguments": tool_call.get("args", {}),
                                }
                            content = output_message.text
                            if content:
                                yield {
                                    "type": "text_message_content",
                                    "content": content,
                                }
                        elif isinstance(output_message, ToolMessage):
                            yield {
                                "type": "tool_result",
                                "tool_name": output_message.name or "",
                                "tool_call_id": output_message.tool_call_id,
                                "status": output_message.status,
                                "content": output_message.text,
                            }
                    governance = node_update.get("last_governance")
                    strategies = (
                        governance.get("strategies", [])
                        if isinstance(governance, dict)
                        else []
                    )
                    if any(strategy != "none" for strategy in strategies):
                        yield {
                            "type": "context_compressed",
                            "details": governance,
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
