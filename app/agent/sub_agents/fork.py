"""创建上下文隔离、可并发执行的同质子 AgentLoop。"""

import asyncio
import json
import random
from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol, cast
from uuid import uuid4

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

ForkReason = Literal["parallel", "context_isolation", "deep_chain"]


class AgentInvoker(Protocol):
    """描述子 AgentLoop 需要提供的最小异步调用接口。"""

    async def ainvoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]: ...


def should_fork(
    *,
    can_run_parallel: bool,
    needs_context_isolation: bool,
    call_depth: int,
) -> bool:
    """判断任务是否满足图示中的任意一项 fork 条件。"""

    return can_run_parallel or needs_context_isolation or call_depth >= 3


class ForkedAgentLoop:
    """复用同质 Agent 图，以独立线程并发执行多个隔离的子任务。"""

    def __init__(
        self,
        agent: AgentInvoker,
        *,
        max_concurrency: int = 4,
        max_rate_limit_retries: int = 4,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency 必须大于或等于 1")
        if max_rate_limit_retries < 0:
            raise ValueError("max_rate_limit_retries 不能小于 0")

        self._agent = agent
        self._max_concurrency = max_concurrency
        self._max_rate_limit_retries = max_rate_limit_retries

    @classmethod
    def create(
        cls,
        *,
        model: BaseChatModel,
        tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]],
        system_prompt: str,
        max_concurrency: int = 4,
        max_rate_limit_retries: int = 4,
        checkpointer: BaseCheckpointSaver[Any] | None = None,
    ) -> "ForkedAgentLoop":
        """使用统一模型和工具集创建可 fork 的同质子 AgentLoop。"""

        # thread_id 只有配合 checkpointer 才能形成真正独立的短期消息历史。
        child_agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            checkpointer=checkpointer or InMemorySaver(),
            name="forked_sub_agent",
        )
        return cls(
            child_agent,
            max_concurrency=max_concurrency,
            max_rate_limit_retries=max_rate_limit_retries,
        )

    async def arun(self, demand: str) -> str:
        """在全新线程中运行一个子任务，并且只返回最终回答。"""

        # 每次 fork 都生成独立 thread_id，避免读取其他子任务或主循环的上下文。
        sub_thread_id = f"sub-{uuid4().hex[:8]}"
        config: RunnableConfig = {
            "configurable": {"thread_id": sub_thread_id},
        }
        result = await self._ainvoke_with_rate_limit_retry(demand, config)

        # 不向主 Agent 传播子循环的中间消息和工具轨迹。
        final_message = result["messages"][-1]
        return str(final_message.text)

    async def _ainvoke_with_rate_limit_retry(
        self,
        demand: str,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        """遇到供应商 429 限流时进行带抖动的指数退避重试。"""

        for attempt in range(self._max_rate_limit_retries + 1):
            try:
                return await self._agent.ainvoke(
                    {"messages": [("user", demand)]},
                    config=config,
                )
            except Exception as exc:
                is_rate_limit = getattr(exc, "status_code", None) == 429
                retries_exhausted = attempt >= self._max_rate_limit_retries
                if not is_rate_limit or retries_exhausted:
                    raise

                # 抖动可避免多个并发任务在同一时刻再次请求，形成重试碰撞。
                delay = (2**attempt) + random.uniform(0, 0.5)
                await asyncio.sleep(delay)

        raise RuntimeError("限流重试逻辑意外结束")

    async def arun_many(self, demands: Sequence[str]) -> list[str]:
        """并发执行多个子任务，并按照输入顺序返回最终回答。"""

        if not demands:
            return []

        # 信号量限制同时运行的模型请求数，避免触发供应商限流。
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def run_with_limit(demand: str) -> str:
            async with semaphore:
                return await self.arun(demand)

        # gather 并发调度任务，同时保持返回结果与 demands 的顺序一致。
        return list(await asyncio.gather(*(run_with_limit(item) for item in demands)))


def create_fork_tool(sub_agent_loop: ForkedAgentLoop) -> StructuredTool:
    """创建供主 AgentLoop 调用的 fork 工具。"""

    async def fork_sub_agents(tasks: list[str], reason: ForkReason) -> str:
        """Fork 子 AgentLoop 并只返回各子任务的最终回答。

        当任务可并行、需要隔离上下文，或预计调用链不少于三层时使用。
        reason 分别填写 parallel、context_isolation 或 deep_chain。
        tasks 应拆成彼此完整、可独立执行的任务描述。
        """

        if not tasks:
            return "没有可执行的子任务。"

        # reason 会进入工具参数 Schema，迫使主 Agent 明确说明 fork 原因。
        answers = await sub_agent_loop.arun_many(tasks)
        final_answers = [
            {"task": task, "answer": answer}
            for task, answer in zip(tasks, answers, strict=True)
        ]
        return json.dumps(
            {"fork_reason": reason, "results": final_answers},
            ensure_ascii=False,
        )

    return cast(StructuredTool, StructuredTool.from_function(coroutine=fork_sub_agents))
