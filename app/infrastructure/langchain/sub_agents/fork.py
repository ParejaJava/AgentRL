"""创建上下文隔离、可并发执行的同质子 AgentLoop。"""

import asyncio
import json
import random
from collections.abc import Callable, Sequence
from typing import Any, Protocol
from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from app.application.agents import ForkReason, OrchestrationPolicy
from app.application.runtime import AgentExecutionContext
from app.application.tasking import (
    TaskDispatchService,
    TaskExecutionOutcome,
    TaskWorkItem,
)
from app.infrastructure.context import (
    current_execution_context,
    require_context,
    reset_context,
    set_context,
)
from app.infrastructure.context_governance.runtime import sanitize_thread_id
from app.infrastructure.context_governance.schemas import SessionAgentState


class AgentInvoker(Protocol):
    """描述子 AgentLoop 需要提供的最小异步调用接口。"""

    async def ainvoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]: ...


class ObservabilityCallbacks(Protocol):
    """子 Agent 只依赖的观测回调最小接口。"""

    def create(self, context: AgentExecutionContext, agent_id: str) -> list[Any]: ...

    def metadata(
        self,
        context: AgentExecutionContext,
        agent_id: str,
    ) -> dict[str, object]: ...


class ForkedAgentLoop:
    """复用同质 Agent 图，以独立线程并发执行多个隔离的子任务。"""

    def __init__(
        self,
        agent: AgentInvoker,
        *,
        max_concurrency: int = 4,
        max_rate_limit_retries: int = 4,
        pass_runtime_context: bool = False,
        observability: ObservabilityCallbacks | None = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency 必须大于或等于 1")
        if max_rate_limit_retries < 0:
            raise ValueError("max_rate_limit_retries 不能小于 0")

        self._agent = agent
        self._max_concurrency = max_concurrency
        self._max_rate_limit_retries = max_rate_limit_retries
        self._pass_runtime_context = pass_runtime_context
        self._observability = observability

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
        middleware: Sequence[AgentMiddleware] = (),
        observability: ObservabilityCallbacks | None = None,
    ) -> "ForkedAgentLoop":
        """使用统一模型和工具集创建可 fork 的同质子 AgentLoop。"""

        # thread_id 只有配合 checkpointer 才能形成真正独立的短期消息历史。
        child_agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            middleware=middleware,
            state_schema=SessionAgentState,
            context_schema=AgentExecutionContext,
            checkpointer=checkpointer or InMemorySaver(),
            name="forked_sub_agent",
        )
        return cls(
            child_agent,
            max_concurrency=max_concurrency,
            max_rate_limit_retries=max_rate_limit_retries,
            pass_runtime_context=True,
            observability=observability,
        )

    async def arun(self, demand: str) -> str:
        """在全新线程中运行一个子任务，并且只返回最终回答。"""

        # 每次 fork 都生成独立 thread_id，避免读取其他子任务或主循环的上下文。
        sub_thread_id = f"sub-{uuid4().hex[:8]}"
        config: RunnableConfig = {
            "configurable": {"thread_id": sub_thread_id},
        }
        parent = current_execution_context.get()
        if parent is not None and parent.session_dir:
            parent_session_dir = parent.session_dir.rstrip("/\\")
            child_session_dir = (
                f"{parent_session_dir}/sub_agents/{sanitize_thread_id(sub_thread_id)}"
            )
        elif parent is not None:
            child_session_dir = (
                f"{sanitize_thread_id(parent.thread_id)}/sub_agents/"
                f"{sanitize_thread_id(sub_thread_id)}"
            )
        else:
            child_session_dir = None
        child_context = AgentExecutionContext(
            thread_id=sub_thread_id,
            shopping=parent.shopping if parent is not None else None,
            run_id=str(uuid4()),
            session_dir=child_session_dir,
        )
        if self._observability is not None:
            config["callbacks"] = self._observability.create(
                child_context,
                "forked_sub_agent",
            )
            config["metadata"] = self._observability.metadata(
                child_context,
                "forked_sub_agent",
            )
            config["run_name"] = "globex-forked-sub-agent"
        token = set_context(child_context)
        try:
            result = await self._ainvoke_with_rate_limit_retry(
                demand,
                config,
                child_context,
            )
        finally:
            reset_context(token)

        # 不向主 Agent 传播子循环的中间消息和工具轨迹。
        final_message = result["messages"][-1]
        return str(final_message.text)

    async def _ainvoke_with_rate_limit_retry(
        self,
        demand: str,
        config: RunnableConfig,
        context: AgentExecutionContext,
    ) -> dict[str, Any]:
        """遇到供应商 429 限流时进行带抖动的指数退避重试。"""

        for attempt in range(self._max_rate_limit_retries + 1):
            try:
                invoke_kwargs: dict[str, Any] = {"config": config}
                if self._pass_runtime_context:
                    invoke_kwargs["context"] = context
                return await self._agent.ainvoke(
                    {"messages": [("user", demand)]}, **invoke_kwargs
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

        settled = await self.arun_many_settled(demands)
        answers: list[str] = []
        for result in settled:
            if isinstance(result, Exception):
                raise result
            answers.append(result)
        return answers

    async def arun_many_settled(
        self,
        demands: Sequence[str],
    ) -> list[str | Exception]:
        """并发执行并保留每个任务的成功或失败，避免丢失部分结果。"""

        if not demands:
            return []

        # 信号量限制同时运行的模型请求数，避免触发供应商限流。
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def run_with_limit(demand: str) -> str | Exception:
            async with semaphore:
                try:
                    return await self.arun(demand)
                except Exception as exc:  # noqa: BLE001 - 每个子任务独立结算。
                    return exc

        # gather 并发调度任务，同时保持返回结果与 demands 的顺序一致。
        return list(await asyncio.gather(*(run_with_limit(item) for item in demands)))


class ForkedTaskExecutor:
    """把同质 LangGraph 子 AgentLoop 适配为 Application 执行端口。"""

    def __init__(self, sub_agent_loop: ForkedAgentLoop) -> None:
        self._sub_agent_loop = sub_agent_loop

    async def execute_many(
        self,
        tasks: Sequence[TaskWorkItem],
    ) -> tuple[TaskExecutionOutcome, ...]:
        """并发执行工作项，并逐项保留成功结果或错误。"""

        settled = await self._sub_agent_loop.arun_many_settled(
            [task.to_prompt() for task in tasks]
        )
        return tuple(
            TaskExecutionOutcome(
                task_id=task.task_id,
                succeeded=not isinstance(result, Exception),
                result=result if isinstance(result, str) else None,
                error=str(result) if isinstance(result, Exception) else None,
            )
            for task, result in zip(tasks, settled, strict=True)
        )


def create_fork_tool(
    sub_agent_loop: ForkedAgentLoop,
    task_dispatcher: TaskDispatchService | None = None,
) -> BaseTool:
    """创建供主 AgentLoop 调用的 fork 工具。"""

    policy = OrchestrationPolicy()

    @tool
    async def fork_sub_agents(
        reason: ForkReason,
        task_ids: list[str] | None = None,
        tasks: list[str] | None = None,
    ) -> str:
        """Fork 子 AgentLoop 并只返回各子任务的最终回答。

        reason=parallel 时必须传入至少两个由 TaskList 返回且 runnable=true 的
        task_ids；系统会再次校验依赖、原子认领、并发执行并自动回写 completed 或
        failed。reason=context_isolation 或 deep_chain 时，可以传 task_ids 走任务看板，
        也可以传入自包含的 tasks 做一次性隔离执行。task_ids 与 tasks 不能同时传入。
        """

        if task_ids and tasks:
            return json.dumps(
                {"status": "error", "message": "task_ids 与 tasks 不能同时传入"},
                ensure_ascii=False,
            )
        if task_ids:
            if task_dispatcher is None:
                return json.dumps(
                    {"status": "error", "message": "任务派发服务尚未配置"},
                    ensure_ascii=False,
                )
            try:
                report = await task_dispatcher.dispatch(
                    require_context().thread_id,
                    task_ids,
                    reason=reason,
                )
            except ValueError as exc:
                return json.dumps(
                    {"status": "error", "message": str(exc)},
                    ensure_ascii=False,
                )
            return json.dumps(
                {"status": "ok", **report.to_dict()},
                ensure_ascii=False,
            )

        # 配置任务服务后，parallel 不允许绕过 Task DAG 直接提交自然语言任务。
        if reason == "parallel" and task_dispatcher is not None:
            return json.dumps(
                {
                    "status": "error",
                    "message": "parallel 派发必须先创建任务并传入 task_ids",
                },
                ensure_ascii=False,
            )

        # 一次性隔离执行仍由应用层策略清理空任务和重复任务。
        try:
            plan = policy.create_fork_plan(tasks or [], reason)
        except ValueError as exc:
            return json.dumps(
                {"status": "error", "message": str(exc)},
                ensure_ascii=False,
            )
        answers = await sub_agent_loop.arun_many(plan.tasks)
        final_answers = [
            {"task": task, "answer": answer}
            for task, answer in zip(plan.tasks, answers, strict=True)
        ]
        return json.dumps(
            {
                "status": "ok",
                "fork_reason": plan.reason,
                "results": final_answers,
            },
            ensure_ascii=False,
        )

    return fork_sub_agents
