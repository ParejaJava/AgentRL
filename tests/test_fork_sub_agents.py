"""验证子 AgentLoop 的 fork 判断、并发调度和上下文隔离。"""

import asyncio
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from app.application.agents import OrchestrationPolicy, should_fork
from app.application.runtime import AgentExecutionContext, ShoppingContextSnapshot
from app.application.tasking import TaskBoardService, TaskDispatchService
from app.infrastructure.context import reset_context, set_context
from app.infrastructure.langchain.sub_agents import (
    ForkedAgentLoop,
    ForkedTaskExecutor,
    create_fork_tool,
)
from app.infrastructure.tasking import SQLiteTaskBoardRepository


class FakeAgent:
    """记录调用配置的轻量 Agent 替身，避免测试依赖真实模型服务。"""

    def __init__(self) -> None:
        self.configs: list[RunnableConfig | None] = []
        self.contexts: list[AgentExecutionContext | None] = []
        self.active_calls = 0
        self.max_active_calls = 0

    async def ainvoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        context: AgentExecutionContext | None = None,
    ) -> dict[str, Any]:
        self.configs.append(config)
        self.contexts.append(context)
        demand = input["messages"][0][1]
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            # 短暂挂起，让测试能够观测多个子任务是否真的同时处于运行状态。
            await asyncio.sleep(0.01)
            return {"messages": [AIMessage(content=f"完成：{demand}")]}
        finally:
            self.active_calls -= 1


def test_should_fork_when_any_condition_matches() -> None:
    assert should_fork(
        can_run_parallel=True,
        needs_context_isolation=False,
        call_depth=1,
    )
    assert should_fork(
        can_run_parallel=False,
        needs_context_isolation=True,
        call_depth=1,
    )
    assert should_fork(
        can_run_parallel=False,
        needs_context_isolation=False,
        call_depth=3,
    )
    assert not should_fork(
        can_run_parallel=False,
        needs_context_isolation=False,
        call_depth=2,
    )


def test_orchestration_policy_normalizes_fork_plan() -> None:
    """Application Orchestrator 应清理空任务和重复任务。"""

    plan = OrchestrationPolicy().create_fork_plan(
        [" 搜索亚马逊 ", "", "搜索亚马逊", "搜索 Shopee"],
        "parallel",
    )

    assert plan.tasks == ("搜索亚马逊", "搜索 Shopee")
    assert plan.reason == "parallel"


def test_forked_agent_loop_uses_independent_threads() -> None:
    fake_agent = FakeAgent()
    loop = ForkedAgentLoop(fake_agent, max_concurrency=2)

    answers = asyncio.run(loop.arun_many(["任务一", "任务二", "任务三"]))

    assert answers == ["完成：任务一", "完成：任务二", "完成：任务三"]
    thread_ids = {
        config["configurable"]["thread_id"]
        for config in fake_agent.configs
        if config is not None
    }
    assert len(thread_ids) == 3
    assert all(thread_id.startswith("sub-") for thread_id in thread_ids)
    assert fake_agent.max_active_calls == 2


def test_create_fork_tool_uses_decorated_async_tool() -> None:
    loop = ForkedAgentLoop(FakeAgent())

    fork_tool = create_fork_tool(loop)
    result = asyncio.run(fork_tool.ainvoke({"tasks": ["任务一"], "reason": "parallel"}))

    assert isinstance(fork_tool, BaseTool)
    assert fork_tool.name == "fork_sub_agents"
    assert json.loads(result) == {
        "status": "ok",
        "fork_reason": "parallel",
        "results": [{"task": "任务一", "answer": "完成：任务一"}],
    }


def test_fork_inherits_shopping_session_but_isolates_execution_ids() -> None:
    """子 Agent 继承购物身份，但必须拥有独立 thread_id 和 run_id。"""

    fake_agent = FakeAgent()
    loop = ForkedAgentLoop(fake_agent, pass_runtime_context=True)
    parent = AgentExecutionContext(
        thread_id="main-thread",
        run_id="main-run",
        shopping=ShoppingContextSnapshot("shopping-1", "buyer-1"),
    )

    token = set_context(parent)
    try:
        asyncio.run(loop.arun("搜索亚马逊"))
    finally:
        reset_context(token)

    child = fake_agent.contexts[0]
    assert child is not None
    assert child.shopping == parent.shopping
    assert child.thread_id != parent.thread_id
    assert child.run_id != parent.run_id


def test_parallel_fork_dispatches_only_runnable_task_ids_and_writes_back() -> None:
    """parallel 模式应经过 TaskBoard 认领并自动保存子 Agent 结果。"""

    database = (
        Path("data/test-output/tasking-tests") / uuid4().hex / "fork-tasks.db"
    )
    tasks = TaskBoardService(SQLiteTaskBoardRepository(database))
    fake_agent = FakeAgent()
    loop = ForkedAgentLoop(fake_agent, max_concurrency=2)
    dispatcher = TaskDispatchService(tasks, ForkedTaskExecutor(loop))
    fork_tool = create_fork_tool(loop, dispatcher)

    async def scenario() -> dict[str, Any]:
        first = await tasks.create(
            "main-thread",
            subject="检索平台一",
            description="检索平台一的商品",
        )
        second = await tasks.create(
            "main-thread",
            subject="检索平台二",
            description="检索平台二的商品",
        )
        context = AgentExecutionContext(thread_id="main-thread")
        token = set_context(context)
        try:
            raw = await fork_tool.ainvoke(
                {
                    "reason": "parallel",
                    "task_ids": [first.id, second.id],
                }
            )
        finally:
            reset_context(token)
        return json.loads(raw)

    result = asyncio.run(scenario())

    assert result["status"] == "ok"
    assert [task["status"] for task in result["tasks"]] == [
        "completed",
        "completed",
    ]
    assert fake_agent.max_active_calls == 2
