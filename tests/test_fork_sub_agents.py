"""验证子 AgentLoop 的 fork 判断、并发调度和上下文隔离。"""

import asyncio
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from app.agent.sub_agents import ForkedAgentLoop, should_fork


class FakeAgent:
    """记录调用配置的轻量 Agent 替身，避免测试依赖真实模型服务。"""

    def __init__(self) -> None:
        self.configs: list[RunnableConfig | None] = []
        self.active_calls = 0
        self.max_active_calls = 0

    async def ainvoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        self.configs.append(config)
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
