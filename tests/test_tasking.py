"""验证任务看板、依赖 DAG、并发认领和四个工具。"""

import asyncio
import json
from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.tools import BaseTool

from app.application.runtime import AgentExecutionContext
from app.application.tasking import (
    TaskBoardService,
    TaskDependencyError,
    TaskDispatchService,
    TaskExecutionOutcome,
    TaskWorkItem,
)
from app.infrastructure.context import reset_context, set_context
from app.infrastructure.langchain.tools import create_task_tools
from app.infrastructure.tasking import SQLiteTaskBoardRepository

TEST_ROOT = Path("data/test-output/tasking-tests") / uuid4().hex


def _service(name: str) -> TaskBoardService:
    return TaskBoardService(
        SQLiteTaskBoardRepository(TEST_ROOT / f"{name}.db")
    )


def test_task_board_crud_metadata_and_session_isolation() -> None:
    service = _service("crud")

    async def scenario() -> None:
        created = await service.create(
            "thread-a",
            subject="检索商品",
            description="检索旅行收纳袋",
            metadata={"platform": "global"},
        )
        assert created.id == "1"
        assert created.runnable

        updated = await service.update(
            "thread-a",
            "1",
            owner="main_agent",
            metadata={"platform": None, "priority": 1},
        )
        assert updated is not None
        assert updated.owner == "main_agent"
        assert updated.metadata == {"priority": 1}
        assert await service.list("thread-b") == ()

        deleted = await service.update("thread-a", "1", status="deleted")
        assert deleted is None
        assert await service.list("thread-a") == ()

    asyncio.run(scenario())


def test_dependency_dag_blocks_and_unblocks_tasks() -> None:
    service = _service("unblock")

    async def scenario() -> None:
        first = await service.create(
            "thread",
            subject="搜索候选",
            description="搜索候选商品",
        )
        second = await service.create(
            "thread",
            subject="比较候选",
            description="比较候选商品",
        )
        await service.update("thread", second.id, add_blocked_by=[first.id])

        blocked = await service.get("thread", second.id)
        assert blocked.effective_status == "blocked"
        assert blocked.active_blocked_by == (first.id,)
        assert not blocked.runnable
        assert (await service.get("thread", first.id)).blocks == (second.id,)

        with pytest.raises(TaskDependencyError):
            await service.update("thread", second.id, status="in_progress")

        await service.update("thread", first.id, status="in_progress")
        await service.update(
            "thread",
            first.id,
            status="completed",
            result="已找到候选",
        )
        unblocked = await service.get("thread", second.id)
        assert unblocked.runnable
        assert unblocked.active_blocked_by == ()

    asyncio.run(scenario())


def test_dependency_dag_rejects_self_reference_and_cycle() -> None:
    service = _service("cycle")

    async def scenario() -> None:
        first = await service.create(
            "thread",
            subject="任务一",
            description="任务一描述",
        )
        second = await service.create(
            "thread",
            subject="任务二",
            description="任务二描述",
        )
        with pytest.raises(TaskDependencyError):
            await service.update("thread", first.id, add_blocked_by=[first.id])
        await service.update("thread", second.id, add_blocked_by=[first.id])
        with pytest.raises(TaskDependencyError):
            await service.update("thread", first.id, add_blocked_by=[second.id])

    asyncio.run(scenario())


def test_concurrent_task_creation_assigns_unique_sequential_ids() -> None:
    service = _service("concurrency")

    async def scenario() -> None:
        created = await asyncio.gather(
            *(
                service.create(
                    "thread",
                    subject=f"任务 {index}",
                    description=f"执行任务 {index}",
                )
                for index in range(20)
            )
        )
        assert sorted(int(task.id) for task in created) == list(range(1, 21))
        assert len(await service.list("thread")) == 20

    asyncio.run(scenario())


class FakeParallelExecutor:
    """为派发测试返回一个成功和一个失败结果。"""

    async def execute_many(
        self,
        tasks: list[TaskWorkItem] | tuple[TaskWorkItem, ...],
    ) -> tuple[TaskExecutionOutcome, ...]:
        return (
            TaskExecutionOutcome(
                task_id=tasks[0].task_id,
                succeeded=True,
                result="任务一完成",
            ),
            TaskExecutionOutcome(
                task_id=tasks[1].task_id,
                succeeded=False,
                error="任务二失败",
            ),
        )


def test_deterministic_dispatch_claims_and_writes_back_results() -> None:
    tasks = _service("dispatch")
    dispatcher = TaskDispatchService(tasks, FakeParallelExecutor())

    async def scenario() -> None:
        first = await tasks.create(
            "thread",
            subject="并行任务一",
            description="执行并行任务一",
        )
        second = await tasks.create(
            "thread",
            subject="并行任务二",
            description="执行并行任务二",
        )
        report = await dispatcher.dispatch(
            "thread",
            [first.id, second.id],
            reason="parallel",
        )
        assert [task.status.value for task in report.tasks] == [
            "completed",
            "failed",
        ]
        assert report.tasks[0].result == "任务一完成"
        assert report.tasks[1].error == "任务二失败"
        assert all(task.owner and task.owner.startswith("fork:") for task in report.tasks)

    asyncio.run(scenario())


def test_parallel_dispatch_rejects_blocked_or_single_task() -> None:
    tasks = _service("dispatch-reject")
    dispatcher = TaskDispatchService(tasks, FakeParallelExecutor())

    async def scenario() -> None:
        first = await tasks.create(
            "thread",
            subject="前置任务",
            description="完成前置任务",
        )
        second = await tasks.create(
            "thread",
            subject="后置任务",
            description="完成后置任务",
        )
        await tasks.update("thread", second.id, add_blocked_by=[first.id])
        with pytest.raises(TaskDependencyError):
            await dispatcher.dispatch(
                "thread",
                [first.id, second.id],
                reason="parallel",
            )
        with pytest.raises(ValueError, match="至少需要两个"):
            await dispatcher.dispatch("thread", [first.id], reason="parallel")

    asyncio.run(scenario())


def test_task_tools_are_decorated_and_use_context_thread() -> None:
    service = _service("tools")
    tools = create_task_tools(service)
    by_name = {tool.name: tool for tool in tools}
    assert set(by_name) == {"TaskCreate", "TaskGet", "TaskList", "TaskUpdate"}
    assert all(isinstance(tool, BaseTool) for tool in tools)

    context = AgentExecutionContext(thread_id="tool-thread")
    token = set_context(context)
    try:
        created_raw = asyncio.run(
            by_name["TaskCreate"].ainvoke(
                {"subject": "工具任务", "description": "验证工具调用"}
            )
        )
        listed_raw = asyncio.run(by_name["TaskList"].ainvoke({}))
    finally:
        reset_context(token)

    assert json.loads(created_raw)["task"]["id"] == "1"
    assert json.loads(listed_raw)["count"] == 1
