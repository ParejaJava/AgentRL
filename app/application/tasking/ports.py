"""任务看板的仓储与子 Agent 执行端口。"""

from collections.abc import Sequence
from typing import Protocol

from .models import TaskBoard, TaskExecutionOutcome, TaskWorkItem


class TaskBoardConflictError(RuntimeError):
    """仓储版本已经变化，当前写入必须重新读取后重试。"""


class TaskBoardRepository(Protocol):
    """保存会话任务看板的乐观并发仓储端口。"""

    async def load(self, scope_id: str) -> TaskBoard: ...

    async def save(
        self,
        board: TaskBoard,
        *,
        expected_version: int,
    ) -> TaskBoard: ...


class ParallelTaskExecutor(Protocol):
    """批量执行已经被认领的独立任务。"""

    async def execute_many(
        self,
        tasks: Sequence[TaskWorkItem],
    ) -> tuple[TaskExecutionOutcome, ...]: ...
