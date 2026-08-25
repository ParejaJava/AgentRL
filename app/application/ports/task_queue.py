"""后台 Agent 任务队列端口。"""

from typing import Protocol


class TaskQueue(Protocol):
    """把可持久化任务提交给独立 Worker。"""

    async def enqueue(self, task_id: str, payload: dict[str, object]) -> None: ...

    async def acknowledge(self, task_id: str) -> None: ...
