"""Agent Run 快照持久化端口。"""

from typing import Protocol

from app.application.runtime import RunCheckpoint, RunId


class CheckpointStore(Protocol):
    """保存并恢复平台定义的 RunCheckpoint。"""

    async def save(self, checkpoint: RunCheckpoint) -> None: ...

    async def latest(self, run_id: RunId) -> RunCheckpoint | None: ...
