"""会话存储出站端口。"""

from typing import Protocol

from app.domain.runtime import RunId

from .models import RunCheckpoint


class CheckpointRepository(Protocol):
    """保存并恢复平台定义的 RunCheckpoint。"""

    async def save(self, checkpoint: RunCheckpoint) -> None: ...

    async def latest(self, run_id: RunId) -> RunCheckpoint | None: ...
