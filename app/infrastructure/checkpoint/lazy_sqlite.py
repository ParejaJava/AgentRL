"""可在同步 Composition Root 中创建的惰性异步 SQLite Checkpointer。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)


class LazyAsyncSqliteSaver(BaseCheckpointSaver[Any]):
    """首次异步访问时打开官方 AsyncSqliteSaver，并复用同一连接。"""

    def __init__(self, database_path: Path) -> None:
        super().__init__()
        self._database_path = database_path
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._delegate: Any = None
        self._context_manager: Any = None
        self._lock = asyncio.Lock()

    async def _get_delegate(self) -> Any:
        if self._delegate is not None:
            return self._delegate
        async with self._lock:
            if self._delegate is not None:
                return self._delegate
            try:
                from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
            except ImportError as exc:
                raise RuntimeError(
                    "SQLite Checkpointer 已启用；请运行 `uv sync --extra platform`"
                ) from exc
            manager = AsyncSqliteSaver.from_conn_string(str(self._database_path))
            delegate = await manager.__aenter__()
            await delegate.setup()
            self._context_manager = manager
            self._delegate = delegate
            return delegate

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await (await self._get_delegate()).aget_tuple(config)

    async def setup(self) -> None:
        """显式初始化数据库；正常运行也会在首次访问时自动执行。"""

        await self._get_delegate()

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        delegate = await self._get_delegate()
        async for item in delegate.alist(
            config,
            filter=filter,
            before=before,
            limit=limit,
        ):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict[str, str | int | float],
    ) -> RunnableConfig:
        return await (await self._get_delegate()).aput(
            config,
            checkpoint,
            metadata,
            new_versions,
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await (await self._get_delegate()).aput_writes(
            config,
            writes,
            task_id,
            task_path,
        )

    async def adelete_thread(self, thread_id: str) -> None:
        await (await self._get_delegate()).adelete_thread(thread_id)

    async def close(self) -> None:
        """关闭底层 aiosqlite 连接。"""

        if self._context_manager is not None:
            await self._context_manager.__aexit__(None, None, None)
            self._context_manager = None
            self._delegate = None

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        del config
        raise NotImplementedError("仅支持异步 Agent 执行")

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        del config, filter, before, limit
        raise NotImplementedError("仅支持异步 Agent 执行")

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict[str, str | int | float],
    ) -> RunnableConfig:
        del config, checkpoint, metadata, new_versions
        raise NotImplementedError("仅支持异步 Agent 执行")

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        del config, writes, task_id, task_path
        raise NotImplementedError("仅支持异步 Agent 执行")
