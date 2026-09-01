"""使用 SQLite WAL 持久化会话级 TaskBoard。"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

from app.application.tasking import (
    TaskBoard,
    TaskBoardConflictError,
    TaskRecord,
    TaskStatus,
)


class SQLiteTaskBoardRepository:
    """以 thread_id 为键，使用乐观版本保护任务看板并发写入。"""

    def __init__(self, database_path: Path) -> None:
        self._path = database_path

    async def load(self, scope_id: str) -> TaskBoard:
        """在线程池中读取一个看板快照。"""

        return await asyncio.to_thread(self._load_sync, scope_id)

    async def save(
        self,
        board: TaskBoard,
        *,
        expected_version: int,
    ) -> TaskBoard:
        """仅当版本未变化时提交完整看板。"""

        return await asyncio.to_thread(
            self._save_sync,
            board,
            expected_version,
        )

    def _connect(self) -> sqlite3.Connection:
        """创建支持跨线程并发访问的短生命周期连接。"""

        self._path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS task_boards (
                scope_id TEXT PRIMARY KEY,
                version INTEGER NOT NULL,
                next_numeric_id INTEGER NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        return connection

    def _load_sync(self, scope_id: str) -> TaskBoard:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT version, next_numeric_id, payload
                FROM task_boards
                WHERE scope_id = ?
                """,
                (scope_id,),
            ).fetchone()
        if row is None:
            return TaskBoard(scope_id=scope_id)
        return self._decode_board(
            scope_id=scope_id,
            version=int(row[0]),
            next_numeric_id=int(row[1]),
            payload=str(row[2]),
        )

    def _save_sync(
        self,
        board: TaskBoard,
        expected_version: int,
    ) -> TaskBoard:
        new_version = expected_version + 1
        payload = json.dumps(
            [self._encode_task(task) for task in board.tasks.values()],
            ensure_ascii=False,
            sort_keys=True,
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT version FROM task_boards WHERE scope_id = ?",
                (board.scope_id,),
            ).fetchone()
            current_version = int(row[0]) if row is not None else 0
            if current_version != expected_version:
                connection.rollback()
                raise TaskBoardConflictError(
                    f"任务看板版本冲突：期望 {expected_version}，"
                    f"实际 {current_version}"
                )
            if row is None:
                connection.execute(
                    """
                    INSERT INTO task_boards (
                        scope_id, version, next_numeric_id, payload
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        board.scope_id,
                        new_version,
                        board.next_numeric_id,
                        payload,
                    ),
                )
            else:
                connection.execute(
                    """
                    UPDATE task_boards
                    SET version = ?, next_numeric_id = ?, payload = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE scope_id = ? AND version = ?
                    """,
                    (
                        new_version,
                        board.next_numeric_id,
                        payload,
                        board.scope_id,
                        expected_version,
                    ),
                )
            connection.commit()
        board.version = new_version
        return board

    @staticmethod
    def _encode_task(task: TaskRecord) -> dict[str, Any]:
        return {
            "id": task.id,
            "subject": task.subject,
            "description": task.description,
            "status": task.status.value,
            "owner": task.owner,
            "blocked_by": list(task.blocked_by),
            "result": task.result,
            "error": task.error,
            "metadata": task.metadata,
            "created_at": task.created_at,
            "updated_at": task.updated_at,
        }

    @staticmethod
    def _decode_board(
        *,
        scope_id: str,
        version: int,
        next_numeric_id: int,
        payload: str,
    ) -> TaskBoard:
        raw_tasks = json.loads(payload)
        tasks = {
            str(raw["id"]): TaskRecord(
                id=str(raw["id"]),
                subject=str(raw["subject"]),
                description=str(raw["description"]),
                status=TaskStatus(str(raw["status"])),
                owner=str(raw["owner"]) if raw.get("owner") else None,
                blocked_by=[str(item) for item in raw.get("blocked_by", [])],
                result=str(raw["result"]) if raw.get("result") is not None else None,
                error=str(raw["error"]) if raw.get("error") is not None else None,
                metadata=dict(raw.get("metadata", {})),
                created_at=str(raw["created_at"]),
                updated_at=str(raw["updated_at"]),
            )
            for raw in raw_tasks
        }
        return TaskBoard(
            scope_id=scope_id,
            tasks=tasks,
            next_numeric_id=next_numeric_id,
            version=version,
        )
