"""基于 SQLite 的 append-only 会话事件日志。"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, messages_from_dict

from .messages import message_event_id, message_payload, message_text
from .schemas import EpochBaseline, TaskState, TokenLedger


@dataclass(frozen=True, slots=True)
class EventRecord:
    """表示一条准备写入 Event Log 的不可变事件。"""

    event_id: str
    thread_id: str
    agent_id: str
    sequence: int
    event_type: str
    payload: dict[str, Any]
    cache_epoch: int
    message_id: str | None = None
    tool_call_id: str | None = None
    artifact_ref: str | None = None


class SQLiteEventStore:
    """为单个会话目录维护 SQLite 事件日志。"""

    def __init__(self, session_dir: Path) -> None:
        self._path = session_dir / "events.db"

    async def append(self, event: EventRecord) -> None:
        """异步追加事件；相同 event_id 的重放不会重复写入。"""

        await asyncio.to_thread(self._append_sync, event)

    async def append_messages(
        self,
        *,
        thread_id: str,
        agent_id: str,
        messages: list[BaseMessage],
        cache_epoch: int,
    ) -> None:
        """把当前消息状态幂等同步到事件日志。"""

        events = [
            EventRecord(
                event_id=message_event_id(thread_id, position, message),
                thread_id=thread_id,
                agent_id=agent_id,
                sequence=position,
                event_type=f"message.{message.type}",
                message_id=message.id,
                tool_call_id=getattr(message, "tool_call_id", None),
                payload=message_payload(message),
                cache_epoch=cache_epoch,
            )
            for position, message in enumerate(messages)
        ]
        await asyncio.to_thread(self._append_many_sync, events)

    async def list_events(self) -> list[dict[str, Any]]:
        """按插入顺序读取当前会话的全部事件。"""

        return await asyncio.to_thread(self._list_events_sync)

    async def rebuild_state(self) -> dict[str, Any]:
        """从完整事件日志恢复可继续交给 LangGraph 的上下文状态。"""

        events = await self.list_events()
        message_events = sorted(
            (
                event
                for event in events
                if str(event["event_type"]).startswith("message.")
            ),
            key=lambda event: int(event["sequence"]),
        )
        messages = messages_from_dict([event["payload"] for event in message_events])
        first_user = next(
            (
                message_text(message)
                for message in messages
                if isinstance(message, HumanMessage)
            ),
            "",
        )
        task_state = TaskState(goal=first_user)
        working_memory: dict[str, dict[str, Any]] = {}
        compressed_ids: set[str] = set()
        cold_refs: list[str] = []
        baseline = EpochBaseline(goal=first_user)
        ledger = TokenLedger()
        cache_epoch = 0

        for event in events:
            event_type = event["event_type"]
            payload = event["payload"]
            cache_epoch = max(cache_epoch, int(event["cache_epoch"]))
            if event_type == "compression.incremental_summary":
                task_state = TaskState.model_validate(
                    payload.get("task_state", task_state.model_dump())
                )
                working_memory = dict(payload.get("working_memory", working_memory))
                compressed_ids.update(payload.get("compressed_event_ids", []))
            elif event_type == "cache.epoch_roll":
                baseline = EpochBaseline.model_validate(
                    payload.get("baseline", baseline.model_dump())
                )
            elif event_type == "tool.offload" and payload.get("artifact_ref"):
                cold_refs.append(str(payload["artifact_ref"]))
            elif event_type == "model.usage":
                ledger.model_calls += 1
                ledger.input_tokens += int(payload.get("input_tokens", 0))
                ledger.output_tokens += int(payload.get("output_tokens", 0))
                ledger.cached_input_tokens += int(payload.get("cached_input_tokens", 0))
                ledger.cache_write_tokens += int(payload.get("cache_write_tokens", 0))

        return {
            "messages": messages,
            "task_state": task_state.model_dump(mode="json"),
            "task_delta": {},
            "working_memory": working_memory,
            "epoch_baseline": baseline.model_dump(mode="json"),
            "cache_epoch": cache_epoch,
            "cache_breakpoints": [],
            "compressed_event_ids": sorted(compressed_ids),
            "cold_event_refs": list(dict.fromkeys(cold_refs)),
            "token_ledger": ledger.model_dump(mode="json"),
            "context_version": len(events),
        }

    def _connect(self) -> sqlite3.Connection:
        """创建连接并确保父目录及表结构存在。"""

        self._path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                thread_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                message_id TEXT,
                tool_call_id TEXT,
                payload TEXT NOT NULL,
                artifact_ref TEXT,
                cache_epoch INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        return connection

    def _append_sync(self, event: EventRecord) -> None:
        with closing(self._connect()) as connection, connection:
            self._insert(connection, event)

    def _append_many_sync(self, events: list[EventRecord]) -> None:
        with closing(self._connect()) as connection, connection:
            for event in events:
                self._insert(connection, event)

    @staticmethod
    def _insert(connection: sqlite3.Connection, event: EventRecord) -> None:
        connection.execute(
            """
            INSERT OR IGNORE INTO events (
                event_id, thread_id, agent_id, sequence, event_type,
                message_id, tool_call_id, payload, artifact_ref,
                cache_epoch, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.thread_id,
                event.agent_id,
                event.sequence,
                event.event_type,
                event.message_id,
                event.tool_call_id,
                json.dumps(event.payload, ensure_ascii=False, default=str),
                event.artifact_ref,
                event.cache_epoch,
                datetime.now(UTC).isoformat(),
            ),
        )

    def _list_events_sync(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                """
                SELECT event_id, thread_id, agent_id, sequence, event_type,
                       message_id, tool_call_id, payload, artifact_ref,
                       cache_epoch, created_at
                FROM events
                ORDER BY row_id
                """
            ).fetchall()
        return [
            {
                "event_id": row[0],
                "thread_id": row[1],
                "agent_id": row[2],
                "sequence": row[3],
                "event_type": row[4],
                "message_id": row[5],
                "tool_call_id": row[6],
                "payload": json.loads(row[7]),
                "artifact_ref": row[8],
                "cache_epoch": row[9],
                "created_at": row[10],
            }
            for row in rows
        ]
