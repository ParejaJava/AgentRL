"""使用 SQLite WAL 持久化跨会话买家偏好。"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime
from pathlib import Path

from app.domain.buyer import BuyerPreference, PreferenceKind


class SQLitePreferenceStore:
    """按 buyer_id 分区、以原文精确去重的偏好仓储。"""

    def __init__(self, database_path: Path, *, max_likes_per_buyer: int = 50) -> None:
        if max_likes_per_buyer < 1:
            raise ValueError("max_likes_per_buyer 必须大于 0")
        self._path = database_path
        self._max_likes_per_buyer = max_likes_per_buyer

    async def save(self, preference: BuyerPreference) -> BuyerPreference:
        """幂等保存偏好；同买家同原文再次写入时更新类型和时间。"""

        return await asyncio.to_thread(self._save_sync, preference)

    async def list_by_buyer(self, buyer_id: str) -> tuple[BuyerPreference, ...]:
        """在线程池中读取一个买家的全部长期偏好。"""

        return await asyncio.to_thread(self.list_by_buyer_sync, buyer_id)

    async def delete_exact(self, buyer_id: str, statement: str) -> bool:
        """按规范化后的完整原文精确删除，避免相似语句被误删。"""

        return await asyncio.to_thread(self._delete_exact_sync, buyer_id, statement)

    def list_by_buyer_sync(self, buyer_id: str) -> tuple[BuyerPreference, ...]:
        """为在线同步商品检索提供只读路径。"""

        normalized_buyer = buyer_id.strip()
        if not normalized_buyer:
            raise ValueError("buyer_id 不能为空")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT buyer_id, kind, statement, created_at, updated_at
                FROM buyer_preferences
                WHERE buyer_id = ?
                ORDER BY created_at ASC, preference_id ASC
                """,
                (normalized_buyer,),
            ).fetchall()
        return tuple(self._decode(row) for row in rows)

    def _connect(self) -> sqlite3.Connection:
        """创建带 WAL 和 busy timeout 的短生命周期连接。"""

        self._path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS buyer_preferences (
                preference_id TEXT PRIMARY KEY,
                buyer_id TEXT NOT NULL,
                kind TEXT NOT NULL CHECK (kind IN ('like', 'dislike')),
                statement TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (buyer_id, statement)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_buyer_preferences_buyer
            ON buyer_preferences (buyer_id, updated_at DESC)
            """
        )
        return connection

    def _save_sync(self, preference: BuyerPreference) -> BuyerPreference:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT created_at
                FROM buyer_preferences
                WHERE buyer_id = ? AND statement = ?
                """,
                (preference.buyer_id, preference.statement),
            ).fetchone()
            created_at = (
                datetime.fromisoformat(str(existing[0]))
                if existing is not None
                else preference.created_at
            )
            persisted = BuyerPreference(
                buyer_id=preference.buyer_id,
                kind=preference.kind,
                statement=preference.statement,
                created_at=created_at,
                updated_at=preference.updated_at,
            )
            connection.execute(
                """
                INSERT INTO buyer_preferences (
                    preference_id, buyer_id, kind, statement, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (buyer_id, statement) DO UPDATE SET
                    kind = excluded.kind,
                    updated_at = excluded.updated_at
                """,
                (
                    persisted.preference_id,
                    persisted.buyer_id,
                    persisted.kind.value,
                    persisted.statement,
                    persisted.created_at.isoformat(),
                    persisted.updated_at.isoformat(),
                ),
            )
            self._prune_old_likes(connection, persisted.buyer_id)
            connection.commit()
        return persisted

    def _prune_old_likes(
        self,
        connection: sqlite3.Connection,
        buyer_id: str,
    ) -> None:
        """只淘汰超限的旧 like，永远不自动删除 dislike/黑名单。"""

        connection.execute(
            """
            DELETE FROM buyer_preferences
            WHERE preference_id IN (
                SELECT preference_id
                FROM buyer_preferences
                WHERE buyer_id = ? AND kind = 'like'
                ORDER BY updated_at DESC, preference_id DESC
                LIMIT -1 OFFSET ?
            )
            """,
            (buyer_id, self._max_likes_per_buyer),
        )

    def _delete_exact_sync(self, buyer_id: str, statement: str) -> bool:
        normalized_buyer = buyer_id.strip()
        normalized_statement = statement.strip()
        if not normalized_buyer or not normalized_statement:
            raise ValueError("buyer_id 和偏好原文不能为空")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM buyer_preferences
                WHERE buyer_id = ? AND statement = ?
                """,
                (normalized_buyer, normalized_statement),
            )
            connection.commit()
        return cursor.rowcount > 0

    @staticmethod
    def _decode(row: tuple[object, ...]) -> BuyerPreference:
        return BuyerPreference(
            buyer_id=str(row[0]),
            kind=PreferenceKind(str(row[1])),
            statement=str(row[2]),
            created_at=datetime.fromisoformat(str(row[3])),
            updated_at=datetime.fromisoformat(str(row[4])),
        )
