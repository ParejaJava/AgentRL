"""Redis Stream Agent 任务队列。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class QueuedAgentTask:
    """Worker 从 Stream 读取的一条可确认任务。"""

    stream_id: str
    stream: str
    task_id: str
    payload: dict[str, Any]
    attempts: int


class RedisStreamTaskQueue:
    """提供幂等提交、consumer group、重试和死信的任务队列。"""

    def __init__(
        self,
        redis_url: str,
        *,
        stream: str = "globex:agent:jobs",
        group: str = "globex-workers",
        dead_letter_stream: str = "globex:agent:jobs:dead",
        max_attempts: int = 3,
        large_stream: str | None = None,
        large_request_turns: int = 30,
    ) -> None:
        try:
            from redis.asyncio import Redis
        except ImportError as exc:
            raise RuntimeError(
                "已启用 Redis；请运行 `uv sync --extra platform`"
            ) from exc
        self._redis = Redis.from_url(redis_url, decode_responses=True)
        self._stream = stream
        self._large_stream = large_stream or f"{stream}:large"
        self._group = group
        self._dead = dead_letter_stream
        self._max_attempts = max_attempts
        self._large_request_turns = max(1, large_request_turns)

    async def initialize(self) -> None:
        """幂等创建 consumer group。"""

        for stream in (self._stream, self._large_stream):
            try:
                await self._redis.xgroup_create(
                    stream,
                    self._group,
                    id="0",
                    mkstream=True,
                )
            except Exception as exc:
                if "BUSYGROUP" not in str(exc):
                    raise

    async def enqueue(self, task_id: str, payload: dict[str, object]) -> bool:
        """提交任务；同 task_id 重复提交不产生第二条 Stream 消息。"""

        claimed = await self._redis.set(
            f"{self._stream}:idempotency:{task_id}",
            "1",
            nx=True,
            ex=86_400,
        )
        if not claimed:
            return False
        thread_id = str(payload.get("thread_id", task_id))
        turn_key = f"{self._stream}:turns:{thread_id}"
        turn_count = int(await self._redis.incr(turn_key))
        await self._redis.expire(turn_key, 86_400)
        target_stream = (
            self._large_stream
            if turn_count >= self._large_request_turns
            else self._stream
        )
        await self._redis.hset(
            f"{self._stream}:status:{task_id}",
            mapping={
                "status": "queued",
                "attempts": "0",
                "priority": "large" if target_stream == self._large_stream else "normal",
            },
        )
        await self._redis.xadd(
            target_stream,
            {"task_id": task_id, "payload": json.dumps(payload, ensure_ascii=False), "attempts": "0"},
        )
        return True

    async def consume(
        self,
        consumer: str,
        *,
        block_ms: int = 5000,
    ) -> QueuedAgentTask | None:
        """从 consumer group 读取一条新任务。"""

        await self.initialize()
        # 先无阻塞读取普通流；为空时再阻塞等待大请求流，避免一次返回两条流时
        # 只消费第一条而把另一条遗留在 pending entries list。
        rows = await self._redis.xreadgroup(
            self._group,
            consumer,
            {self._stream: ">"},
            count=1,
            block=1,
        )
        if not rows:
            rows = await self._redis.xreadgroup(
                self._group,
                consumer,
                {self._large_stream: ">"},
                count=1,
                block=block_ms,
            )
        if not rows:
            return None
        source_stream, entries = rows[0]
        stream_id, fields = entries[0]
        task = QueuedAgentTask(
            stream_id=stream_id,
            stream=str(source_stream),
            task_id=str(fields["task_id"]),
            payload=json.loads(str(fields["payload"])),
            attempts=int(fields.get("attempts", 0)),
        )
        await self._redis.hset(
            f"{self._stream}:status:{task.task_id}",
            mapping={"status": "running", "attempts": str(task.attempts)},
        )
        return task

    async def acknowledge(self, task: QueuedAgentTask, result: object) -> None:
        """确认成功并保存可查询的最终结果。"""

        await self._redis.xack(task.stream, self._group, task.stream_id)
        await self._redis.hset(
            f"{self._stream}:status:{task.task_id}",
            mapping={
                "status": "completed",
                "result": json.dumps(result, ensure_ascii=False),
            },
        )

    async def fail(self, task: QueuedAgentTask, error: str) -> None:
        """确认失败消息；未耗尽重试则重入队，否则送入死信流。"""

        attempts = task.attempts + 1
        await self._redis.xack(task.stream, self._group, task.stream_id)
        target = self._dead if attempts >= self._max_attempts else task.stream
        await self._redis.xadd(
            target,
            {
                "task_id": task.task_id,
                "payload": json.dumps(task.payload, ensure_ascii=False),
                "attempts": str(attempts),
                "error": error[:1000],
            },
        )
        await self._redis.hset(
            f"{self._stream}:status:{task.task_id}",
            mapping={
                "status": "dead_letter" if target == self._dead else "retrying",
                "attempts": str(attempts),
                "error": error[:1000],
            },
        )

    async def status(self, task_id: str) -> dict[str, str]:
        """读取任务状态；不存在时返回空字典。"""

        return dict(await self._redis.hgetall(f"{self._stream}:status:{task_id}"))

    async def ping(self) -> bool:
        return bool(await self._redis.ping())

    async def close(self) -> None:
        await self._redis.aclose()
