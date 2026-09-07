"""以 Redis 共享工具熔断状态，并在 Redis 故障时放行本地保护。"""

from __future__ import annotations

import logging
import math
import time
from typing import Protocol

logger = logging.getLogger(__name__)


class SharedCircuitBreaker(Protocol):
    """ToolResilienceMiddleware 所需的跨实例熔断最小接口。"""

    async def allow(self, tool_name: str) -> bool: ...

    async def record_success(self, tool_name: str) -> None: ...

    async def record_failure(self, tool_name: str) -> None: ...


class RedisSharedCircuitBreaker:
    """让多个 API/worker 副本共享失败次数、打开时间和半开探针。"""

    _FAILURE_SCRIPT = """
    local failures = redis.call('HINCRBY', KEYS[1], 'failures', 1)
    if failures >= tonumber(ARGV[1]) then
      redis.call('HSET', KEYS[1], 'opened_at', ARGV[2])
    end
    redis.call('EXPIRE', KEYS[1], ARGV[3])
    return failures
    """

    def __init__(
        self,
        redis_url: str,
        *,
        prefix: str,
        failure_threshold: int,
        recovery_seconds: float,
    ) -> None:
        try:
            from redis.asyncio import Redis
        except ImportError as exc:
            raise RuntimeError(
                "已启用 Redis 共享熔断；请运行 `uv sync --extra platform`"
            ) from exc
        self._redis = Redis.from_url(redis_url, decode_responses=True)
        self._prefix = prefix.rstrip(":")
        self._threshold = failure_threshold
        self._recovery = recovery_seconds
        self._ttl = max(60, math.ceil(recovery_seconds * 10))

    async def allow(self, tool_name: str) -> bool:
        """关闭时放行；冷却期结束后只允许一个跨实例半开探针。"""

        try:
            key = self._key(tool_name)
            opened_at = await self._redis.hget(key, "opened_at")
            if opened_at is None:
                return True
            if time.time() - float(opened_at) < self._recovery:
                return False
            return bool(
                await self._redis.set(
                    f"{key}:half-open",
                    "1",
                    nx=True,
                    ex=max(1, math.ceil(self._recovery)),
                )
            )
        except Exception as exc:  # noqa: BLE001 - 状态存储故障不能放大为业务事故。
            logger.warning("共享熔断读取失败，回退本地保护：%s", exc)
            return True

    async def record_success(self, tool_name: str) -> None:
        """成功关闭熔断并释放可能存在的半开探针。"""

        try:
            key = self._key(tool_name)
            await self._redis.delete(key, f"{key}:half-open")
        except Exception as exc:  # noqa: BLE001
            logger.warning("共享熔断成功状态写入失败：%s", exc)

    async def record_failure(self, tool_name: str) -> None:
        """原子累计失败，并在达到阈值时记录打开时刻。"""

        try:
            key = self._key(tool_name)
            await self._redis.eval(
                self._FAILURE_SCRIPT,
                1,
                key,
                self._threshold,
                time.time(),
                self._ttl,
            )
            await self._redis.delete(f"{key}:half-open")
        except Exception as exc:  # noqa: BLE001
            logger.warning("共享熔断失败状态写入失败：%s", exc)

    def _key(self, tool_name: str) -> str:
        return f"{self._prefix}:{tool_name}"
