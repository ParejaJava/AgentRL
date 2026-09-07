"""带严格业务绕过规则的 Redis 语义响应缓存。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re

from app.application.agents.run_agent import AgentCacheLookup
from app.application.catalog.ports import EmbeddingEncoder
from app.application.runtime import ShoppingContextSnapshot
from app.infrastructure.retrieval.item_search.user_tower import UserProfileSource

logger = logging.getLogger(__name__)

_UNSAFE = re.compile(
    r"下单|购买|付款|支付|取消|退款|改地址|记住|忘记|"
    r"刚才|刚刚|上面|前面|那个|这个|它|我的订单|订单号|GBX-",
    re.IGNORECASE,
)


class RedisSemanticResponseCache:
    """只复用首轮只读咨询，Redis 故障时无条件回退 Agent 主链。"""

    def __init__(
        self,
        redis_url: str,
        encoder: EmbeddingEncoder,
        profile_source: UserProfileSource,
        *,
        namespace: str,
        threshold: float = 0.95,
        ttl_seconds: int = 86_400,
        bucket_limit: int = 30,
    ) -> None:
        try:
            from redis.asyncio import Redis
        except ImportError as exc:
            raise RuntimeError(
                "已启用语义缓存；请运行 `uv sync --extra platform`"
            ) from exc
        self._redis = Redis.from_url(redis_url, decode_responses=True)
        self._encoder = encoder
        self._profiles = profile_source
        self._namespace = namespace
        self._threshold = threshold
        self._ttl = ttl_seconds
        self._limit = bucket_limit

    async def lookup(
        self,
        message: str,
        shopping: ShoppingContextSnapshot,
        thread_id: str,
    ) -> AgentCacheLookup:
        """只在线程首轮、只读问句和有效偏好 scope 下查找近邻回答。"""

        if _UNSAFE.search(message):
            return AgentCacheLookup(None, False)
        try:
            first_turn = await self._redis.set(
                f"{self._namespace}:thread:{thread_id}",
                "1",
                nx=True,
                ex=self._ttl,
            )
            if not first_turn:
                return AgentCacheLookup(None, False)
            profile = await asyncio.to_thread(
                self._profiles.get_profile,
                shopping.buyer_id,
                "semantic-cache",
            )
            scope = profile or ""
            entries = await self._load_entries(shopping.buyer_id, scope)
            if not entries:
                return AgentCacheLookup(None, True)
            vector = await self._embed(message)
        except Exception as exc:  # noqa: BLE001 - 缓存不得拖垮主链。
            logger.warning("语义缓存读取失败，按未命中处理：%s", exc)
            return AgentCacheLookup(None, False)

        best_reply: str | None = None
        best_similarity = self._threshold
        for entry in entries:
            similarity = _cosine(vector, entry.get("vector", []))
            if similarity >= best_similarity and entry.get("reply"):
                best_similarity = similarity
                best_reply = str(entry["reply"])
        return AgentCacheLookup(
            best_reply,
            True,
            best_similarity if best_reply else None,
        )

    async def remember(
        self,
        message: str,
        reply: str,
        shopping: ShoppingContextSnapshot,
        *,
        eligible: bool,
    ) -> None:
        """只写入成功的首轮只读回答，并限制每个买家/偏好桶大小。"""

        if not eligible or not reply or _UNSAFE.search(message):
            return
        try:
            profile = await asyncio.to_thread(
                self._profiles.get_profile,
                shopping.buyer_id,
                "semantic-cache",
            )
            scope = profile or ""
            key = self._bucket_key(shopping.buyer_id, scope)
            entries = await self._load_entries(shopping.buyer_id, scope)
            entries.append(
                {
                    "query": _normalize(message),
                    "reply": reply,
                    "vector": await self._embed(message),
                }
            )
            await self._redis.setex(
                key,
                self._ttl,
                json.dumps(entries[-self._limit :], ensure_ascii=False),
            )
        except Exception as exc:  # noqa: BLE001 - 缓存写失败不影响成功响应。
            logger.warning("语义缓存写入失败，已跳过：%s", exc)

    async def _load_entries(self, buyer_id: str, scope: str) -> list[dict[str, object]]:
        raw = await self._redis.get(self._bucket_key(buyer_id, scope))
        if raw is None:
            return []
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []

    async def _embed(self, text: str) -> list[float]:
        vectors = await asyncio.to_thread(
            self._encoder.embed_queries,
            [_normalize(text)],
        )
        return [float(value) for value in vectors[0]]

    def _bucket_key(self, buyer_id: str, scope: str) -> str:
        digest = hashlib.sha256(
            f"{self._namespace}\0{buyer_id}\0{scope}".encode()
        ).hexdigest()[:24]
        return f"{self._namespace}:buyer:{digest}"


def _normalize(query: str) -> str:
    return re.sub(r"\s+", "", query.strip()).rstrip("？?。.!！~")


def _cosine(left: list[float], right: object) -> float:
    if not isinstance(right, list) or len(left) != len(right):
        return 0.0
    values = [float(item) for item in right]
    dot = sum(a * b for a, b in zip(left, values, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in values))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
