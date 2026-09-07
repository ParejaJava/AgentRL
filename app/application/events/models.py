"""与具体消息中间件无关的会话事件契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class TradeEventType(str, Enum):
    """API、Worker 和前端共同理解的稳定事件类型。"""

    RUN_STARTED = "run.started"
    AGENT_DISPATCH = "agent.dispatch"
    TOOL_INVOKE = "tool.invoke"
    TOOL_RESULT = "tool.result"
    TOKEN_DELTA = "token.delta"
    PLAN_UPDATE = "plan.update"
    CONTEXT_COMPRESSED = "context.compressed"
    SAFETY_BLOCKED = "safety.blocked"
    CACHE_HIT = "cache.hit"
    BUDGET_EXHAUSTED = "budget.exhausted"
    DRIFT_DETECTED = "agent.drift_detected"
    FINAL_RESULT = "final.result"
    RUN_FINISHED = "run.finished"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class TradeEvent:
    """按照 shopping_session_id 分区的可序列化事件。"""

    shopping_session_id: str
    type: TradeEventType
    payload: dict[str, Any]
    event_id: str = field(default_factory=lambda: str(uuid4()))
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """转换为 HTTP、WebSocket 或消息队列可传输的字典。"""

        return {
            "event_id": self.event_id,
            "shopping_session_id": self.shopping_session_id,
            "type": self.type.value,
            "payload": self.payload,
            "occurred_at": self.occurred_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TradeEvent:
        """从外部传输格式重建事件，并校验事件类型。"""

        return cls(
            event_id=str(raw["event_id"]),
            shopping_session_id=str(raw["shopping_session_id"]),
            type=TradeEventType(str(raw["type"])),
            payload=dict(raw.get("payload", {})),
            occurred_at=datetime.fromisoformat(str(raw["occurred_at"])),
        )
