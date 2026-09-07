"""买家长期偏好领域模型。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum


class PreferenceKind(str, Enum):
    """区分正向偏好和不可被 Top-K 截断的负向约束。"""

    LIKE = "like"
    DISLIKE = "dislike"


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class BuyerPreference:
    """一条以买家为作用域、可以跨会话复用的稳定偏好。"""

    buyer_id: str
    kind: PreferenceKind
    statement: str
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        """规范化文本并拒绝无法安全分区的偏好。"""

        buyer_id = self.buyer_id.strip()
        statement = self.statement.strip()
        if not buyer_id:
            raise ValueError("buyer_id 不能为空")
        if not statement:
            raise ValueError("偏好内容不能为空")
        if len(statement) > 200:
            raise ValueError("偏好内容不能超过 200 个字符")
        object.__setattr__(self, "buyer_id", buyer_id)
        object.__setattr__(self, "statement", statement)
        if self.updated_at < self.created_at:
            raise ValueError("updated_at 不能早于 created_at")

    @property
    def preference_id(self) -> str:
        """生成不暴露原文且可以稳定去重的偏好标识。"""

        digest = hashlib.sha256(
            f"{self.buyer_id}\n{self.statement}".encode()
        ).hexdigest()
        return f"pref-{digest[:24]}"

    def to_dict(self) -> dict[str, str]:
        """转换为存储、工具返回和事件记录共用的稳定字典。"""

        return {
            "preference_id": self.preference_id,
            "buyer_id": self.buyer_id,
            "kind": self.kind.value,
            "statement": self.statement,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
