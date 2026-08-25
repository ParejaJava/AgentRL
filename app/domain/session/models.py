"""框架无关的会话快照。"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.domain.runtime import RunId


@dataclass(frozen=True, slots=True)
class RunCheckpoint:
    """能够恢复 Agent Run 的领域快照。"""

    checkpoint_id: str
    run_id: RunId
    revision: int
    runtime_state: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
