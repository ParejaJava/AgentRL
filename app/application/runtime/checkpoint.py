"""与存储实现无关的 Agent Run 快照。"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .models import RunId


@dataclass(frozen=True, slots=True)
class RunCheckpoint:
    """可由基础设施持久化并用于恢复执行的快照。"""

    checkpoint_id: str
    run_id: RunId
    revision: int
    runtime_state: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
