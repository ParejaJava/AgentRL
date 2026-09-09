"""与具体 Agent 框架无关的任务看板模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal


def utc_now_iso() -> str:
    """生成可直接持久化和传输的 UTC 时间。"""

    return datetime.now(UTC).isoformat()


class TaskStatus(StrEnum):
    """任务实际持久化的生命周期状态。"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


EffectiveTaskStatus = Literal[
    "pending",
    "blocked",
    "in_progress",
    "completed",
    "failed",
]


@dataclass(slots=True)
class TaskRecord:
    """任务仓储中的单一事实；只持久化 `blocked_by` 方向的依赖。"""

    id: str
    subject: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    owner: str | None = None
    lease_expires_at: str | None = None
    blocked_by: list[str] = field(default_factory=list)
    result: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)


@dataclass(slots=True)
class TaskBoard:
    """一个 LangGraph thread 对应的完整任务看板快照。"""

    scope_id: str
    tasks: dict[str, TaskRecord] = field(default_factory=dict)
    next_numeric_id: int = 1
    version: int = 0


@dataclass(frozen=True, slots=True)
class TaskView:
    """向工具层返回的任务视图，包含动态计算的反向依赖和阻塞状态。"""

    id: str
    subject: str
    description: str
    status: TaskStatus
    effective_status: EffectiveTaskStatus
    owner: str | None
    lease_expires_at: str | None
    blocked_by: tuple[str, ...]
    active_blocked_by: tuple[str, ...]
    blocks: tuple[str, ...]
    runnable: bool
    result: str | None
    error: str | None
    metadata: dict[str, Any]
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        """转换为工具和接口层可直接序列化的数据。"""

        return {
            "id": self.id,
            "subject": self.subject,
            "description": self.description,
            "status": self.status.value,
            "effective_status": self.effective_status,
            "owner": self.owner,
            "lease_expires_at": self.lease_expires_at,
            "blocked_by": list(self.blocked_by),
            "active_blocked_by": list(self.active_blocked_by),
            "blocks": list(self.blocks),
            "runnable": self.runnable,
            "result": self.result,
            "error": self.error,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True, slots=True)
class TaskWorkItem:
    """已经被原子认领、准备交给子 Agent 的工作单元。"""

    task_id: str
    subject: str
    description: str
    metadata: dict[str, Any]
    owner: str

    def to_prompt(self) -> str:
        """生成上下文独立的子 Agent 任务描述。"""

        metadata_line = f"\n任务元数据：{self.metadata}" if self.metadata else ""
        return (
            f"任务 ID：{self.task_id}\n"
            f"任务标题：{self.subject}\n"
            f"任务描述：{self.description}{metadata_line}\n"
            "请独立完成该任务，只返回最终结论、必要证据和错误信息。"
        )


@dataclass(frozen=True, slots=True)
class TaskExecutionOutcome:
    """子 Agent 对一个任务的隔离执行结果。"""

    task_id: str
    succeeded: bool
    result: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class TaskDispatchReport:
    """一次确定性任务派发的最终报告。"""

    dispatch_id: str
    reason: str
    tasks: tuple[TaskView, ...]

    def to_dict(self) -> dict[str, Any]:
        """转换为 fork 工具可以直接返回的结构。"""

        return {
            "dispatch_id": self.dispatch_id,
            "fork_reason": self.reason,
            "tasks": [task.to_dict() for task in self.tasks],
        }
