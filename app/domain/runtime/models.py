"""Agent Runtime 不可依赖具体执行框架的聚合与值对象。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import NewType

AgentId = NewType("AgentId", str)
RunId = NewType("RunId", str)
ThreadId = NewType("ThreadId", str)


class RunStatus(str, Enum):
    """Agent Run 的领域生命周期。"""

    CREATED = "created"
    RUNNING = "running"
    WAITING_TOOL = "waiting_tool"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """一个与 LangChain 无关的 Agent 能力定义。"""

    agent_id: AgentId
    name: str
    capability_names: frozenset[str] = frozenset()
    tool_names: frozenset[str] = frozenset()


@dataclass(slots=True)
class AgentRun:
    """由 Runtime Kernel 管理的 Agent 执行聚合。"""

    run_id: RunId
    thread_id: ThreadId
    agent_id: AgentId
    status: RunStatus = RunStatus.CREATED
    step_count: int = 0
    child_run_ids: list[RunId] = field(default_factory=list)
    failure_reason: str | None = None

    def start(self) -> None:
        """从 CREATED 进入 RUNNING。"""

        if self.status is not RunStatus.CREATED:
            raise ValueError("只有 CREATED 状态的 Run 可以启动")
        self.status = RunStatus.RUNNING

    def complete(self) -> None:
        """完成一个仍在执行的 Run。"""

        if self.status not in {RunStatus.RUNNING, RunStatus.WAITING_TOOL}:
            raise ValueError("当前 Run 状态不允许完成")
        self.status = RunStatus.COMPLETED

    def fail(self, reason: str) -> None:
        """记录不可恢复失败并终止 Run。"""

        if self.status in {RunStatus.COMPLETED, RunStatus.CANCELLED}:
            raise ValueError("已经结束的 Run 不能标记为失败")
        self.status = RunStatus.FAILED
        self.failure_reason = reason
