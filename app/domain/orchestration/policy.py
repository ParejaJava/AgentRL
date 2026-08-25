"""不依赖并发库和 Agent 框架的 fork 决策。"""

from dataclasses import dataclass
from typing import Literal

ForkReason = Literal["parallel", "context_isolation", "deep_chain"]


@dataclass(frozen=True, slots=True)
class ForkPlan:
    """Orchestrator 已批准的不可变子任务计划。"""

    tasks: tuple[str, ...]
    reason: ForkReason


class OrchestrationPolicy:
    """校验并规范化 fork 请求，具体并发方式由外层 Driver 决定。"""

    def create_fork_plan(
        self,
        tasks: list[str],
        reason: ForkReason,
    ) -> ForkPlan:
        """拒绝空任务并去除重复子任务。"""

        normalized = tuple(
            dict.fromkeys(task.strip() for task in tasks if task.strip())
        )
        if not normalized:
            raise ValueError("fork 至少需要一个非空子任务")
        return ForkPlan(tasks=normalized, reason=reason)


def should_fork(
    *,
    can_run_parallel: bool,
    needs_context_isolation: bool,
    call_depth: int,
) -> bool:
    """满足并行、隔离或深调用链任一条件时允许 fork。"""

    return can_run_parallel or needs_context_isolation or call_depth >= 3
