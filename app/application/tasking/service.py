"""任务 CRUD、依赖 DAG 和确定性派发应用服务。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Literal, TypeVar
from uuid import uuid4

from .models import (
    TaskBoard,
    TaskDispatchReport,
    TaskExecutionOutcome,
    TaskRecord,
    TaskStatus,
    TaskView,
    TaskWorkItem,
    utc_now_iso,
)
from .ports import ParallelTaskExecutor, TaskBoardConflictError, TaskBoardRepository

TaskUpdateStatus = Literal[
    "pending",
    "in_progress",
    "completed",
    "failed",
    "deleted",
]


class TaskingError(ValueError):
    """可安全反馈给模型的任务管理错误。"""


class TaskNotFoundError(TaskingError):
    """指定任务不存在。"""


class TaskDependencyError(TaskingError):
    """任务依赖不存在、自指或构成循环。"""


class TaskTransitionError(TaskingError):
    """任务状态迁移不符合生命周期约束。"""


T = TypeVar("T")


class TaskBoardService:
    """维护 TaskBoard，并通过乐观重试保证并发写入不丢失。"""

    _ALLOWED_TRANSITIONS: ClassVar[dict[TaskStatus, set[TaskStatus]]] = {
        TaskStatus.PENDING: {TaskStatus.IN_PROGRESS, TaskStatus.FAILED},
        TaskStatus.IN_PROGRESS: {
            TaskStatus.PENDING,
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
        },
        TaskStatus.FAILED: {TaskStatus.PENDING, TaskStatus.IN_PROGRESS},
        TaskStatus.COMPLETED: set(),
    }

    def __init__(
        self,
        repository: TaskBoardRepository,
        *,
        max_conflict_retries: int = 8,
        claim_lease_seconds: float = 300.0,
    ) -> None:
        if max_conflict_retries < 1:
            raise ValueError("max_conflict_retries 必须大于 0")
        self._repository = repository
        self._max_conflict_retries = max_conflict_retries
        self._claim_lease_seconds = max(0.0, claim_lease_seconds)

    async def create(
        self,
        scope_id: str,
        *,
        subject: str,
        description: str,
        metadata: dict[str, Any] | None = None,
    ) -> TaskView:
        """创建初始状态为 pending 的任务，并原子分配顺序 ID。"""

        normalized_subject = subject.strip()
        normalized_description = description.strip()
        if not normalized_subject:
            raise TaskingError("subject 不能为空")
        if not normalized_description:
            raise TaskingError("description 不能为空")

        def mutation(board: TaskBoard) -> str:
            task_id = str(board.next_numeric_id)
            board.next_numeric_id += 1
            board.tasks[task_id] = TaskRecord(
                id=task_id,
                subject=normalized_subject,
                description=normalized_description,
                metadata=deepcopy(metadata or {}),
            )
            return task_id

        board, task_id = await self._mutate(scope_id, mutation)
        return self._view(board, task_id)

    async def get(self, scope_id: str, task_id: str) -> TaskView:
        """获取一个任务的完整视图。"""

        board = await self._repository.load(self._normalize_scope(scope_id))
        self._require_task(board, task_id)
        return self._view(board, task_id)

    async def list(self, scope_id: str) -> tuple[TaskView, ...]:
        """按数字 ID 顺序返回当前会话全部任务。"""

        board = await self._repository.load(self._normalize_scope(scope_id))
        return tuple(
            self._view(board, task_id)
            for task_id in sorted(board.tasks, key=self._task_sort_key)
        )

    async def update(
        self,
        scope_id: str,
        task_id: str,
        *,
        subject: str | None = None,
        description: str | None = None,
        add_blocks: Sequence[str] | None = None,
        status: TaskUpdateStatus | None = None,
        add_blocked_by: Sequence[str] | None = None,
        owner: str | None = None,
        metadata: dict[str, Any] | None = None,
        result: str | None = None,
    ) -> TaskView | None:
        """更新任务、添加依赖，或使用 deleted 永久删除任务。"""

        normalized_id = task_id.strip()
        if not normalized_id:
            raise TaskingError("task_id 不能为空")

        def mutation(board: TaskBoard) -> str | None:
            task = self._require_task(board, normalized_id)
            if status == "deleted":
                del board.tasks[normalized_id]
                for other in board.tasks.values():
                    other.blocked_by = [
                        blocker
                        for blocker in other.blocked_by
                        if blocker != normalized_id
                    ]
                    other.updated_at = utc_now_iso()
                return None

            changed = False
            if subject is not None:
                normalized_subject = subject.strip()
                if not normalized_subject:
                    raise TaskingError("subject 不能为空")
                task.subject = normalized_subject
                changed = True
            if description is not None:
                task.description = description.strip()
                changed = True
            if owner is not None:
                task.owner = owner.strip() or None
                changed = True
            if result is not None:
                task.result = result
                task.error = None
                changed = True
            if metadata is not None:
                for key, value in metadata.items():
                    if value is None:
                        task.metadata.pop(key, None)
                    else:
                        task.metadata[key] = deepcopy(value)
                changed = True

            for blocker_id in self._normalized_ids(add_blocked_by):
                self._add_dependency(board, blocker_id, normalized_id)
                changed = True
            for blocked_id in self._normalized_ids(add_blocks):
                self._add_dependency(board, normalized_id, blocked_id)
                changed = True

            if status is not None:
                new_status = TaskStatus(status)
                self._transition(board, task, new_status)
                changed = changed or new_status != task.status
                task.status = new_status
                if new_status != TaskStatus.FAILED:
                    task.error = None
                if new_status != TaskStatus.IN_PROGRESS:
                    task.owner = None
                    task.lease_expires_at = None

            if not changed:
                raise TaskingError("没有提供可更新的字段")
            task.updated_at = utc_now_iso()
            return normalized_id

        board, updated_id = await self._mutate(scope_id, mutation)
        return self._view(board, updated_id) if updated_id is not None else None

    async def claim_runnable(
        self,
        scope_id: str,
        task_ids: Sequence[str],
        *,
        dispatch_id: str,
        require_parallel: bool,
    ) -> tuple[TaskWorkItem, ...]:
        """在同一事务中校验 DAG 并认领一组可运行任务。"""

        normalized_ids = self._normalized_ids(task_ids)
        if not normalized_ids:
            raise TaskingError("派发至少需要一个 task_id")
        if require_parallel and len(normalized_ids) < 2:
            raise TaskingError("parallel 派发至少需要两个独立任务")

        def mutation(board: TaskBoard) -> tuple[tuple[str, str], ...]:
            selected = [self._require_task(board, task_id) for task_id in normalized_ids]
            selected_ids = {task.id for task in selected}
            for task in selected:
                view = self._view(board, task.id)
                if task.status != TaskStatus.PENDING:
                    raise TaskTransitionError(
                        f"任务 {task.id} 当前状态为 {task.status.value}，不能派发"
                    )
                if task.owner:
                    raise TaskTransitionError(
                        f"任务 {task.id} 已由 {task.owner} 认领"
                    )
                if view.active_blocked_by:
                    blockers = ", ".join(view.active_blocked_by)
                    raise TaskDependencyError(
                        f"任务 {task.id} 仍被任务 {blockers} 阻塞"
                    )
                if selected_ids.intersection(task.blocked_by):
                    raise TaskDependencyError("待并行任务之间存在依赖关系")

            claims: list[tuple[str, str]] = []
            for task in selected:
                owner = f"fork:{dispatch_id}:{task.id}"
                task.status = TaskStatus.IN_PROGRESS
                task.owner = owner
                task.lease_expires_at = (
                    datetime.now(UTC)
                    + timedelta(seconds=self._claim_lease_seconds)
                ).isoformat()
                task.error = None
                task.updated_at = utc_now_iso()
                claims.append((task.id, owner))
            return tuple(claims)

        board, claims = await self._mutate(scope_id, mutation)
        return tuple(
            TaskWorkItem(
                task_id=task_id,
                subject=board.tasks[task_id].subject,
                description=board.tasks[task_id].description,
                metadata=deepcopy(board.tasks[task_id].metadata),
                owner=owner,
            )
            for task_id, owner in claims
        )

    async def recover_expired_claims(self, scope_id: str) -> tuple[str, ...]:
        """把租约已过期的中断任务恢复为 pending，允许安全重新派发。"""

        now = datetime.now(UTC)

        def mutation(board: TaskBoard) -> tuple[str, ...]:
            recovered: list[str] = []
            for task in board.tasks.values():
                if task.status != TaskStatus.IN_PROGRESS or not task.lease_expires_at:
                    continue
                if not self._lease_is_expired(task.lease_expires_at, now):
                    continue
                task.status = TaskStatus.PENDING
                task.owner = None
                task.lease_expires_at = None
                task.error = "上一次 Worker 租约过期，任务已恢复待派发"
                task.updated_at = utc_now_iso()
                recovered.append(task.id)
            return tuple(recovered)

        board = await self._repository.load(self._normalize_scope(scope_id))
        has_expired = any(
            task.status == TaskStatus.IN_PROGRESS
            and task.lease_expires_at
            and self._lease_is_expired(task.lease_expires_at, now)
            for task in board.tasks.values()
        )
        if not has_expired:
            return ()
        _, recovered = await self._mutate(scope_id, mutation)
        return recovered

    async def finish_dispatch(
        self,
        scope_id: str,
        *,
        work_items: Sequence[TaskWorkItem],
        outcomes: Sequence[TaskExecutionOutcome],
    ) -> tuple[TaskView, ...]:
        """原子回写同一批子 Agent 任务的成功和失败结果。"""

        outcome_by_id = {outcome.task_id: outcome for outcome in outcomes}
        if set(outcome_by_id) != {item.task_id for item in work_items}:
            raise TaskingError("子 Agent 结果与已认领任务不匹配")

        def mutation(board: TaskBoard) -> tuple[str, ...]:
            for item in work_items:
                task = self._require_task(board, item.task_id)
                outcome = outcome_by_id[task.id]
                if task.owner == item.owner and self._same_terminal_outcome(
                    task,
                    outcome,
                ):
                    # 网络重试可能重复提交完全相同的结果；保持已有事实不变。
                    continue
                if task.status != TaskStatus.IN_PROGRESS or task.owner != item.owner:
                    raise TaskTransitionError(
                        f"任务 {task.id} 的认领状态已变化，拒绝覆盖结果"
                    )
                if outcome.succeeded:
                    task.status = TaskStatus.COMPLETED
                    task.result = outcome.result or ""
                    task.error = None
                else:
                    task.status = TaskStatus.FAILED
                    task.error = outcome.error or "子 Agent 执行失败"
                task.lease_expires_at = None
                task.updated_at = utc_now_iso()
            return tuple(item.task_id for item in work_items)

        board, finished_ids = await self._mutate(scope_id, mutation)
        return tuple(self._view(board, task_id) for task_id in finished_ids)

    @staticmethod
    def _lease_is_expired(value: str, now: datetime) -> bool:
        """解析租约时间；损坏的租约按过期处理，避免任务永久卡死。"""

        try:
            return datetime.fromisoformat(value) <= now
        except (TypeError, ValueError):
            return True

    @staticmethod
    def _same_terminal_outcome(
        task: TaskRecord,
        outcome: TaskExecutionOutcome,
    ) -> bool:
        """判断重复回写是否与已经提交的终态完全一致。"""

        if outcome.succeeded:
            return (
                task.status == TaskStatus.COMPLETED
                and task.result == (outcome.result or "")
            )
        return (
            task.status == TaskStatus.FAILED
            and task.error == (outcome.error or "子 Agent 执行失败")
        )

    async def _mutate(
        self,
        scope_id: str,
        mutation: Callable[[TaskBoard], T],
    ) -> tuple[TaskBoard, T]:
        """在版本冲突时重新读取并重放纯内存修改。"""

        normalized_scope = self._normalize_scope(scope_id)
        last_conflict: TaskBoardConflictError | None = None
        for _ in range(self._max_conflict_retries):
            board = await self._repository.load(normalized_scope)
            expected_version = board.version
            result = mutation(board)
            try:
                saved = await self._repository.save(
                    board,
                    expected_version=expected_version,
                )
            except TaskBoardConflictError as exc:
                last_conflict = exc
                continue
            return saved, result
        raise TaskBoardConflictError("任务看板并发更新重试次数已耗尽") from last_conflict

    def _view(self, board: TaskBoard, task_id: str) -> TaskView:
        task = self._require_task(board, task_id)
        active_blocked_by = tuple(
            blocker_id
            for blocker_id in task.blocked_by
            if board.tasks[blocker_id].status != TaskStatus.COMPLETED
        )
        blocks = tuple(
            other.id
            for other in board.tasks.values()
            if task_id in other.blocked_by
        )
        runnable = (
            task.status == TaskStatus.PENDING
            and not task.owner
            and not active_blocked_by
        )
        effective_status: str = (
            "blocked"
            if task.status == TaskStatus.PENDING and active_blocked_by
            else task.status.value
        )
        return TaskView(
            id=task.id,
            subject=task.subject,
            description=task.description,
            status=task.status,
            effective_status=effective_status,  # type: ignore[arg-type]
            owner=task.owner,
            lease_expires_at=task.lease_expires_at,
            blocked_by=tuple(task.blocked_by),
            active_blocked_by=active_blocked_by,
            blocks=tuple(sorted(blocks, key=self._task_sort_key)),
            runnable=runnable,
            result=task.result,
            error=task.error,
            metadata=deepcopy(task.metadata),
            created_at=task.created_at,
            updated_at=task.updated_at,
        )

    def _transition(
        self,
        board: TaskBoard,
        task: TaskRecord,
        new_status: TaskStatus,
    ) -> None:
        if new_status == task.status:
            return
        if new_status not in self._ALLOWED_TRANSITIONS[task.status]:
            raise TaskTransitionError(
                f"不允许从 {task.status.value} 迁移到 {new_status.value}"
            )
        if new_status == TaskStatus.IN_PROGRESS:
            active = self._view(board, task.id).active_blocked_by
            if active:
                raise TaskDependencyError(
                    f"任务 {task.id} 仍被任务 {', '.join(active)} 阻塞"
                )

    def _add_dependency(
        self,
        board: TaskBoard,
        blocker_id: str,
        blocked_id: str,
    ) -> None:
        blocker = self._require_task(board, blocker_id)
        blocked = self._require_task(board, blocked_id)
        if blocker.id == blocked.id:
            raise TaskDependencyError("任务不能依赖自身")
        if blocker.id in blocked.blocked_by:
            return
        blocked.blocked_by.append(blocker.id)
        blocked.blocked_by.sort(key=self._task_sort_key)
        if self._has_cycle(board):
            blocked.blocked_by.remove(blocker.id)
            raise TaskDependencyError("新增依赖会形成循环 DAG")
        blocked.updated_at = utc_now_iso()

    @staticmethod
    def _has_cycle(board: TaskBoard) -> bool:
        """使用三色 DFS 检测 blocker -> dependent 图中的环。"""

        dependents: dict[str, list[str]] = {task_id: [] for task_id in board.tasks}
        for task in board.tasks.values():
            for blocker_id in task.blocked_by:
                dependents[blocker_id].append(task.id)
        colors: dict[str, int] = {task_id: 0 for task_id in board.tasks}

        def visit(task_id: str) -> bool:
            colors[task_id] = 1
            for dependent_id in dependents[task_id]:
                if colors[dependent_id] == 1:
                    return True
                if colors[dependent_id] == 0 and visit(dependent_id):
                    return True
            colors[task_id] = 2
            return False

        return any(
            colors[task_id] == 0 and visit(task_id) for task_id in board.tasks
        )

    @staticmethod
    def _require_task(board: TaskBoard, task_id: str) -> TaskRecord:
        try:
            return board.tasks[task_id]
        except KeyError as exc:
            raise TaskNotFoundError(f"任务 {task_id} 不存在") from exc

    @staticmethod
    def _normalized_ids(task_ids: Sequence[str] | None) -> tuple[str, ...]:
        if not task_ids:
            return ()
        return tuple(dict.fromkeys(task_id.strip() for task_id in task_ids if task_id.strip()))

    @staticmethod
    def _normalize_scope(scope_id: str) -> str:
        normalized = scope_id.strip()
        if not normalized:
            raise TaskingError("任务会话 scope_id 不能为空")
        return normalized

    @staticmethod
    def _task_sort_key(task_id: str) -> tuple[int, int | str]:
        return (0, int(task_id)) if task_id.isdigit() else (1, task_id)


class TaskDispatchService:
    """把可运行任务原子认领后并行交给隔离子 Agent。"""

    def __init__(
        self,
        tasks: TaskBoardService,
        executor: ParallelTaskExecutor,
    ) -> None:
        self._tasks = tasks
        self._executor = executor

    async def dispatch(
        self,
        scope_id: str,
        task_ids: Sequence[str],
        *,
        reason: str,
    ) -> TaskDispatchReport:
        """执行一次经过 Task DAG 校验的确定性派发。"""

        # 先回收中断 Worker 的过期租约，再认领当前可运行任务。
        await self._tasks.recover_expired_claims(scope_id)
        dispatch_id = uuid4().hex[:12]
        work_items = await self._tasks.claim_runnable(
            scope_id,
            task_ids,
            dispatch_id=dispatch_id,
            require_parallel=reason == "parallel",
        )
        try:
            outcomes = await self._executor.execute_many(work_items)
        except Exception as exc:  # noqa: BLE001 - 批次级错误必须回写所有任务。
            outcomes = tuple(
                TaskExecutionOutcome(
                    task_id=item.task_id,
                    succeeded=False,
                    error=str(exc),
                )
                for item in work_items
            )
        finished = await self._tasks.finish_dispatch(
            scope_id,
            work_items=work_items,
            outcomes=outcomes,
        )
        return TaskDispatchReport(
            dispatch_id=dispatch_id,
            reason=reason,
            tasks=finished,
        )
