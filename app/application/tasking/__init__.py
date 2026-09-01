"""Agent 平台的会话级任务管理与确定性派发。"""

from .models import (
    TaskBoard,
    TaskDispatchReport,
    TaskExecutionOutcome,
    TaskRecord,
    TaskStatus,
    TaskView,
    TaskWorkItem,
)
from .ports import ParallelTaskExecutor, TaskBoardConflictError, TaskBoardRepository
from .service import (
    TaskBoardService,
    TaskDependencyError,
    TaskDispatchService,
    TaskingError,
    TaskNotFoundError,
    TaskTransitionError,
    TaskUpdateStatus,
)

__all__ = [
    "ParallelTaskExecutor",
    "TaskBoard",
    "TaskBoardConflictError",
    "TaskBoardRepository",
    "TaskBoardService",
    "TaskDependencyError",
    "TaskDispatchReport",
    "TaskDispatchService",
    "TaskExecutionOutcome",
    "TaskNotFoundError",
    "TaskRecord",
    "TaskStatus",
    "TaskTransitionError",
    "TaskUpdateStatus",
    "TaskView",
    "TaskWorkItem",
    "TaskingError",
]
