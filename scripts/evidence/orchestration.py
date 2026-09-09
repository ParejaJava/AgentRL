"""Task DAG、并行加速、失败隔离和原子认领的确定性基准。"""

from __future__ import annotations

import asyncio
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

from app.application.tasking import (
    TaskBoardService,
    TaskDispatchService,
    TaskExecutionOutcome,
    TaskWorkItem,
)
from app.infrastructure.tasking import SQLiteTaskBoardRepository

from .utils import percentile, runtime_environment, utc_now


class DelayedExecutor:
    """模拟相互独立的 I/O Worker，并支持指定任务失败。"""

    def __init__(self, delay_seconds: float, failed_subject: str | None = None) -> None:
        self._delay = delay_seconds
        self._failed_subject = failed_subject

    async def execute_many(
        self,
        tasks: Sequence[TaskWorkItem],
    ) -> tuple[TaskExecutionOutcome, ...]:
        """并行执行任务并按输入顺序返回 settled 结果。"""

        async def execute(task: TaskWorkItem) -> TaskExecutionOutcome:
            await asyncio.sleep(self._delay)
            if task.subject == self._failed_subject:
                return TaskExecutionOutcome(task.task_id, False, error="injected failure")
            return TaskExecutionOutcome(task.task_id, True, result=f"done:{task.subject}")

        return tuple(await asyncio.gather(*(execute(task) for task in tasks)))


async def _run_once(database: Path, iteration: int, delay: float) -> dict[str, float | bool]:
    """运行一轮编排、并发、恢复和幂等场景并返回断言指标。"""

    repository = SQLiteTaskBoardRepository(database)
    service = TaskBoardService(repository)
    scope = f"parallel-{iteration}"
    tasks = [
        await service.create(scope, subject=f"task-{index}", description=f"canary-{index}")
        for index in range(4)
    ]

    serial_started = time.perf_counter()
    for _ in tasks:
        await asyncio.sleep(delay)
    serial = time.perf_counter() - serial_started

    dispatcher = TaskDispatchService(service, DelayedExecutor(delay))
    parallel_started = time.perf_counter()
    parallel_report = await dispatcher.dispatch(
        scope,
        [task.id for task in tasks],
        reason="parallel",
    )
    parallel = time.perf_counter() - parallel_started
    parallel_ok = all(task.status.value == "completed" for task in parallel_report.tasks)

    dag_scope = f"dag-{iteration}"
    a = await service.create(dag_scope, subject="A", description="root")
    b = await service.create(dag_scope, subject="B", description="left")
    c = await service.create(dag_scope, subject="C", description="right")
    d = await service.create(dag_scope, subject="D", description="join")
    await service.update(dag_scope, b.id, add_blocked_by=[a.id])
    await service.update(dag_scope, c.id, add_blocked_by=[a.id])
    await service.update(dag_scope, d.id, add_blocked_by=[b.id, c.id])
    initial = {item.id: item for item in await service.list(dag_scope)}
    dag_blocked = not initial[b.id].runnable and not initial[d.id].runnable
    await TaskDispatchService(service, DelayedExecutor(0)).dispatch(
        dag_scope, [a.id], reason="context_isolation"
    )
    after_a = {item.id: item for item in await service.list(dag_scope)}
    dag_unlocked = after_a[b.id].runnable and after_a[c.id].runnable

    failure_scope = f"failure-{iteration}"
    success = await service.create(failure_scope, subject="success", description="ok")
    failure = await service.create(failure_scope, subject="failure", description="bad")
    failure_report = await TaskDispatchService(
        service,
        DelayedExecutor(0, failed_subject="failure"),
    ).dispatch(failure_scope, [success.id, failure.id], reason="parallel")
    statuses = {item.subject: item.status.value for item in failure_report.tasks}
    partial_preserved = statuses == {"success": "completed", "failure": "failed"}

    claim_scope = f"claim-{iteration}"
    claimed = [
        await service.create(claim_scope, subject=f"claim-{index}", description="once")
        for index in range(2)
    ]
    competing = [
        TaskDispatchService(service, DelayedExecutor(delay)).dispatch(
            claim_scope,
            [item.id for item in claimed],
            reason="parallel",
        )
        for _ in range(2)
    ]
    settled = await asyncio.gather(*competing, return_exceptions=True)
    duplicate_prevented = sum(not isinstance(item, Exception) for item in settled) == 1

    isolation_scope = f"isolation-{iteration}"
    other_scope = f"other-{iteration}"
    await service.create(isolation_scope, subject="private", description="canary-secret")
    other = await service.list(other_scope)
    isolation_ok = not other

    # 模拟 Worker 原子认领后进程退出：租约到期后由下一次派发自动回收。
    interrupted_scope = f"interrupted-{iteration}"
    short_lease_service = TaskBoardService(repository, claim_lease_seconds=0)
    interrupted = await short_lease_service.create(
        interrupted_scope,
        subject="interrupted",
        description="recover expired worker lease",
    )
    await short_lease_service.claim_runnable(
        interrupted_scope,
        [interrupted.id],
        dispatch_id="lost-worker",
        require_parallel=False,
    )
    recovered = await service.recover_expired_claims(interrupted_scope)
    recovered_report = await TaskDispatchService(
        service,
        DelayedExecutor(0),
    ).dispatch(
        interrupted_scope,
        [interrupted.id],
        reason="context_isolation",
    )
    interruption_recovered = (
        recovered == (interrupted.id,)
        and recovered_report.tasks[0].status.value == "completed"
    )

    # 模拟网络重试重复提交相同终态：第二次写回必须是安全 no-op。
    idempotent_scope = f"idempotent-{iteration}"
    idempotent_task = await service.create(
        idempotent_scope,
        subject="idempotent",
        description="repeat the same terminal writeback",
    )
    work_items = await service.claim_runnable(
        idempotent_scope,
        [idempotent_task.id],
        dispatch_id="stable-dispatch",
        require_parallel=False,
    )
    outcomes = (
        TaskExecutionOutcome(
            task_id=idempotent_task.id,
            succeeded=True,
            result="stable-result",
        ),
    )
    first_write = await service.finish_dispatch(
        idempotent_scope,
        work_items=work_items,
        outcomes=outcomes,
    )
    second_write = await service.finish_dispatch(
        idempotent_scope,
        work_items=work_items,
        outcomes=outcomes,
    )
    idempotent_writeback = first_write[0].to_dict() == second_write[0].to_dict()
    return {
        "serial": serial,
        "parallel": parallel,
        "parallel_ok": parallel_ok,
        "dag_ok": dag_blocked and dag_unlocked,
        "partial_preserved": partial_preserved,
        "duplicate_prevented": duplicate_prevented,
        "isolation_ok": isolation_ok,
        "interruption_recovered": interruption_recovered,
        "idempotent_writeback": idempotent_writeback,
    }


async def run_orchestration_benchmark(
    root: Path,
    *,
    repetitions: int = 30,
    delay_seconds: float = 0.05,
) -> dict[str, object]:
    """执行完整确定性编排基准并计算验收指标。"""

    started_at = utc_now()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="globex-evidence-") as directory:
        database = Path(directory) / "tasking.db"
        results = [
            await _run_once(database, iteration, delay_seconds)
            for iteration in range(repetitions)
        ]
    serial = [float(item["serial"]) for item in results]
    parallel = [float(item["parallel"]) for item in results]
    speedups = [before / after for before, after in zip(serial, parallel, strict=True)]
    metrics = {
        "repetitions": repetitions,
        "serial_p50_seconds": percentile(serial, 0.5),
        "serial_p95_seconds": percentile(serial, 0.95),
        "parallel_p50_seconds": percentile(parallel, 0.5),
        "parallel_p95_seconds": percentile(parallel, 0.95),
        "speedup_p50": percentile(speedups, 0.5),
        "duplicate_dispatch_rate": round(
            1 - sum(bool(item["duplicate_prevented"]) for item in results) / repetitions,
            6,
        ),
        "context_leak_rate": round(
            1 - sum(bool(item["isolation_ok"]) for item in results) / repetitions,
            6,
        ),
        "partial_failure_preservation_rate": round(
            sum(bool(item["partial_preserved"]) for item in results) / repetitions,
            6,
        ),
        "dag_correct_rate": round(
            sum(bool(item["dag_ok"]) for item in results) / repetitions,
            6,
        ),
        "interruption_recovery_rate": round(
            sum(bool(item["interruption_recovered"]) for item in results)
            / repetitions,
            6,
        ),
        "idempotent_writeback_rate": round(
            sum(bool(item["idempotent_writeback"]) for item in results)
            / repetitions,
            6,
        ),
    }
    passed = (
        metrics["speedup_p50"] >= 1.5
        and metrics["duplicate_dispatch_rate"] == 0
        and metrics["context_leak_rate"] == 0
        and metrics["partial_failure_preservation_rate"] == 1
        and metrics["dag_correct_rate"] == 1
        and metrics["interruption_recovery_rate"] == 1
        and metrics["idempotent_writeback_rate"] == 1
    )
    return {
        "claim_id": "ORCH-001",
        "capability": "Task DAG 与隔离子 Agent 并行编排",
        "status": "verified" if passed else "code_verified",
        "started_at": started_at,
        "duration_seconds": round(time.perf_counter() - started, 6),
        "command": "uv run python -m scripts.evidence run --suite offline",
        "environment": runtime_environment(root),
        "metrics": metrics,
        "failures": [
            {"iteration": index, **item}
            for index, item in enumerate(results, start=1)
            if not all(
                bool(item[key])
                for key in (
                    "parallel_ok",
                    "dag_ok",
                    "partial_preserved",
                    "duplicate_prevented",
                    "isolation_ok",
                    "interruption_recovered",
                    "idempotent_writeback",
                )
            )
        ],
        "limitations": [
            "延迟基准使用确定性 I/O 替身，用于证明调度并行性，不代表 Kimi 网络延迟。",
            "资源互斥锁仍未进入 Task DAG 模型。",
        ],
        "passed": passed,
    }
