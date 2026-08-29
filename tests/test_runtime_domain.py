"""验证 Agent Runtime 应用状态机不依赖 LangGraph。"""

from app.application.runtime import AgentId, AgentRun, RunId, RunStatus, ThreadId


def test_agent_run_lifecycle() -> None:
    """Run 只能按照应用状态机从创建进入运行和完成。"""

    run = AgentRun(
        run_id=RunId("run-1"),
        thread_id=ThreadId("thread-1"),
        agent_id=AgentId("agent-1"),
    )

    run.start()
    assert run.status is RunStatus.RUNNING

    run.complete()
    assert run.status is RunStatus.COMPLETED


def test_completed_run_cannot_fail() -> None:
    """终态 Run 不允许被后续异常反向污染。"""

    run = AgentRun(
        run_id=RunId("run-2"),
        thread_id=ThreadId("thread-1"),
        agent_id=AgentId("agent-1"),
    )
    run.start()
    run.complete()

    try:
        run.fail("late error")
    except ValueError:
        pass
    else:
        raise AssertionError("completed Run accepted a late failure")
