"""Redis Stream 后台 Agent 任务消费者入口。"""

from __future__ import annotations

import asyncio
import socket

from app.composition import build_container
from app.presentation.dto import AgentRequest
from app.presentation.server import _to_command


async def run_worker() -> None:
    """持续消费 Agent 任务，并将成功结果或失败原因写回队列状态。"""

    container = build_container()
    queue = container.task_queue
    if queue is None:
        raise RuntimeError("Worker 需要 REDIS_ENABLED=true")
    consumer = f"{socket.gethostname()}-{id(asyncio.current_task())}"
    while True:
        task = await queue.consume(consumer)
        if task is None:
            continue
        try:
            request = AgentRequest.model_validate(task.payload)
            events = [
                event
                async for event in container.run_agent.execute(_to_command(request))
            ]
            await queue.acknowledge(task, events)
        except Exception as exc:  # noqa: BLE001 - Worker 边界负责重试/死信。
            await queue.fail(task, str(exc))


if __name__ == "__main__":
    asyncio.run(run_worker())
