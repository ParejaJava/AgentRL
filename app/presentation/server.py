"""FastAPI、SSE 与 WebSocket 入站适配器。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from functools import lru_cache
from uuid import uuid4

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.application.agents import RunAgent, RunAgentCommand
from app.application.runtime import ShoppingContextSnapshot
from app.composition import ApplicationContainer, build_container

from .connection import ConnectionManager
from .dto import AgentRequest, CancelOrderRequest
from .monitor import as_sse


@lru_cache(maxsize=1)
def _default_container() -> ApplicationContainer:
    """首次业务请求时构建容器，健康检查不触发模型初始化。"""

    return build_container()


def create_app(
    container_provider: Callable[[], ApplicationContainer] = _default_container,
) -> FastAPI:
    """创建只依赖 Application 用例的 FastAPI 接口。"""

    api = FastAPI(title="Globex Agent API", version="0.1.0")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    connections = ConnectionManager()

    @api.get("/health")
    async def health() -> dict[str, str]:
        """返回不依赖外部模型服务的进程健康状态。"""

        return {"status": "ok"}

    @api.get("/health/ready")
    async def readiness() -> dict[str, object]:
        """检查已启用的跨进程依赖；模型和检索模型保持惰性初始化。"""

        container = container_provider()
        redis_ok: bool | None = None
        if container.task_queue is not None:
            try:
                redis_ok = await container.task_queue.ping()
            except Exception:  # noqa: BLE001 - 健康接口需返回状态而不是 500。
                redis_ok = False
        opensearch = None
        if container.opensearch_probe is not None:
            opensearch = await asyncio.to_thread(container.opensearch_probe.check)
        opensearch_ok = None if opensearch is None else opensearch.ready
        ready = redis_ok is not False and opensearch_ok is not False
        return {
            "status": "ready" if ready else "degraded",
            "redis": "disabled" if redis_ok is None else redis_ok,
            "langfuse": container.settings.langfuse_enabled,
            "opensearch": (
                "disabled" if opensearch_ok is None else opensearch_ok
            ),
            "details": {
                "opensearch": (
                    {"configured": False}
                    if opensearch is None
                    else {"configured": True, **opensearch.to_dict()}
                ),
                # Langfuse 是非关键遥测依赖；这里仅报告是否启用，不阻断就绪。
                "langfuse": {"configured": container.settings.langfuse_enabled},
            },
        }

    @api.post("/api/agent")
    async def run_agent(request: AgentRequest) -> StreamingResponse:
        """通过 SSE 流式执行一次 Agent Run。"""

        use_case = container_provider().run_agent
        command = _to_command(request)

        async def event_stream() -> AsyncIterator[str]:
            async for event in as_sse(use_case.execute(command)):
                yield event

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @api.post("/api/agent/jobs", status_code=202)
    async def enqueue_agent(request: AgentRequest) -> dict[str, object]:
        """把长任务提交给 Redis Stream；未启用 Redis 时明确拒绝。"""

        queue = container_provider().task_queue
        if queue is None:
            raise HTTPException(status_code=503, detail="Redis 任务队列未启用")
        task_id = str(uuid4())
        accepted = await queue.enqueue(task_id, request.model_dump(mode="json"))
        return {"task_id": task_id, "accepted": accepted, "status": "queued"}

    @api.get("/api/agent/jobs/{task_id}")
    async def agent_job_status(task_id: str) -> dict[str, object]:
        """查询异步 Agent 任务状态和最终结果。"""

        queue = container_provider().task_queue
        if queue is None:
            raise HTTPException(status_code=503, detail="Redis 任务队列未启用")
        status = await queue.status(task_id)
        if not status:
            raise HTTPException(status_code=404, detail="任务不存在")
        return {"task_id": task_id, **status}

    @api.get("/api/orders/{order_id}")
    async def query_order(order_id: str, buyer_id: str) -> dict[str, object]:
        """查询属于指定买家的订单意向；生产环境应由认证层提供 buyer_id。"""

        try:
            order = await container_provider().orders.query(order_id, buyer_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return order.to_dict()

    @api.post("/api/orders/{order_id}/cancel")
    async def cancel_order(
        order_id: str,
        request: CancelOrderRequest,
    ) -> dict[str, object]:
        """取消订单意向；该接口不处理退款，因为平台尚未接入支付。"""

        try:
            order = await container_provider().orders.cancel(
                order_id,
                request.buyer_id,
                request.reason,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return order.to_dict()

    @api.websocket("/ws/agent")
    async def agent_websocket(websocket: WebSocket) -> None:
        """在一个 WebSocket 连接上执行一个或多个同线程请求。"""

        await connections.connect(websocket)
        connection_thread_id = str(uuid4())
        try:
            while True:
                payload = await websocket.receive_json()
                request = AgentRequest.model_validate(
                    {
                        **payload,
                        "thread_id": payload.get("thread_id")
                        or connection_thread_id,
                    }
                )
                use_case: RunAgent = container_provider().run_agent
                async for event in use_case.execute(_to_command(request)):
                    await connections.send(websocket, event)
        except WebSocketDisconnect:
            connections.disconnect(websocket)

    @api.websocket("/ws/events/{shopping_session_id}")
    async def session_events(
        websocket: WebSocket,
        shopping_session_id: str,
    ) -> None:
        """订阅指定购物会话的过程事件，不在该连接中启动 Agent。"""

        await websocket.accept()
        event_bus = container_provider().event_bus
        queue = event_bus.subscribe(shopping_session_id)
        try:
            while True:
                event = await queue.get()
                try:
                    await websocket.send_json(event.to_dict())
                finally:
                    queue.task_done()
        except WebSocketDisconnect:
            pass
        finally:
            event_bus.unsubscribe(shopping_session_id, queue)

    return api


def _to_command(request: AgentRequest) -> RunAgentCommand:
    """把不可信接口 DTO 映射为应用层不可变命令。"""

    shopping = ShoppingContextSnapshot(
        shopping_session_id=request.shopping_session_id,
        buyer_id=request.buyer_id,
        locale=request.locale,
        currency=request.currency,
    )
    return RunAgentCommand(
        message=request.message,
        shopping=shopping,
        thread_id=request.thread_id,
        session_dir=request.session_dir,
    )


app = create_app()
