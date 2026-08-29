"""FastAPI、SSE 与 WebSocket 入站适配器。"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from functools import lru_cache
from uuid import uuid4

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.application.agents import RunAgent, RunAgentCommand
from app.application.runtime import ShoppingContextSnapshot
from app.composition import ApplicationContainer, build_container

from .connection import ConnectionManager
from .dto import AgentRequest
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

    @api.post("/api/agent")
    async def run_agent(request: AgentRequest) -> StreamingResponse:
        """通过 SSE 流式执行一次 Agent Run。"""

        use_case = container_provider().run_agent
        command = _to_command(request)

        async def event_stream() -> AsyncIterator[str]:
            async for event in as_sse(use_case.execute(command)):
                yield event

        return StreamingResponse(event_stream(), media_type="text/event-stream")

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
