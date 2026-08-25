"""FastAPI、SSE 与 WebSocket 入站适配器。"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from functools import lru_cache
from uuid import uuid4

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.application.agents import RunAgent
from app.composition import ApplicationContainer, build_container
from app.infrastructure.context import RequestContext, reset_context, set_context

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

        async def event_stream() -> AsyncIterator[str]:
            token = set_context(RequestContext(request.thread_id, request.session_dir))
            try:
                async for event in as_sse(use_case.execute(request.message)):
                    yield event
            finally:
                reset_context(token)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @api.websocket("/ws/agent")
    async def agent_websocket(websocket: WebSocket) -> None:
        """在一个 WebSocket 连接上执行一个或多个同线程请求。"""

        await connections.connect(websocket)
        connection_thread_id = str(uuid4())
        try:
            while True:
                payload = await websocket.receive_json()
                message = str(payload.get("message", ""))
                thread_id = str(payload.get("thread_id") or connection_thread_id)
                session_dir = payload.get("session_dir")
                token = set_context(RequestContext(thread_id, session_dir))
                try:
                    use_case: RunAgent = container_provider().run_agent
                    async for event in use_case.execute(message):
                        await connections.send(websocket, event)
                finally:
                    reset_context(token)
        except WebSocketDisconnect:
            connections.disconnect(websocket)

    return api


app = create_app()
