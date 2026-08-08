"""HTTP, upload, cancellation, and WebSocket entry points."""

from collections.abc import AsyncIterator
from uuid import uuid4

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent import MainAgent
from app.api.context import RequestContext, reset_context, set_context
from app.api.monitor import as_sse

app = FastAPI(title="Globex Agent API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = MainAgent()


class AgentRequest(BaseModel):
    message: str = Field(min_length=1)
    thread_id: str = Field(default_factory=lambda: str(uuid4()))
    session_dir: str | None = None


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/agent")
async def run_agent(request: AgentRequest) -> StreamingResponse:
    async def event_stream() -> AsyncIterator[str]:
        token = set_context(RequestContext(request.thread_id, request.session_dir))
        try:
            async for event in as_sse(agent.run_agent(request.message)):
                yield event
        finally:
            reset_context(token)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.websocket("/ws/agent")
async def agent_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_json()
            message = str(payload.get("message", ""))
            async for event in agent.run_agent(message):
                await websocket.send_json(event)
    except WebSocketDisconnect:
        return

