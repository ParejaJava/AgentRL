"""HTTP 和 WebSocket 边界使用的 DTO。"""

from uuid import uuid4

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    """启动一次 Agent Run 的接口请求。"""

    message: str = Field(min_length=1)
    thread_id: str = Field(default_factory=lambda: str(uuid4()))
    session_dir: str | None = None
