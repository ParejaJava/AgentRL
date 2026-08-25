"""WebSocket 连接生命周期管理。"""

from fastapi import WebSocket


class ConnectionManager:
    """跟踪活动连接并提供统一发送入口。"""

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        """接受并登记一个 WebSocket。"""

        await websocket.accept()
        self._connections.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        """移除已经关闭的 WebSocket。"""

        self._connections.discard(websocket)

    async def send(self, websocket: WebSocket, event: dict[str, object]) -> None:
        """向指定连接发送标准 Agent 事件。"""

        await websocket.send_json(event)
