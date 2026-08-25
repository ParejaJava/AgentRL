"""启动 Agent Run 的应用用例。"""

from collections.abc import AsyncIterator
from typing import Any, Protocol


class AgentRuntimePort(Protocol):
    """Application 驱动 Agent Runtime 所需的最小端口。"""

    def run_agent(self, message: str) -> AsyncIterator[dict[str, Any]]: ...


class RunAgent:
    """校验请求并把执行委托给可替换的 Runtime Driver。"""

    def __init__(self, runtime: AgentRuntimePort) -> None:
        self._runtime = runtime

    async def execute(self, message: str) -> AsyncIterator[dict[str, Any]]:
        """执行一次 Agent Run 并原样转发平台事件。"""

        normalized = message.strip()
        if not normalized:
            raise ValueError("message 不能为空")
        async for event in self._runtime.run_agent(normalized):
            yield event
