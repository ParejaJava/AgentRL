"""请求范围的 Agent 执行上下文适配器。"""

from contextvars import ContextVar, Token
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RequestContext:
    """连接入口传给 Agent Runtime 的线程和存储范围。"""

    thread_id: str
    session_dir: str | None = None


current_context: ContextVar[RequestContext | None] = ContextVar(
    "current_context",
    default=None,
)


def set_context(context: RequestContext) -> Token[RequestContext | None]:
    """为当前异步调用链绑定请求上下文。"""

    return current_context.set(context)


def reset_context(token: Token[RequestContext | None]) -> None:
    """在请求结束后恢复进入前的上下文。"""

    current_context.reset(token)
