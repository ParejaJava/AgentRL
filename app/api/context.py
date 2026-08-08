"""Request-scoped thread and session context."""

from contextvars import ContextVar, Token
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RequestContext:
    thread_id: str
    session_dir: str | None = None


current_context: ContextVar[RequestContext | None] = ContextVar(
    "current_context", default=None
)


def set_context(context: RequestContext) -> Token[RequestContext | None]:
    return current_context.set(context)


def reset_context(token: Token[RequestContext | None]) -> None:
    current_context.reset(token)

