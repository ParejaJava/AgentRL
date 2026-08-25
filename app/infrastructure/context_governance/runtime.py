"""解析当前 AgentLoop 的线程标识和会话目录。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from langgraph.config import get_config

from app.infrastructure.context import RequestContext, current_context

from .config import GovernanceConfig

_SAFE_THREAD_ID = re.compile(r"[^A-Za-z0-9_.-]+")


def sanitize_thread_id(thread_id: str) -> str:
    """把 thread_id 转换为安全的目录名称。"""

    sanitized = _SAFE_THREAD_ID.sub("_", thread_id).strip("._")
    return sanitized or "default"


def resolve_request_context(runtime: Any | None = None) -> RequestContext:
    """优先从 LangGraph Runtime 获取上下文，并提供兼容回退。"""

    runtime_context = getattr(runtime, "context", None)
    if isinstance(runtime_context, RequestContext):
        return runtime_context

    scoped_context = current_context.get()
    if scoped_context is not None:
        return scoped_context

    try:
        config = get_config()
        thread_id = str(config.get("configurable", {}).get("thread_id", "default"))
    except RuntimeError:
        thread_id = "default"
    return RequestContext(thread_id=thread_id)


def resolve_session_dir(
    context: RequestContext,
    config: GovernanceConfig,
) -> Path:
    """在配置的会话根目录内解析并校验当前线程目录。"""

    root = config.session_root.resolve()
    if context.session_dir:
        requested = Path(context.session_dir)
        candidate = (
            requested.resolve()
            if requested.is_absolute()
            else (root / requested).resolve()
        )
    else:
        candidate = (root / sanitize_thread_id(context.thread_id)).resolve()

    if candidate != root and root not in candidate.parents:
        raise ValueError("session_dir 必须位于 AGENT_SESSION_ROOT 内")
    return candidate
