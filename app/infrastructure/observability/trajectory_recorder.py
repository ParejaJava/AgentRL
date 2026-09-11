"""Opt-in local model-attempt capture, after routing and prompt governance.

Raw messages can contain personal data. Keep the directory private and ignored by
Git; training export must explicitly review/redact captured conversations.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import convert_to_openai_messages, messages_to_dict
from langchain_core.utils.function_calling import convert_to_openai_tool

from app.infrastructure.context import current_execution_context

_call_id: ContextVar[str | None] = ContextVar("trajectory_call_id", default=None)


def canonical_hash(value: Any) -> str:
    """Hash the canonical JSON representation for version and input matching."""

    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()
    ).hexdigest()


class ModelCallScopeMiddleware(AgentMiddleware):
    """Group all retries, fallback and emergency attempts of one logical call."""

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        token = _call_id.set(uuid4().hex)
        try:
            return await handler(request)
        finally:
            _call_id.reset(token)


class TrajectoryRecorderMiddleware(AgentMiddleware):
    """Innermost middleware: persist the actual request and its matching response.

    A separate JSON file per attempt avoids cross-process JSONL interleaving.
    A start record survives cancellation/crash; only finished successes may be
    selected for supervised training. Recording failures are explicit errors.
    """

    def __init__(self, root: Path, *, agent_role: str) -> None:
        self.root = root
        self.agent_role = agent_role

    def _write(self, path: Path, record: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        temporary.replace(path)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        context = current_execution_context.get()
        attempt_id = uuid4().hex
        path = self.root / f"{attempt_id}.json"
        messages = ([request.system_message] if request.system_message else []) + list(
            request.messages
        )
        tools = [convert_to_openai_tool(tool) for tool in request.tools]
        model = request.model
        record: dict[str, Any] = {
            "schema_version": 1,
            "source": "runtime_capture",
            "attempt_id": attempt_id,
            "step_id": _call_id.get() or attempt_id,
            "run_id": context.run_id if context else None,
            "thread_id": context.thread_id if context else None,
            "trace_id": context.trace_id if context else None,
            "parent_run_id": context.parent_run_id if context else None,
            "session_group_hash": (
                canonical_hash(
                    context.shopping.shopping_session_id
                    if context.shopping
                    else context.trace_id or context.parent_run_id or context.thread_id
                )
                if context
                else None
            ),
            "agent_role": self.agent_role,
            "created_at": datetime.now(UTC).isoformat(),
            "model": getattr(model, "model_name", None)
            or getattr(model, "model", None),
            "model_class": type(model).__name__,
            "messages": convert_to_openai_messages(messages),
            "raw_messages": messages_to_dict(messages),
            "tools": tools,
            # Explicit allowlist: do not serialize credentials, headers or URLs.
            "generation": {
                key: request.model_settings.get(key, getattr(model, key, None))
                for key in (
                    "temperature",
                    "max_tokens",
                    "max_completion_tokens",
                    "seed",
                )
                if key in request.model_settings
                or getattr(model, key, None) is not None
            },
            "tool_choice": request.tool_choice,
            "tools_hash": canonical_hash(tools),
            "prompt_hash": canonical_hash(messages_to_dict(messages)),
            "status": "started",
        }
        await asyncio.to_thread(self._write, path, record)
        started = time.perf_counter()
        try:
            response = await handler(request)
            record["status"] = "success"
            record["response"] = convert_to_openai_messages(response.result)
            record["raw_response"] = messages_to_dict(response.result)
            record["usage"] = [
                getattr(m, "usage_metadata", None) for m in response.result
            ]
            return response
        except BaseException as exc:
            record["status"] = (
                "cancelled" if isinstance(exc, asyncio.CancelledError) else "error"
            )
            record["error_type"] = type(exc).__name__
            raise
        finally:
            record["duration_seconds"] = time.perf_counter() - started
            await asyncio.to_thread(self._write, path, record)
