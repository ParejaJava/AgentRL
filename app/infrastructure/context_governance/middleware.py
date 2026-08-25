"""把上下文治理接入 LangChain AgentLoop 生命周期。"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.types import Command

from app.domain.context import (
    CompressionPolicyInput,
    CompressionStrategy,
    DeterministicCompressionPolicy,
    has_semantic_invalidation,
    infer_task_phase,
)

from .artifact_store import FileArtifactStore
from .breakpoint import CacheBreakpointManager
from .compressor import ContextCompressor
from .config import GovernanceConfig
from .event_store import EventRecord, SQLiteEventStore
from .messages import (
    IndexedMessage,
    canonical_json,
    collect_protected_event_ids,
    index_messages,
    message_text,
)
from .projection import PromptProjectionBuilder
from .provider_cache import create_cache_adapter
from .runtime import resolve_request_context, resolve_session_dir
from .schemas import (
    CompressionDelta,
    CompressionRequest,
    EpochBaseline,
    GovernanceSnapshot,
    SessionAgentState,
    TaskDelta,
    TaskState,
    TokenLedger,
    ToolResultEnvelope,
    WorkingMemoryItem,
)
from .strategies import (
    apply_compression_delta,
    clear_old_tool_results,
    select_summary_candidates,
)
from .token_counter import count_context, count_messages, count_text

logger = logging.getLogger(__name__)


def _model_name(model: Any) -> str:
    """从不同 ChatModel 实现中提取稳定模型名称。"""

    return str(
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or type(model).__name__
    )


def _latest_user(indexed: Sequence[IndexedMessage]) -> IndexedMessage | None:
    """返回当前消息状态中的最后一条用户消息。"""

    return next(
        (item for item in reversed(indexed) if isinstance(item.message, HumanMessage)),
        None,
    )


def _task_delta_from_states(
    existing: TaskDelta,
    old: TaskState,
    new: TaskState,
) -> TaskDelta:
    """根据完整任务状态的变化计算当前 epoch delta。"""

    return TaskDelta(
        current_step=new.current_step,
        newly_completed_steps=list(
            dict.fromkeys(
                [
                    *existing.newly_completed_steps,
                    *(
                        step
                        for step in new.completed_steps
                        if step not in old.completed_steps
                    ),
                ]
            )
        ),
        new_constraints=list(
            dict.fromkeys(
                [
                    *existing.new_constraints,
                    *(item for item in new.constraints if item not in old.constraints),
                ]
            )
        ),
        failed_tools=list(
            dict.fromkeys(
                [
                    *existing.failed_tools,
                    *(
                        item
                        for item in new.failed_tools
                        if item not in old.failed_tools
                    ),
                ]
            )
        ),
    )


def _compression_event_id(thread_id: str, event_type: str, payload: Any) -> str:
    """生成可幂等重放的治理事件 ID。"""

    digest = hashlib.sha256(
        f"{thread_id}:{event_type}:{canonical_json(payload)}".encode()
    ).hexdigest()
    return f"ctx-{digest[:24]}"


class ContextGovernanceMiddleware(AgentMiddleware):
    """在每次模型调用前治理 L2，并构造 cache-aware Prompt。"""

    state_schema = SessionAgentState

    def __init__(
        self,
        *,
        compressor: ContextCompressor | None,
        config: GovernanceConfig | None = None,
        agent_id: str = "agent",
        static_system_prompt: str = "",
        static_tools: Sequence[BaseTool | dict[str, Any]] = (),
        static_model_name: str = "model",
        static_model_settings: dict[str, Any] | None = None,
    ) -> None:
        self._config = config or GovernanceConfig.from_env()
        self._compressor = compressor
        self._agent_id = agent_id
        self._static_system_message = (
            SystemMessage(content=static_system_prompt)
            if static_system_prompt
            else None
        )
        self._static_tools = list(static_tools)
        self._static_model_name = static_model_name
        self._static_model_settings = dict(static_model_settings or {})
        self._breakpoints = CacheBreakpointManager()
        self._policy = DeterministicCompressionPolicy(self._config)
        self._projection = PromptProjectionBuilder(self._breakpoints)

    async def abefore_model(
        self,
        state: SessionAgentState,
        runtime: Any,
    ) -> dict[str, Any] | None:
        """执行确定性决策，并仅在约定场景调用压缩 LLM。"""

        messages = list(state.get("messages", []))
        if not messages:
            return None

        context = resolve_request_context(runtime)
        session_dir = resolve_session_dir(context, self._config)
        event_store = SQLiteEventStore(session_dir)
        epoch = int(state.get("cache_epoch", 0))
        await event_store.append_messages(
            thread_id=context.thread_id,
            agent_id=self._agent_id,
            messages=messages,
            cache_epoch=epoch,
        )

        indexed = index_messages(context.thread_id, messages)
        latest_user = _latest_user(indexed)
        latest_user_text = message_text(latest_user.message) if latest_user else ""

        task_state = TaskState.model_validate(state.get("task_state", {}))
        if not task_state.goal and latest_user_text:
            task_state.goal = latest_user_text
        inferred_phase = infer_task_phase(latest_user_text, task_state.task_phase)
        if task_state.task_phase == "initial":
            task_state.task_phase = inferred_phase

        task_delta = TaskDelta.model_validate(state.get("task_delta", {}))
        working_memory = dict(state.get("working_memory", {}))
        cold_refs = list(dict.fromkeys(state.get("cold_event_refs", [])))
        for item in indexed:
            if not isinstance(item.message, ToolMessage):
                continue
            try:
                envelope = ToolResultEnvelope.model_validate_json(
                    message_text(item.message)
                )
            except (ValueError, TypeError):
                continue
            if envelope.artifact_ref:
                cold_refs.append(envelope.artifact_ref)
        cold_refs = list(dict.fromkeys(cold_refs))
        compressed_ids = set(state.get("compressed_event_ids", []))
        token_ledger = TokenLedger.model_validate(state.get("token_ledger", {}))
        repeated_compactions = int(state.get("repeated_compactions", 0))
        epoch_started_call = int(state.get("epoch_started_model_call", 0))

        baseline = EpochBaseline.model_validate(
            state.get(
                "epoch_baseline",
                EpochBaseline(
                    goal=task_state.goal,
                    task_phase=task_state.task_phase,
                    constraints=task_state.constraints,
                    baseline_plan=task_state.current_plan,
                    completed_before_epoch=task_state.completed_steps,
                ).model_dump(mode="json"),
            )
        )
        baseline_message = self._breakpoints.baseline_message(baseline, epoch)
        usage = count_context(
            system_message=self._static_system_message,
            tools=self._static_tools,
            baseline_messages=[baseline_message],
            active_messages=messages,
            context_window_tokens=self._config.context_window_tokens,
        )

        protected_ids = collect_protected_event_ids(
            indexed,
            hot_message_count=self._config.hot_message_count,
        )
        candidates = select_summary_candidates(
            indexed,
            protected_event_ids=protected_ids,
            compressed_event_ids=compressed_ids,
            max_tokens=self._config.max_compression_input_tokens,
        )
        candidate_ids = [item.event_id for item in candidates]
        candidate_tokens = count_messages([item.message for item in candidates])
        old_tool_tokens = sum(
            count_messages([item.message])
            for item in indexed
            if isinstance(item.message, ToolMessage)
            and item.event_id not in protected_ids | compressed_ids
        )

        current_breakpoints = self._breakpoints.build_breakpoints(
            epoch=epoch,
            system_message=self._static_system_message,
            tools=self._static_tools,
            baseline=baseline,
            model_name=self._static_model_name,
            model_settings=self._static_model_settings,
        )
        stored_breakpoints = list(state.get("cache_breakpoints", []))
        prefix_mismatch = bool(
            stored_breakpoints
        ) and not self._breakpoints.validate_breakpoints(
            stored_breakpoints,
            current_breakpoints,
        )

        semantic_signal_id = latest_user.event_id if latest_user else None
        last_signal_id = state.get("last_semantic_signal_id")
        semantic_invalidated = bool(
            semantic_signal_id
            and semantic_signal_id != last_signal_id
            and has_semantic_invalidation(latest_user_text)
        )
        phase_changed = bool(
            token_ledger.model_calls > epoch_started_call
            and inferred_phase != baseline.task_phase
        )
        if phase_changed:
            task_state.task_phase = inferred_phase
        if semantic_invalidated and semantic_signal_id:
            # 纠正消息属于 D1，不能送入摘要候选；先作为明确边界供 Baseline 合并器读取。
            correction_id = f"correction-{semantic_signal_id}"
            working_memory[correction_id] = WorkingMemoryItem(
                id=correction_id,
                kind="boundary",
                content=latest_user_text,
                source_event_ids=[semantic_signal_id],
            ).model_dump(mode="json")
        pending_raw = state.get("pending_compression_request")
        pending_request = (
            CompressionRequest.model_validate(pending_raw) if pending_raw else None
        )
        decision = self._policy.decide(
            CompressionPolicyInput(
                suffix_event_ids=[item.event_id for item in indexed],
                candidate_event_ids=candidate_ids,
                suffix_token_count=usage.l2_tokens,
                candidate_token_count=candidate_tokens,
                tool_result_token_count=old_tool_tokens,
                task_phase=task_state.task_phase,
                context_usage_ratio=usage.usage_ratio,
                frozen_context_ratio=usage.l1_tokens
                / self._config.context_window_tokens,
                cache_epoch_age=token_ledger.model_calls - epoch_started_call,
                repeated_compactions=repeated_compactions,
                semantic_invalidated=semantic_invalidated,
                phase_changed=phase_changed,
                prefix_mismatch=prefix_mismatch,
                pending_request=pending_request,
            )
        )

        compression_succeeded = False
        compression_input_tokens = 0
        compression_output_tokens = 0
        if (
            CompressionStrategy.INCREMENTAL_SUMMARY in decision.strategies
            and candidates
            and self._compressor is not None
        ):
            try:
                delta = await self._compressor.summarize_incrementally(
                    task_state=task_state,
                    working_memory=working_memory,
                    candidates=candidates,
                    target_tokens=decision.target_tokens,
                )
                self._validate_delta(delta, candidate_ids, protected_ids)
                compression_input_tokens = candidate_tokens
                compression_output_tokens = count_text(delta.compressed_summary)
                if compression_output_tokens >= compression_input_tokens:
                    raise ValueError("压缩结果没有减少 token")

                previous_task_state = task_state
                task_state, working_memory, newly_compressed = apply_compression_delta(
                    task_state=task_state,
                    working_memory=working_memory,
                    delta=delta,
                    allowed_cold_event_ids=set(candidate_ids),
                    protected_event_ids=protected_ids,
                )
                task_delta = _task_delta_from_states(
                    task_delta,
                    previous_task_state,
                    task_state,
                )
                summary_id = _compression_event_id(
                    context.thread_id,
                    "summary-memory",
                    candidate_ids,
                )
                working_memory[summary_id] = WorkingMemoryItem(
                    id=summary_id,
                    kind="conclusion",
                    content=delta.compressed_summary,
                    source_event_ids=newly_compressed,
                ).model_dump(mode="json")
                compressed_ids.update(newly_compressed)
                repeated_compactions += 1
                compression_succeeded = True
                await self._record_governance_event(
                    event_store,
                    context.thread_id,
                    epoch,
                    "compression.incremental_summary",
                    {
                        "candidate_event_ids": candidate_ids,
                        "compressed_event_ids": newly_compressed,
                        "summary": delta.compressed_summary,
                        "delta": delta.model_dump(mode="json"),
                        "task_state": task_state.model_dump(mode="json"),
                        "working_memory": working_memory,
                    },
                )
            except Exception as exc:  # noqa: BLE001 - 压缩失败必须回退原状态。
                decision = replace(
                    decision,
                    reason=f"{decision.reason}；增量摘要失败并已回退：{exc}",
                )

        if compression_succeeded:
            token_ledger.compression_calls += 1
            token_ledger.compression_input_tokens += compression_input_tokens
            token_ledger.compression_output_tokens += compression_output_tokens

        if decision.should_roll_epoch:
            previous_epoch = epoch
            baseline, baseline_llm_used = await self._roll_epoch(
                baseline=baseline,
                task_state=task_state,
                task_delta=task_delta,
                working_memory=working_memory,
                cold_refs=cold_refs,
            )
            epoch += 1
            if baseline_llm_used:
                token_ledger.compression_calls += 1
            task_delta = TaskDelta()
            repeated_compactions = 0
            epoch_started_call = token_ledger.model_calls
            if semantic_signal_id:
                last_signal_id = semantic_signal_id
            await self._record_governance_event(
                event_store,
                context.thread_id,
                epoch,
                "cache.epoch_roll",
                {
                    "from_epoch": previous_epoch,
                    "to_epoch": epoch,
                    "reason": decision.reason,
                    "baseline": baseline.model_dump(mode="json"),
                },
            )

        remaining = [
            item.message for item in indexed if item.event_id not in compressed_ids
        ]
        baseline_message = self._breakpoints.baseline_message(baseline, epoch)
        structured_tokens = count_text(
            canonical_json(
                {
                    "task_state": task_state.model_dump(mode="json"),
                    "task_delta": task_delta.model_dump(mode="json"),
                    "working_memory": list(working_memory.values()),
                }
            )
        )
        l2_after = count_messages(remaining) + structured_tokens
        after_tokens = usage.l0_tokens + count_messages([baseline_message]) + l2_after
        snapshot = GovernanceSnapshot(
            l1_tokens=usage.l1_tokens,
            l2_tokens_before=usage.l2_tokens,
            l2_tokens_after=l2_after,
            total_tokens_before=usage.total_tokens,
            total_tokens_after=after_tokens,
            context_usage_ratio=usage.usage_ratio,
            cache_epoch=epoch,
            strategies=decision.strategies,
            reason=decision.reason,
            prefix_mismatch=prefix_mismatch,
        )

        return {
            "task_state": task_state.model_dump(mode="json"),
            "task_delta": task_delta.model_dump(mode="json"),
            "working_memory": working_memory,
            "epoch_baseline": baseline.model_dump(mode="json"),
            "cache_epoch": epoch,
            "cache_breakpoints": [
                item.model_dump(mode="json")
                for item in self._breakpoints.build_breakpoints(
                    epoch=epoch,
                    system_message=self._static_system_message,
                    tools=self._static_tools,
                    baseline=baseline,
                    model_name=self._static_model_name,
                    model_settings=self._static_model_settings,
                )
            ],
            "compressed_event_ids": sorted(compressed_ids),
            "cold_event_refs": cold_refs,
            "pending_compression_request": None,
            "token_ledger": token_ledger.model_dump(mode="json"),
            "context_version": int(state.get("context_version", 0)) + 1,
            "epoch_started_model_call": epoch_started_call,
            "repeated_compactions": repeated_compactions,
            "last_semantic_signal_id": last_signal_id,
            "last_governance": snapshot.model_dump(mode="json"),
        }

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """生成最终 Prompt Projection、添加 cache marker，并处理溢出重试。"""

        state = request.state
        context = resolve_request_context(request.runtime)
        indexed = index_messages(context.thread_id, request.messages)
        protected_ids = collect_protected_event_ids(
            indexed,
            hot_message_count=self._config.hot_message_count,
        )
        baseline = EpochBaseline.model_validate(state.get("epoch_baseline", {}))
        task_state = TaskState.model_validate(state.get("task_state", {}))
        task_delta = TaskDelta.model_validate(state.get("task_delta", {}))
        working_memory = dict(state.get("working_memory", {}))
        compressed_ids = set(state.get("compressed_event_ids", []))
        last_governance = GovernanceSnapshot.model_validate(
            state.get("last_governance", {})
        )
        governed_messages = (
            clear_old_tool_results(indexed, protected_event_ids=protected_ids)
            if CompressionStrategy.CLEAR_TOOL_RESULTS in last_governance.strategies
            else [item.message for item in indexed]
        )
        projection = self._projection.build(
            epoch=int(state.get("cache_epoch", 0)),
            baseline=baseline,
            task_state=task_state,
            task_delta=task_delta,
            working_memory=working_memory,
            indexed_messages=indexed,
            compressed_event_ids=compressed_ids,
            governed_messages=governed_messages,
        )

        model_name = _model_name(request.model)
        adapter = create_cache_adapter(
            provider=self._config.cache_provider,
            explicit_cache=self._config.explicit_cache,
            model_name=model_name,
        )
        decorated = adapter.decorate_request(
            system_message=request.system_message,
            messages=projection.messages,
            model_settings=request.model_settings,
            baseline_message_id=projection.baseline_message_id,
        )
        governed_request = request.override(
            system_message=decorated.system_message,
            messages=decorated.messages,
            model_settings=decorated.model_settings,
            state=state,
        )

        if (
            last_governance.total_tokens_after
            >= self._config.context_window_tokens * self._config.emergency_ratio
        ):
            emergency_request = self._build_emergency_request(
                request=governed_request,
                indexed=indexed,
                protected_ids=protected_ids,
                baseline=baseline,
                task_state=task_state,
                task_delta=task_delta,
                projection_working_memory=working_memory,
                adapter=adapter,
            )
            return await handler(emergency_request)

        try:
            return await handler(governed_request)
        except Exception as exc:
            if not self._is_context_overflow(exc):
                raise
            emergency_request = self._build_emergency_request(
                request=governed_request,
                indexed=indexed,
                protected_ids=protected_ids,
                baseline=baseline,
                task_state=task_state,
                task_delta=task_delta,
                projection_working_memory=working_memory,
                adapter=adapter,
            )
            return await handler(emergency_request)

    async def aafter_model(
        self,
        state: SessionAgentState,
        runtime: Any,
    ) -> dict[str, Any] | None:
        """记录模型 token/cache 指标，并把最新消息追加到 Event Log。"""

        messages = list(state.get("messages", []))
        if not messages:
            return None
        context = resolve_request_context(runtime)
        session_dir = resolve_session_dir(context, self._config)
        event_store = SQLiteEventStore(session_dir)
        epoch = int(state.get("cache_epoch", 0))
        await event_store.append_messages(
            thread_id=context.thread_id,
            agent_id=self._agent_id,
            messages=messages,
            cache_epoch=epoch,
        )

        last_ai = next(
            (
                message
                for message in reversed(messages)
                if isinstance(message, AIMessage)
            ),
            None,
        )
        if last_ai is None:
            return None
        adapter = create_cache_adapter(
            provider=self._config.cache_provider,
            explicit_cache=self._config.explicit_cache,
            model_name="",
        )
        usage = adapter.extract_metrics(last_ai)
        ledger = TokenLedger.model_validate(state.get("token_ledger", {}))
        ledger.model_calls += 1
        ledger.input_tokens += usage.input_tokens
        ledger.output_tokens += usage.output_tokens
        ledger.cached_input_tokens += usage.cached_input_tokens
        ledger.cache_write_tokens += usage.cache_write_tokens
        await self._record_governance_event(
            event_store,
            context.thread_id,
            epoch,
            "model.usage",
            {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cached_input_tokens": usage.cached_input_tokens,
                "cache_write_tokens": usage.cache_write_tokens,
                "model_call": ledger.model_calls,
            },
        )
        return {"token_ledger": ledger.model_dump(mode="json")}

    async def _roll_epoch(
        self,
        *,
        baseline: EpochBaseline,
        task_state: TaskState,
        task_delta: TaskDelta,
        working_memory: dict[str, dict[str, Any]],
        cold_refs: Sequence[str],
    ) -> tuple[EpochBaseline, bool]:
        """规则已批准 roll 后，优先调用 LLM 合并，否则确定性降级。"""

        if self._compressor is not None:
            try:
                return (
                    await self._compressor.consolidate_baseline(
                        previous=baseline,
                        task_state=task_state,
                        task_delta=task_delta,
                        working_memory=working_memory,
                        artifact_refs=cold_refs,
                    ),
                    True,
                )
            except Exception as exc:  # noqa: BLE001 - 基线合并失败允许确定性降级。
                logger.warning("LLM 基线合并失败，改用确定性合并：%s", exc)
        facts = [
            str(item.get("content", ""))
            for item in working_memory.values()
            if item.get("kind") in {"fact", "conclusion", "boundary"}
        ]
        return (
            self._breakpoints.merge_baseline(
                task_state=task_state,
                task_delta=task_delta,
                validated_facts=facts,
                artifact_refs=cold_refs,
            ),
            False,
        )

    def _build_emergency_request(
        self,
        *,
        request: ModelRequest,
        indexed: Sequence[IndexedMessage],
        protected_ids: set[str],
        baseline: EpochBaseline,
        task_state: TaskState,
        task_delta: TaskDelta,
        projection_working_memory: dict[str, dict[str, Any]],
        adapter: Any,
    ) -> ModelRequest:
        """溢出时仅保留 L1、D2、最小 D3 和 D1，然后重试一次。"""

        minimal_memory = dict(list(projection_working_memory.items())[-3:])
        emergency_compressed = {
            item.event_id for item in indexed if item.event_id not in protected_ids
        }
        projection = self._projection.build(
            epoch=int(request.state.get("cache_epoch", 0)),
            baseline=baseline,
            task_state=task_state,
            task_delta=task_delta,
            working_memory=minimal_memory,
            indexed_messages=indexed,
            compressed_event_ids=emergency_compressed,
        )
        decorated = adapter.decorate_request(
            system_message=request.system_message,
            messages=projection.messages,
            model_settings=request.model_settings,
            baseline_message_id=projection.baseline_message_id,
        )
        return request.override(
            system_message=decorated.system_message,
            messages=decorated.messages,
            model_settings=decorated.model_settings,
        )

    @staticmethod
    def _validate_delta(
        delta: CompressionDelta,
        candidate_ids: Sequence[str],
        protected_ids: set[str],
    ) -> None:
        """确保 LLM 只归档确定性规则提供的候选事件。"""

        if set(delta.cold_event_ids) != set(candidate_ids):
            raise ValueError("cold_event_ids 必须与候选事件完全一致")
        if set(delta.hot_context_keep_ids) - protected_ids:
            raise ValueError("hot_context_keep_ids 包含非 D1 事件")
        if not delta.compressed_summary.strip():
            raise ValueError("compressed_summary 不能为空")

    async def _record_governance_event(
        self,
        store: SQLiteEventStore,
        thread_id: str,
        epoch: int,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        """以幂等 ID 写入压缩、roll 或 usage 事件。"""

        await store.append(
            EventRecord(
                event_id=_compression_event_id(thread_id, event_type, payload),
                thread_id=thread_id,
                agent_id=self._agent_id,
                sequence=0,
                event_type=event_type,
                payload=payload,
                cache_epoch=epoch,
            )
        )

    @staticmethod
    def _is_context_overflow(exc: Exception) -> bool:
        """兼容供应商异常类型和文本，识别上下文窗口溢出。"""

        name = type(exc).__name__.lower()
        text = str(exc).lower()
        return "contextoverflow" in name or any(
            marker in text
            for marker in (
                "context length",
                "context window",
                "maximum context",
                "too many tokens",
            )
        )


class ToolResultMiddleware(AgentMiddleware):
    """在大型工具结果进入长期消息历史前将其写入 D4。"""

    def __init__(
        self,
        *,
        config: GovernanceConfig | None = None,
        agent_id: str = "agent",
    ) -> None:
        self._config = config or GovernanceConfig.from_env()
        self._agent_id = agent_id

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """保存大型工具原文，并用轻量 ToolResultEnvelope 替换模型可见内容。"""

        result = await handler(request)
        if not isinstance(result, ToolMessage):
            return result

        text = message_text(result)
        original_tokens = count_text(text)
        context = resolve_request_context(request.runtime)
        session_dir = resolve_session_dir(context, self._config)
        event_store = SQLiteEventStore(session_dir)
        epoch = int(request.state.get("cache_epoch", 0))

        if original_tokens < self._config.tool_offload_tokens:
            return result

        artifact_store = FileArtifactStore(session_dir)
        tool_name = request.tool.name if request.tool is not None else "tool"
        artifact_ref = await artifact_store.put_text(
            text,
            prefix=tool_name,
            suffix=".txt",
        )
        preview = text[: self._config.tool_preview_chars]
        summary = f"{tool_name} 返回了大型结果，完整内容已保存到 {artifact_ref}。"
        envelope = ToolResultEnvelope(
            status=result.status,
            summary=summary,
            preview=preview,
            artifact_ref=artifact_ref,
            original_token_count=original_tokens,
            visible_token_count=0,
        )
        visible = json.dumps(envelope.model_dump(mode="json"), ensure_ascii=False)
        envelope.visible_token_count = count_text(visible)
        visible = json.dumps(envelope.model_dump(mode="json"), ensure_ascii=False)
        governed = result.model_copy(update={"content": visible})

        payload = {
            "tool_name": tool_name,
            "tool_call_id": result.tool_call_id,
            "artifact_ref": artifact_ref,
            "original_token_count": original_tokens,
            "visible_token_count": envelope.visible_token_count,
        }
        await event_store.append(
            EventRecord(
                event_id=_compression_event_id(
                    context.thread_id,
                    "tool.offload",
                    payload,
                ),
                thread_id=context.thread_id,
                agent_id=self._agent_id,
                sequence=0,
                event_type="tool.offload",
                payload=payload,
                tool_call_id=result.tool_call_id,
                artifact_ref=artifact_ref,
                cache_epoch=epoch,
            )
        )
        return governed


class CacheMetricsMiddleware(AgentMiddleware):
    """预留独立指标扩展点；基础 usage 已由治理 Middleware 持久化。"""

    async def aafter_model(
        self,
        state: SessionAgentState,
        runtime: Any,
    ) -> dict[str, Any] | None:
        """保持 Middleware 链职责清晰，当前不重复修改状态。"""

        del state, runtime
        return None
