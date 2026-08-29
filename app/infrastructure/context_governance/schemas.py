"""上下文治理使用的状态、策略和事件 Schema。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal, NotRequired

from langchain.agents.middleware import AgentState
from pydantic import BaseModel, Field

from app.application.context_governance import CompressionStrategy


def utc_now() -> datetime:
    """返回带时区的 UTC 时间。"""

    return datetime.now(UTC)


class TaskState(BaseModel):
    """保存当前会话的完整结构化任务状态。"""

    goal: str = ""
    task_phase: str = "initial"
    constraints: list[str] = Field(default_factory=list)
    current_plan: list[str] = Field(default_factory=list)
    completed_steps: list[str] = Field(default_factory=list)
    current_step: str | None = None
    failed_tools: list[str] = Field(default_factory=list)
    pending_questions: list[str] = Field(default_factory=list)


class TaskDelta(BaseModel):
    """保存当前 cache epoch 相对基线产生的任务增量。"""

    current_step: str | None = None
    newly_completed_steps: list[str] = Field(default_factory=list)
    new_constraints: list[str] = Field(default_factory=list)
    failed_tools: list[str] = Field(default_factory=list)


class WorkingMemoryItem(BaseModel):
    """表示一条可增量更新的会话内工作记忆。"""

    id: str
    kind: Literal["fact", "preference", "conclusion", "failure", "boundary"]
    content: str
    source_event_ids: list[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=utc_now)


class EpochBaseline(BaseModel):
    """表示一个 cache epoch 内不可变的稳定任务快照。"""

    goal: str = ""
    task_phase: str = "initial"
    constraints: list[str] = Field(default_factory=list)
    baseline_plan: list[str] = Field(default_factory=list)
    completed_before_epoch: list[str] = Field(default_factory=list)
    validated_facts: list[str] = Field(default_factory=list)
    stable_artifact_refs: list[str] = Field(default_factory=list)


class CacheBreakpoint(BaseModel):
    """描述缓存前缀中一个可验证的稳定边界。"""

    epoch: int
    layer: Literal["L0", "L1"]
    message_id: str | None = None
    semantic_hash: str
    provider_projection_hash: str
    prefix_token_count: int
    created_at: datetime = Field(default_factory=utc_now)


class CompressionRequest(BaseModel):
    """记录 Agent 主动提交但尚未执行的压缩请求。"""

    reason: str
    preferred_strategy: CompressionStrategy | None = None
    requested_at: datetime = Field(default_factory=utc_now)


class CompressionDelta(BaseModel):
    """压缩模型返回的结构化增量，应用前必须再次校验。"""

    task_state_patch: dict[str, Any] = Field(default_factory=dict)
    working_memory_upserts: list[WorkingMemoryItem] = Field(default_factory=list)
    working_memory_delete_ids: list[str] = Field(default_factory=list)
    hot_context_keep_ids: list[str] = Field(default_factory=list)
    cold_event_ids: list[str] = Field(default_factory=list)
    compressed_summary: str


class ToolResultEnvelope(BaseModel):
    """表示经过工具侧治理后允许进入 Prompt 的返回值。"""

    status: str
    summary: str
    preview: str | None = None
    artifact_ref: str | None = None
    original_token_count: int
    visible_token_count: int


class TokenLedger(BaseModel):
    """累计记录模型和上下文治理的 token 使用情况。"""

    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    compression_calls: int = 0
    compression_input_tokens: int = 0
    compression_output_tokens: int = 0


class GovernanceSnapshot(BaseModel):
    """保存最近一次模型调用前的治理指标和决策。"""

    l0_tokens: int = 0
    l1_tokens: int = 0
    l2_tokens_before: int = 0
    l2_tokens_after: int = 0
    total_tokens_before: int = 0
    total_tokens_after: int = 0
    context_usage_ratio: float = 0.0
    cache_epoch: int = 0
    strategies: list[CompressionStrategy] = Field(default_factory=list)
    reason: str = ""
    prefix_mismatch: bool = False


class SessionAgentState(AgentState):
    """LangGraph Checkpointer 持久化的单个 AgentLoop 会话状态。"""

    task_state: NotRequired[dict[str, Any]]
    task_delta: NotRequired[dict[str, Any]]
    working_memory: NotRequired[dict[str, dict[str, Any]]]
    epoch_baseline: NotRequired[dict[str, Any]]
    cache_epoch: NotRequired[int]
    cache_breakpoints: NotRequired[list[dict[str, Any]]]
    compressed_event_ids: NotRequired[list[str]]
    cold_event_refs: NotRequired[list[str]]
    pending_compression_request: NotRequired[dict[str, Any] | None]
    token_ledger: NotRequired[dict[str, Any]]
    context_version: NotRequired[int]
    epoch_started_model_call: NotRequired[int]
    repeated_compactions: NotRequired[int]
    last_semantic_signal_id: NotRequired[str | None]
    last_governance: NotRequired[dict[str, Any]]


class ChildAgentResult(BaseModel):
    """子 AgentLoop 向主 Agent 返回的隔离结果。"""

    task_id: str
    status: Literal["completed", "failed"]
    summary: str
    evidence_refs: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
