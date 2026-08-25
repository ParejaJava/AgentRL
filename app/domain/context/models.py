"""不依赖 Agent 框架的上下文治理值对象。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


class CompressionStrategy(str, Enum):
    """上下文治理内核支持的动作。"""

    NONE = "none"
    CLEAR_TOOL_RESULTS = "clear_tool_results"
    OFFLOAD_ARTIFACTS = "offload_artifacts"
    INCREMENTAL_SUMMARY = "incremental_summary"
    SEMANTIC_PRUNE = "semantic_prune"
    ROLL_CACHE_EPOCH = "roll_cache_epoch"


class CompressionDirective(Protocol):
    """描述外部提交给治理内核的显式压缩意图。"""

    preferred_strategy: CompressionStrategy | None


@dataclass(frozen=True, slots=True)
class CompressionPolicyInput:
    """确定性治理策略一次决策所需的全部事实。"""

    suffix_event_ids: list[str] = field(default_factory=list)
    candidate_event_ids: list[str] = field(default_factory=list)
    suffix_token_count: int = 0
    candidate_token_count: int = 0
    tool_result_token_count: int = 0
    open_tool_call_ids: list[str] = field(default_factory=list)
    task_phase: str = "initial"
    context_usage_ratio: float = 0.0
    frozen_context_ratio: float = 0.0
    cache_epoch_age: int = 0
    repeated_compactions: int = 0
    semantic_invalidated: bool = False
    phase_changed: bool = False
    prefix_mismatch: bool = False
    pending_request: CompressionDirective | None = None


@dataclass(frozen=True, slots=True)
class CompressionDecision:
    """治理内核输出的不可变策略决策。"""

    strategies: list[CompressionStrategy] = field(default_factory=list)
    target_event_ids: list[str] = field(default_factory=list)
    preserve_event_ids: list[str] = field(default_factory=list)
    target_tokens: int = 0
    should_roll_epoch: bool = False
    reason: str = ""
