"""不调用模型、不依赖框架的上下文治理策略。"""

from __future__ import annotations

import re
from typing import Protocol

from .models import CompressionDecision, CompressionPolicyInput, CompressionStrategy


class CompressionPolicyConfig(Protocol):
    """治理策略所需配置，具体来源由组合根注入。"""

    context_window_tokens: int
    summary_trigger_ratio: float
    forced_summary_ratio: float
    target_ratio: float
    min_summary_candidate_tokens: int
    epoch_max_model_calls: int
    repeated_compactions_to_roll: int
    frozen_context_ratio: float


_SEMANTIC_INVALIDATION = re.compile(
    r"(纠正|更正|不是.{0,12}而是|取消之前|忽略之前|改成|不要再|"
    r"新任务|换个任务|correction|instead of|ignore previous|no longer|new task)",
    re.IGNORECASE,
)

_PHASE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("research", re.compile(r"(调研|研究|搜索|检索|research|search)", re.IGNORECASE)),
    (
        "implementation",
        re.compile(r"(实现|编写代码|开发|改代码|implement|coding)", re.IGNORECASE),
    ),
    ("testing", re.compile(r"(测试|验证|排错|test|verify|debug)", re.IGNORECASE)),
    ("final", re.compile(r"(总结|最终输出|交付|final|synthesi[sz]e)", re.IGNORECASE)),
)


def has_semantic_invalidation(text: str) -> bool:
    """识别会令当前 Cache Epoch 基线失效的显式纠正。"""

    return bool(_SEMANTIC_INVALIDATION.search(text))


def infer_task_phase(text: str, current_phase: str) -> str:
    """根据显式阶段词推进任务阶段，否则保持原阶段。"""

    for phase, pattern in _PHASE_PATTERNS:
        if pattern.search(text):
            return phase
    return current_phase


class DeterministicCompressionPolicy:
    """依据硬阈值选择治理动作，本策略本身永远不调用 LLM。"""

    def __init__(self, config: CompressionPolicyConfig) -> None:
        self._config = config

    def decide(self, policy_input: CompressionPolicyInput) -> CompressionDecision:
        """按工具清理、增量摘要、Epoch Roll 的顺序生成决策。"""

        strategies: list[CompressionStrategy] = []
        reasons: list[str] = []

        if policy_input.tool_result_token_count > 0:
            strategies.append(CompressionStrategy.CLEAR_TOOL_RESULTS)
            reasons.append("存在已闭合且不属于 D1 的旧工具结果")

        should_summarize = (
            policy_input.context_usage_ratio >= self._config.summary_trigger_ratio
            and policy_input.candidate_token_count
            >= self._config.min_summary_candidate_tokens
        )
        forced_summary = (
            policy_input.context_usage_ratio >= self._config.forced_summary_ratio
            and bool(policy_input.candidate_event_ids)
        )
        requested_summary = (
            policy_input.pending_request is not None
            and policy_input.pending_request.preferred_strategy
            in {None, CompressionStrategy.INCREMENTAL_SUMMARY}
            and bool(policy_input.candidate_event_ids)
        )
        if should_summarize or forced_summary or requested_summary:
            strategies.append(CompressionStrategy.INCREMENTAL_SUMMARY)
            reasons.append("动态工作集超过摘要阈值且存在已闭合历史")

        should_roll = (
            policy_input.semantic_invalidated
            or policy_input.phase_changed
            or policy_input.prefix_mismatch
            or policy_input.cache_epoch_age >= self._config.epoch_max_model_calls
            or policy_input.repeated_compactions
            >= self._config.repeated_compactions_to_roll
            or policy_input.frozen_context_ratio >= self._config.frozen_context_ratio
            or (
                policy_input.pending_request is not None
                and policy_input.pending_request.preferred_strategy
                == CompressionStrategy.ROLL_CACHE_EPOCH
            )
        )
        if should_roll:
            if (
                policy_input.candidate_event_ids
                and CompressionStrategy.INCREMENTAL_SUMMARY not in strategies
            ):
                strategies.append(CompressionStrategy.INCREMENTAL_SUMMARY)
                reasons.append("Epoch Roll 前先沉淀尚未压缩的闭合历史")
            strategies.append(CompressionStrategy.ROLL_CACHE_EPOCH)
            reasons.append("Epoch 基线失效、阶段切换或冻结上下文债务过高")

        if not strategies:
            strategies.append(CompressionStrategy.NONE)
            reasons.append("上下文仍位于安全水位")

        return CompressionDecision(
            strategies=list(dict.fromkeys(strategies)),
            target_event_ids=policy_input.candidate_event_ids,
            preserve_event_ids=list(
                set(policy_input.suffix_event_ids)
                - set(policy_input.candidate_event_ids)
            ),
            target_tokens=int(
                self._config.context_window_tokens * self._config.target_ratio
            ),
            should_roll_epoch=should_roll,
            reason="；".join(reasons),
        )
