"""上下文治理的环境变量配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    """读取布尔环境变量，并兼容常见真假写法。"""

    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class GovernanceConfig:
    """集中保存会话级上下文治理参数。"""

    context_window_tokens: int = 131_072
    summary_trigger_ratio: float = 0.70
    forced_summary_ratio: float = 0.85
    emergency_ratio: float = 0.95
    target_ratio: float = 0.60
    hot_message_count: int = 8
    min_summary_candidate_tokens: int = 2_048
    max_compression_input_tokens: int = 16_000
    tool_offload_tokens: int = 4_000
    tool_preview_chars: int = 1_000
    epoch_max_model_calls: int = 20
    frozen_context_ratio: float = 0.50
    repeated_compactions_to_roll: int = 3
    session_root: Path = Path("data/sessions")
    cache_provider: str = "auto"
    explicit_cache: bool = False

    @classmethod
    def from_env(cls) -> GovernanceConfig:
        """从环境变量创建配置，并在启动时完成基础校验。"""

        defaults = cls()
        config = cls(
            context_window_tokens=int(
                os.getenv("LLM_CONTEXT_WINDOW", defaults.context_window_tokens)
            ),
            summary_trigger_ratio=float(
                os.getenv(
                    "CONTEXT_SUMMARY_TRIGGER_RATIO",
                    defaults.summary_trigger_ratio,
                )
            ),
            forced_summary_ratio=float(
                os.getenv(
                    "CONTEXT_FORCED_SUMMARY_RATIO",
                    defaults.forced_summary_ratio,
                )
            ),
            emergency_ratio=float(
                os.getenv("CONTEXT_EMERGENCY_RATIO", defaults.emergency_ratio)
            ),
            target_ratio=float(
                os.getenv("CONTEXT_TARGET_RATIO", defaults.target_ratio)
            ),
            hot_message_count=int(
                os.getenv("CONTEXT_HOT_MESSAGE_COUNT", defaults.hot_message_count)
            ),
            min_summary_candidate_tokens=int(
                os.getenv(
                    "CONTEXT_MIN_SUMMARY_TOKENS",
                    defaults.min_summary_candidate_tokens,
                )
            ),
            max_compression_input_tokens=int(
                os.getenv(
                    "CONTEXT_MAX_COMPRESSION_INPUT_TOKENS",
                    defaults.max_compression_input_tokens,
                )
            ),
            tool_offload_tokens=int(
                os.getenv(
                    "CONTEXT_TOOL_OFFLOAD_TOKENS",
                    defaults.tool_offload_tokens,
                )
            ),
            tool_preview_chars=int(
                os.getenv(
                    "CONTEXT_TOOL_PREVIEW_CHARS",
                    defaults.tool_preview_chars,
                )
            ),
            epoch_max_model_calls=int(
                os.getenv(
                    "CONTEXT_EPOCH_MAX_MODEL_CALLS",
                    defaults.epoch_max_model_calls,
                )
            ),
            frozen_context_ratio=float(
                os.getenv("CONTEXT_FROZEN_RATIO", defaults.frozen_context_ratio)
            ),
            repeated_compactions_to_roll=int(
                os.getenv(
                    "CONTEXT_REPEATED_COMPACTIONS_TO_ROLL",
                    defaults.repeated_compactions_to_roll,
                )
            ),
            session_root=Path(os.getenv("AGENT_SESSION_ROOT", "data/sessions")),
            cache_provider=os.getenv("LLM_CACHE_PROVIDER", "auto").lower(),
            explicit_cache=_env_bool("LLM_EXPLICIT_CACHE", False),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """拒绝会造成治理顺序错误或非法容量的配置。"""

        ratios = (
            self.target_ratio,
            self.summary_trigger_ratio,
            self.forced_summary_ratio,
            self.emergency_ratio,
        )
        if not all(0 < ratio < 1 for ratio in ratios):
            raise ValueError("上下文比例必须位于 0 和 1 之间")
        if ratios != tuple(sorted(ratios)):
            raise ValueError("上下文阈值必须按 target、summary、forced、emergency 递增")
        if self.context_window_tokens < 1:
            raise ValueError("LLM_CONTEXT_WINDOW 必须大于 0")
        if self.hot_message_count < 1:
            raise ValueError("CONTEXT_HOT_MESSAGE_COUNT 必须大于 0")
        if self.max_compression_input_tokens < self.min_summary_candidate_tokens:
            raise ValueError("压缩模型最大输入必须不小于最小摘要候选 token")
