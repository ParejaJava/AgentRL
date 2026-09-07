"""按购物会话隔离的 Agent Silent-Drift 确定性检测。"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

_WORD_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]{2,}")


def _keywords(text: str) -> set[str]:
    """抽取英文整词与中文二元组，供低成本目标一致性判断。"""

    values: set[str] = set()
    for token in _WORD_RE.findall(text or ""):
        if token.isascii():
            values.add(token.lower())
        else:
            values.update(token[index : index + 2] for index in range(len(token) - 1))
    return values


@dataclass(frozen=True, slots=True)
class DriftReport:
    """一次检测产生的可观测漂移信号。"""

    reasons: tuple[str, ...] = ()

    @property
    def drifted(self) -> bool:
        return bool(self.reasons)


@dataclass(slots=True)
class _SessionTrace:
    original_query: str = ""
    query_keywords: set[str] = field(default_factory=set)
    rounds: int = 0
    recent_actions: list[str] = field(default_factory=list)
    consecutive_empty: int = 0
    token_history: list[int] = field(default_factory=list)


class DriftDetector:
    """检测目标遗忘、连续空召回和相对 Token 成本突增。"""

    def __init__(
        self,
        *,
        check_interval: int = 3,
        keyword_hit_floor: float = 0.2,
        empty_result_limit: int = 3,
        cost_spike_multiplier: float = 2.0,
    ) -> None:
        self._check_interval = max(1, check_interval)
        self._keyword_hit_floor = keyword_hit_floor
        self._empty_result_limit = max(1, empty_result_limit)
        self._cost_spike_multiplier = cost_spike_multiplier
        self._traces: dict[str, _SessionTrace] = defaultdict(_SessionTrace)

    def start_turn(self, session_id: str, original_query: str) -> None:
        """只在该会话首次运行时固定原始目标，避免被后续动作覆盖。"""

        trace = self._traces[session_id]
        if not trace.original_query:
            trace.original_query = original_query
            trace.query_keywords = _keywords(original_query)

    def observe_action(
        self,
        session_id: str,
        summary: str,
        *,
        result_empty: bool = False,
        tokens: int = 0,
    ) -> None:
        """记录一次工具动作摘要；只保留最近三次，限制内存增长。"""

        trace = self._traces[session_id]
        trace.rounds += 1
        trace.recent_actions.append(summary[:1000])
        del trace.recent_actions[:-3]
        trace.consecutive_empty = trace.consecutive_empty + 1 if result_empty else 0
        if tokens > 0:
            trace.token_history.append(tokens)

    def check(self, session_id: str) -> DriftReport:
        """每 N 个工具动作执行一次纯计算检测，不额外调用 LLM。"""

        trace = self._traces[session_id]
        if trace.rounds == 0 or trace.rounds % self._check_interval:
            return DriftReport()
        reasons: list[str] = []
        if trace.query_keywords and trace.recent_actions:
            recent = _keywords(" ".join(trace.recent_actions))
            hit_ratio = len(trace.query_keywords & recent) / len(trace.query_keywords)
            if hit_ratio < self._keyword_hit_floor:
                reasons.append(f"最近动作与原始目标关键词命中率仅 {hit_ratio:.0%}")
        if trace.consecutive_empty >= self._empty_result_limit:
            reasons.append(f"连续 {trace.consecutive_empty} 次工具结果为空")
        history = trace.token_history
        if len(history) >= 6:
            recent_average = sum(history[-3:]) / 3
            baseline = sum(history[:-3]) / len(history[:-3])
            if baseline > 0 and recent_average > baseline * self._cost_spike_multiplier:
                reasons.append("最近三轮 Token 成本超过历史均值两倍")
        return DriftReport(tuple(reasons))

    def reset(self, session_id: str) -> None:
        """购物会话明确结束时可主动释放检测状态。"""

        self._traces.pop(session_id, None)
