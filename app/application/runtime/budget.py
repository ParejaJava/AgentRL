"""单次 Agent 意图共享的模型 Token 预算账本。"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from enum import Enum


class BudgetTier(str, Enum):
    """按照剩余预算比例划分的模型服务档位。"""

    MAIN = "main"
    LITE = "lite"
    MINIMAL = "minimal"
    FALLBACK = "fallback"


@dataclass(slots=True)
class TokenBudget:
    """主 Agent 与 fork 子任务共享的单次意图 Token 账本。"""

    total: int
    used: int = 0
    entries: list[tuple[str, int]] = field(default_factory=list)

    def charge(self, source: str, tokens: int) -> None:
        """记录一次真实模型用量；缺失或非正 usage 不污染账本。"""

        if tokens <= 0:
            return
        self.used += tokens
        self.entries.append((source, tokens))

    @property
    def remaining_ratio(self) -> float:
        """返回 0-1 的剩余预算比例。"""

        if self.total <= 0:
            return 1.0
        return max(0.0, (self.total - self.used) / self.total)

    @property
    def tier(self) -> BudgetTier:
        """使用稳定阈值选择 main/lite/minimal/fallback。"""

        ratio = self.remaining_ratio
        if ratio > 0.50:
            return BudgetTier.MAIN
        if ratio > 0.20:
            return BudgetTier.LITE
        if ratio > 0.05:
            return BudgetTier.MINIMAL
        return BudgetTier.FALLBACK


_current_budget: ContextVar[TokenBudget | None] = ContextVar(
    "globex_token_budget",
    default=None,
)


def bind_token_budget(total: int) -> Token[TokenBudget | None] | None:
    """在入口创建预算；配置为 0 时完全关闭该能力。"""

    if total <= 0:
        return None
    return _current_budget.set(TokenBudget(total=total))


def current_token_budget() -> TokenBudget | None:
    """读取当前异步执行链及其 fork 子任务共享的预算对象。"""

    return _current_budget.get()


def reset_token_budget(token: Token[TokenBudget | None] | None) -> None:
    """在 Agent Run 结束时恢复调用方原有 ContextVar。"""

    if token is not None:
        _current_budget.reset(token)
