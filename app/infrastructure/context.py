"""使用 ContextVar 承载当前异步执行链的上下文。"""

from __future__ import annotations

from contextvars import ContextVar, Token

from app.application.runtime import AgentExecutionContext, ShoppingContextSnapshot

# ContextVar 会随 asyncio Task 的创建复制上下文，但不会自动跨进程传播。
current_execution_context: ContextVar[AgentExecutionContext | None] = ContextVar(
    "globex_agent_execution_context",
    default=None,
)

# 兼容旧模块名；新代码应使用 current_execution_context。
current_context = current_execution_context

# 兼容旧构造名；新代码应使用 AgentExecutionContext。
RequestContext = AgentExecutionContext


def set_context(
    context: AgentExecutionContext,
) -> Token[AgentExecutionContext | None]:
    """为当前同步或异步调用链绑定执行上下文。"""

    return current_execution_context.set(context)


def reset_context(token: Token[AgentExecutionContext | None]) -> None:
    """在 finally 中恢复进入当前调用链之前的上下文。"""

    current_execution_context.reset(token)


def require_context() -> AgentExecutionContext:
    """返回当前执行上下文；未绑定时立即失败，避免跨会话串台。"""

    context = current_execution_context.get()
    if context is None:
        raise RuntimeError("当前调用链未绑定 AgentExecutionContext")
    return context


class ShoppingContext:
    """向业务工具暴露只读购物快照，而不暴露 ContextVar 细节。"""

    @staticmethod
    def current() -> ShoppingContextSnapshot | None:
        """读取当前购物快照；非购物型后台任务可能返回 None。"""

        context = current_execution_context.get()
        return context.shopping if context is not None else None

    @staticmethod
    def require_current() -> ShoppingContextSnapshot:
        """读取必需的购物快照，未绑定时拒绝继续执行。"""

        return require_context().require_shopping()
