"""跨入口、主 Agent 与子 Agent 传递的不可变执行快照。"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ShoppingContextSnapshot:
    """一次购物会话中稳定的业务身份与本地化信息。"""

    shopping_session_id: str
    buyer_id: str
    locale: str = "zh-CN"
    currency: str = "CNY"

    def __post_init__(self) -> None:
        """拒绝会导致事件串台的空会话标识。"""

        if not self.shopping_session_id.strip():
            raise ValueError("shopping_session_id 不能为空")
        if not self.buyer_id.strip():
            raise ValueError("buyer_id 不能为空")


@dataclass(frozen=True, slots=True)
class AgentExecutionContext:
    """单条 Agent 执行链所需的显式上下文。"""

    thread_id: str
    shopping: ShoppingContextSnapshot | None = None
    run_id: str | None = None
    trace_id: str | None = None
    session_dir: str | None = None

    def __post_init__(self) -> None:
        """LangGraph 的线程标识必须显式存在。"""

        if not self.thread_id.strip():
            raise ValueError("thread_id 不能为空")

    def require_shopping(self) -> ShoppingContextSnapshot:
        """业务工具需要购物身份时，禁止静默降级到 anonymous。"""

        if self.shopping is None:
            raise RuntimeError("当前执行链未绑定 ShoppingContextSnapshot")
        return self.shopping
