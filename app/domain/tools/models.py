"""与 LangChain BaseTool 无关的工具领域对象。"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolExecutionMode(str, Enum):
    """工具由当前进程还是任务队列执行。"""

    INLINE = "inline"
    BACKGROUND = "background"


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """平台内稳定的工具契约。"""

    name: str
    description: str
    input_schema: dict[str, Any]
    permissions: frozenset[str] = frozenset()
    execution_mode: ToolExecutionMode = ToolExecutionMode.INLINE


@dataclass(frozen=True, slots=True)
class ToolCall:
    """一次可审计的工具调用请求。"""

    call_id: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolResult:
    """工具执行后返回给 Runtime Kernel 的标准结果。"""

    call_id: str
    status: str
    content: str
    artifact_refs: tuple[str, ...] = ()
