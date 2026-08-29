"""与 LangChain BaseTool 无关的平台工具契约。"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolExecutionMode(str, Enum):
    """工具由当前进程还是后台任务队列执行。"""

    INLINE = "inline"
    BACKGROUND = "background"


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """平台内部稳定的工具元数据。"""

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
    """工具执行后返回给 Runtime 的标准结果。"""

    call_id: str
    status: str
    content: str
    artifact_refs: tuple[str, ...] = ()
