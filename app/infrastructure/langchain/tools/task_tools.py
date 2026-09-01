"""把会话级任务服务适配为 MainAgent 专属 LangChain 工具。"""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.tools import BaseTool, tool

from app.application.tasking import TaskBoardService, TaskingError
from app.infrastructure.context import require_context


def _json_response(payload: dict[str, Any]) -> str:
    """统一生成模型容易稳定解析的中文 JSON 工具结果。"""

    return json.dumps(payload, ensure_ascii=False)


def create_task_tools(service: TaskBoardService) -> tuple[BaseTool, ...]:
    """创建 TaskCreate、TaskGet、TaskList 和 TaskUpdate 四个工具。"""

    @tool("TaskCreate")
    async def task_create(
        subject: str,
        description: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """为当前会话创建一个结构化任务。

        适用于至少三个步骤、存在依赖或需要并行执行的复杂请求；简单问答不要创建
        任务。subject 是简短且可执行的任务标题，description 是完整、自包含的任务
        说明，metadata 可保存额外的结构化信息。新任务状态固定为 pending。创建前宜先
        调用 TaskList，避免重复任务。
        """

        try:
            task = await service.create(
                require_context().thread_id,
                subject=subject,
                description=description,
                metadata=metadata,
            )
        except TaskingError as exc:
            return _json_response({"status": "error", "message": str(exc)})
        return _json_response(
            {
                "status": "ok",
                "message": f"任务 {task.id} 创建成功",
                "task": task.to_dict(),
            }
        )

    @tool("TaskGet")
    async def task_get(task_id: str) -> str:
        """读取当前会话中一个任务的完整信息。

        task_id 是 TaskCreate 或 TaskList 返回的任务 ID。在开始任务或修改任务之前
        使用本工具核对最新状态、负责人、依赖、结果和错误。active_blocked_by 非空
        时任务尚不能开始。
        """

        try:
            task = await service.get(require_context().thread_id, task_id)
        except TaskingError as exc:
            return _json_response({"status": "error", "message": str(exc)})
        return _json_response({"status": "ok", "task": task.to_dict()})

    @tool("TaskList")
    async def task_list() -> str:
        """列出当前会话中的完整任务看板。

        用于查看总体进度、寻找 runnable=true 的任务、识别依赖阻塞和选择下一项
        工作。多个 runnable 任务需要并行时，把它们的 task_id 交给
        fork_sub_agents；不要把被阻塞任务派发出去。
        """

        try:
            tasks = await service.list(require_context().thread_id)
        except TaskingError as exc:
            return _json_response({"status": "error", "message": str(exc)})
        return _json_response(
            {
                "status": "ok",
                "count": len(tasks),
                "tasks": [task.to_dict() for task in tasks],
            }
        )

    @tool("TaskUpdate")
    async def task_update(
        task_id: str,
        subject: str | None = None,
        description: str | None = None,
        add_blocks: list[str] | None = None,
        status: Literal[
            "pending",
            "in_progress",
            "completed",
            "failed",
            "deleted",
        ]
        | None = None,
        add_blocked_by: list[str] | None = None,
        owner: str | None = None,
        metadata: dict[str, Any] | None = None,
        result: str | None = None,
    ) -> str:
        """更新当前会话中的任务状态、内容、负责人、依赖或结果。

        task_id 是待修改任务 ID。add_blocks 表示哪些任务必须等待当前任务完成，
        add_blocked_by 表示当前任务必须等待哪些任务完成。metadata 会合并更新，值为
        null 的键会被删除。开始人工执行前设置 in_progress；完全完成后设置 completed
        并填写 result；无法完成时设置 failed。deleted 会永久删除任务和关联依赖。
        并行子任务由 fork_sub_agents 自动认领和回写，不应在派发前手工改为
        in_progress。
        """

        try:
            task = await service.update(
                require_context().thread_id,
                task_id,
                subject=subject,
                description=description,
                add_blocks=add_blocks,
                status=status,
                add_blocked_by=add_blocked_by,
                owner=owner,
                metadata=metadata,
                result=result,
            )
        except TaskingError as exc:
            return _json_response({"status": "error", "message": str(exc)})
        if task is None:
            return _json_response(
                {
                    "status": "ok",
                    "message": f"任务 {task_id} 已删除",
                }
            )
        return _json_response(
            {
                "status": "ok",
                "message": f"任务 {task_id} 更新成功",
                "task": task.to_dict(),
            }
        )

    return task_create, task_get, task_list, task_update
