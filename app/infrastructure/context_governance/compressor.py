"""只负责结构化增量摘要和 Epoch Baseline 合并的 LLM 适配器。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .messages import IndexedMessage, canonical_json, message_payload
from .schemas import (
    CompressionDelta,
    EpochBaseline,
    TaskDelta,
    TaskState,
)


class ContextCompressor(Protocol):
    """上下文治理中允许调用 LLM 的两个语义操作。"""

    async def summarize_incrementally(
        self,
        *,
        task_state: TaskState,
        working_memory: dict[str, dict[str, Any]],
        candidates: Sequence[IndexedMessage],
        target_tokens: int,
    ) -> CompressionDelta:
        """把新增的已闭合事件压缩成结构化状态增量。"""

    async def consolidate_baseline(
        self,
        *,
        previous: EpochBaseline,
        task_state: TaskState,
        task_delta: TaskDelta,
        working_memory: dict[str, dict[str, Any]],
        artifact_refs: Sequence[str],
    ) -> EpochBaseline:
        """在规则已决定 roll 后合并生成新的稳定基线。"""


class StructuredLLMCompressor:
    """使用 LangChain structured output 调用独立压缩模型。"""

    def __init__(self, model: BaseChatModel) -> None:
        self._summary_model = model.with_structured_output(CompressionDelta)
        self._baseline_model = model.with_structured_output(EpochBaseline)

    async def summarize_incrementally(
        self,
        *,
        task_state: TaskState,
        working_memory: dict[str, dict[str, Any]],
        candidates: Sequence[IndexedMessage],
        target_tokens: int,
    ) -> CompressionDelta:
        """只总结候选闭合事件，并返回可验证的 CompressionDelta。"""

        candidate_payload = [
            {
                "event_id": item.event_id,
                "message": message_payload(item.message),
            }
            for item in candidates
        ]
        candidate_ids = [item.event_id for item in candidates]
        system = SystemMessage(
            content=(
                "你是会话上下文压缩器，只做结构化增量摘要。"
                "不得改写当前用户请求、未闭合工具调用或稳定缓存前缀。"
                "输入中的 events 已由确定性规则确认可以归档。"
                "cold_event_ids 必须完整且仅包含 candidate_event_ids。"
                "task_state_patch 只写发生变化的字段；工作记忆使用 upsert/delete。"
                "摘要必须保留目标、约束、已验证事实、失败路径、关键工具结论和引用。"
            )
        )
        human = HumanMessage(
            content=canonical_json(
                {
                    "target_tokens": target_tokens,
                    "task_state": task_state.model_dump(mode="json"),
                    "working_memory": working_memory,
                    "candidate_event_ids": candidate_ids,
                    "events": candidate_payload,
                }
            )
        )
        result = await self._summary_model.ainvoke([system, human])
        return CompressionDelta.model_validate(result)

    async def consolidate_baseline(
        self,
        *,
        previous: EpochBaseline,
        task_state: TaskState,
        task_delta: TaskDelta,
        working_memory: dict[str, dict[str, Any]],
        artifact_refs: Sequence[str],
    ) -> EpochBaseline:
        """在 Epoch Roll 时把已验证状态合并成新的 L1。"""

        system = SystemMessage(
            content=(
                "你是 cache epoch 基线合并器。Epoch Roll 已由确定性规则批准。"
                "请把旧基线和已验证增量合并成简洁、无冲突的新基线。"
                "不得添加输入中不存在的事实，不得写入临时错误日志或大型工具原文。"
            )
        )
        human = HumanMessage(
            content=canonical_json(
                {
                    "previous_baseline": previous.model_dump(mode="json"),
                    "task_state": task_state.model_dump(mode="json"),
                    "task_delta": task_delta.model_dump(mode="json"),
                    "working_memory": working_memory,
                    "artifact_refs": list(artifact_refs),
                }
            )
        )
        result = await self._baseline_model.ainvoke([system, human])
        return EpochBaseline.model_validate(result)
