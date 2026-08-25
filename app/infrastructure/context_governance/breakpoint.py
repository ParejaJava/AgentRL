"""Provider-independent cache breakpoint 与 epoch 管理。"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import BaseTool

from .messages import canonical_json, message_payload
from .schemas import CacheBreakpoint, EpochBaseline, TaskDelta, TaskState
from .token_counter import count_messages


def _tool_payload(tool: BaseTool | dict[str, Any]) -> dict[str, Any]:
    """把工具定义转换为稳定缓存哈希所需的字典。"""

    if isinstance(tool, dict):
        return tool
    schema = tool.args_schema.model_json_schema() if tool.args_schema else {}
    return {
        "name": tool.name,
        "description": tool.description,
        "args_schema": schema,
    }


def _sha256(value: Any) -> str:
    """计算规范化对象的 SHA-256。"""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class CacheBreakpointManager:
    """维护 L0/L1 的语义哈希和 provider request 哈希。"""

    def build_breakpoints(
        self,
        *,
        epoch: int,
        system_message: BaseMessage | None,
        tools: Sequence[BaseTool | dict[str, Any]],
        baseline: EpochBaseline,
        model_name: str,
        model_settings: dict[str, Any],
    ) -> list[CacheBreakpoint]:
        """为当前稳定根和 epoch 基线生成两个缓存边界。"""

        system_payload = (
            message_payload(system_message) if system_message is not None else None
        )
        tool_payloads = [_tool_payload(tool) for tool in tools]
        l0_semantics = {
            "system": system_payload,
            "tools": tool_payloads,
        }
        l0_projection = {
            **l0_semantics,
            "model": model_name,
            "model_settings": model_settings,
        }
        baseline_message = self.baseline_message(baseline, epoch)
        l1_semantics = baseline.model_dump(mode="json")
        l1_projection = {
            "l0_projection": l0_projection,
            "baseline": message_payload(baseline_message),
        }

        l0_tokens = count_messages(
            [system_message] if system_message is not None else []
        )
        l1_tokens = l0_tokens + count_messages([baseline_message])
        return [
            CacheBreakpoint(
                epoch=epoch,
                layer="L0",
                semantic_hash=_sha256(l0_semantics),
                provider_projection_hash=_sha256(l0_projection),
                prefix_token_count=l0_tokens,
            ),
            CacheBreakpoint(
                epoch=epoch,
                layer="L1",
                semantic_hash=_sha256(l1_semantics),
                provider_projection_hash=_sha256(l1_projection),
                prefix_token_count=l1_tokens,
            ),
        ]

    def validate_breakpoints(
        self,
        stored: Sequence[dict[str, Any]],
        current: Sequence[CacheBreakpoint],
    ) -> bool:
        """验证当前实际投影是否仍与已保存的 breakpoint 一致。"""

        if len(stored) != len(current):
            return False
        parsed = [CacheBreakpoint.model_validate(item) for item in stored]
        return all(
            old.layer == new.layer
            and old.epoch == new.epoch
            and old.semantic_hash == new.semantic_hash
            and old.provider_projection_hash == new.provider_projection_hash
            for old, new in zip(parsed, current, strict=True)
        )

    def baseline_message(
        self,
        baseline: EpochBaseline,
        epoch: int,
    ) -> SystemMessage:
        """用稳定序列化格式生成 L1 Epoch Baseline 消息。"""

        content = (
            "<epoch_baseline>\n"
            f"{canonical_json(baseline.model_dump(mode='json'))}\n"
            "</epoch_baseline>"
        )
        return SystemMessage(content=content, id=f"epoch-baseline-{epoch}")

    def merge_baseline(
        self,
        *,
        task_state: TaskState,
        task_delta: TaskDelta,
        validated_facts: Sequence[str],
        artifact_refs: Sequence[str],
    ) -> EpochBaseline:
        """确定性合并已验证状态，用于无 LLM 时的 epoch roll。"""

        constraints = list(
            dict.fromkeys([*task_state.constraints, *task_delta.new_constraints])
        )
        completed = list(
            dict.fromkeys(
                [*task_state.completed_steps, *task_delta.newly_completed_steps]
            )
        )
        return EpochBaseline(
            goal=task_state.goal,
            task_phase=task_state.task_phase,
            constraints=constraints,
            baseline_plan=task_state.current_plan,
            completed_before_epoch=completed,
            validated_facts=list(dict.fromkeys(validated_facts)),
            stable_artifact_refs=list(dict.fromkeys(artifact_refs)),
        )
