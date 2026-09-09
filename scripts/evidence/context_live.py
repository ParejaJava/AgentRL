"""三档上下文治理的真实长会话 A/B/C 证据。"""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.tools import BaseTool, tool

from app.application.runtime import AgentExecutionContext, ShoppingContextSnapshot
from app.infrastructure.context_governance.compressor import StructuredLLMCompressor
from app.infrastructure.evidence_budget import EvidenceUsageBudget
from app.infrastructure.langchain.main_agent import MainAgent
from app.infrastructure.langchain.reliability_middleware import (
    EvidenceBudgetExceeded,
    ModelGatewayMiddleware,
)
from app.infrastructure.llm import create_chat_model, create_compression_model
from app.infrastructure.observability import LangfuseCallbacks
from app.infrastructure.settings import Settings

from .utils import runtime_environment, sha256_file, utc_now

CONTEXT_EVIDENCE_PROMPT = """你是上下文治理评测 Agent。
只处理用户明确给出的预算、目的地、候选证据和任务依赖，不补充外部事实。
当用户明确要求“加载证据”时，必须调用 evidence_blob 一次；其他时候不要调用工具。
用户纠正约束后，以新值覆盖旧值。每次回答不超过 80 个汉字。
最后一轮必须逐字包含当前有效约束和用户要求复述的任务名称。
"""


def create_evidence_blob_tool(payload_items: int = 18) -> BaseTool:
    """创建只用于长上下文实验的确定性大结果工具。"""

    @tool
    def evidence_blob(topic: str) -> str:
        """加载指定主题的固定证据块，用于验证大工具结果卸载与摘要。

        Args:
            topic: 需要加载证据的主题，例如旅行箱、露营灯或到手价。
        """

        rows = [
            {
                "evidence_id": f"{topic}-{index:02d}",
                "summary": (
                    f"{topic}证据样本{index}：这是固定的评测载荷，"
                    "用于制造可重复的大型工具返回，不代表真实商品事实。"
                ),
            }
            for index in range(1, payload_items + 1)
        ]
        return json.dumps(
            {"topic": topic, "items": rows},
            ensure_ascii=False,
            sort_keys=True,
        )

    return evidence_blob


def _prefix_stability(snapshots: list[dict[str, Any]]) -> float:
    """计算同一 Epoch、同一层内 Provider 投影哈希的一致率。"""

    groups: dict[tuple[int, str], set[str]] = {}
    observed = 0
    stable = 0
    for snapshot in snapshots:
        for item in snapshot.get("cache_breakpoints", []):
            key = (int(item.get("epoch", 0)), str(item.get("layer", "")))
            value = str(item.get("provider_projection_hash", ""))
            if not value:
                continue
            values = groups.setdefault(key, set())
            if values:
                observed += 1
                if value in values:
                    stable += 1
            values.add(value)
    return round(stable / observed, 6) if observed else 1.0


def _incremental_summary_verified(full: dict[str, Any]) -> bool:
    """确认 full 模式每个场景都至少完成一次产生净收益的增量摘要。"""

    full_cases = full.get("cases", [])
    return bool(full_cases) and all(
        int(item.get("incremental_summary_successes", 0)) > 0
        and int(item.get("compression_input_tokens", 0))
        > int(item.get("compression_output_tokens", 0))
        > 0
        for item in full_cases
    )


def _empty_report(
    root: Path,
    dataset_path: Path,
    cases: list[dict[str, Any]],
    *,
    started_at: str,
    started: float,
    reason: str,
) -> dict[str, Any]:
    """在共享证据预算已经耗尽时生成明确的未验证报告。"""

    return {
        "claim_id": "CTX-001",
        "capability": "Cache-aware 会话上下文治理",
        "status": "code_verified",
        "started_at": started_at,
        "duration_seconds": round(time.perf_counter() - started, 6),
        "command": "uv run python -m scripts.evidence run --suite live-context",
        "environment": runtime_environment(root),
        "dataset": {
            "path": "eval/evidence/context_cases.json",
            "sha256": sha256_file(dataset_path),
            "cases": len(cases),
            "modes": ["off", "deterministic", "full"],
        },
        "metrics": {
            "input_token_reduction": 0.0,
            "total_model_requests": 0,
            "total_observed_tokens": 0,
            "stopped_reason": reason,
        },
        "modes": {},
        "failures": [{"reason": reason}],
        "limitations": ["端到端套件已耗尽共享预算，上下文实验未启动。"],
        "passed": False,
    }


async def _run_mode(
    *,
    base: Settings,
    mode: str,
    cases: list[dict[str, Any]],
    temporary: Path,
    max_model_requests: int,
    max_total_tokens: int,
) -> tuple[dict[str, Any], str | None]:
    """用真实 Kimi 和真实治理中间件运行一种上下文模式。"""

    governance = replace(
        base.governance,
        mode=mode,
        session_root=temporary / mode / "sessions",
        context_window_tokens=4_096,
        summary_trigger_ratio=0.45,
        forced_summary_ratio=0.65,
        emergency_ratio=0.90,
        target_ratio=0.35,
        hot_message_count=4,
        min_summary_candidate_tokens=128,
        max_compression_input_tokens=3_000,
        tool_offload_tokens=240,
        tool_preview_chars=240,
        epoch_max_model_calls=8,
    )
    settings = replace(
        base,
        model_max_concurrency=4,
        model_min_interval_seconds=0.75,
        model_max_retries=1,
        model_run_max_requests=max_model_requests,
        model_run_max_observed_tokens=max_total_tokens,
        context_llm_enabled=mode == "full",
        langfuse_enabled=True,
        governance=governance,
    )
    shared_budget = EvidenceUsageBudget(
        max_total_requests=max_model_requests,
        max_observed_tokens=max_total_tokens,
        min_interval_seconds=0.75,
    )
    model = create_chat_model(settings, disable_thinking=True)
    gateway = ModelGatewayMiddleware(
        max_concurrency=4,
        min_interval_seconds=0.75,
        max_retries=1,
        evidence_budget=shared_budget,
    )
    compressor = (
        StructuredLLMCompressor(
            create_compression_model(settings),
            evidence_budget=shared_budget,
        )
        if mode == "full"
        else None
    )
    tools = [create_evidence_blob_tool()]
    observability = LangfuseCallbacks(enabled=settings.langfuse_enabled)
    runtime = MainAgent(
        model=model,
        tools=tools,
        governance_config=governance,
        compressor=compressor,
        shared_middleware=(gateway,),
        observability=observability,
        main_system_prompt=CONTEXT_EVIDENCE_PROMPT,
        sub_agent_system_prompt=CONTEXT_EVIDENCE_PROMPT,
    )
    mode_cases: list[dict[str, Any]] = []
    stopped_reason: str | None = None

    for case in cases:
        thread_id = f"context-{mode}-{case['case_id']}"
        shopping = ShoppingContextSnapshot(
            shopping_session_id=thread_id,
            buyer_id=f"context-buyer-{case['case_id']}",
        )
        session_dir = str(governance.session_root / str(case["case_id"]))
        snapshots: list[dict[str, Any]] = []
        turn_metrics: list[dict[str, Any]] = []
        trace_runs: list[dict[str, str]] = []
        final_answer = ""
        tool_names: list[str] = []
        for turn_index, turn in enumerate(case["turns"], start=1):
            print(
                f"[context:{mode} case={case['case_id']} "
                f"turn={turn_index}/{len(case['turns'])}]",
                flush=True,
            )
            usage_before = shared_budget.snapshot()
            # 与正式 RunAgent 用例一致：会话 thread_id 不变，每一轮 run_id 独立。
            run_id = str(uuid4())
            execution_context = AgentExecutionContext(
                thread_id=thread_id,
                shopping=shopping,
                run_id=run_id,
                trace_id=run_id.replace("-", ""),
                session_dir=session_dir,
            )
            try:
                events = [
                    event
                    async for event in runtime.stream(str(turn), execution_context)
                ]
            except EvidenceBudgetExceeded:
                stopped_reason = "evidence_budget_exceeded"
                break
            trace_runs.append(
                {
                    "run_id": run_id,
                    "trace_id": run_id.replace("-", ""),
                    "turn": str(turn_index),
                }
            )
            state = await runtime.state_snapshot(thread_id)
            final_answer = "".join(
                str(event.get("content", ""))
                for event in events
                if event.get("type") == "text_message_content"
            )
            tool_names = list(
                dict.fromkeys(
                    [
                        *tool_names,
                        *(
                            str(event.get("tool_name", ""))
                            for event in events
                            if event.get("type") == "tool_invoke"
                        ),
                    ]
                )
            )
            usage_after = shared_budget.snapshot()
            governance_snapshot = {
                "cache_epoch": state.get("cache_epoch", 0),
                "cache_breakpoints": state.get("cache_breakpoints", []),
                "token_ledger": state.get("token_ledger", {}),
                "last_governance": state.get("last_governance", {}),
                "cold_event_refs": state.get("cold_event_refs", []),
            }
            snapshots.append(governance_snapshot)
            turn_metrics.append(
                {
                    "turn": turn_index,
                    "model_requests": usage_after["started_requests"]
                    - usage_before["started_requests"],
                    "input_tokens": usage_after["observed_input_tokens"]
                    - usage_before["observed_input_tokens"],
                    "output_tokens": usage_after["observed_output_tokens"]
                    - usage_before["observed_output_tokens"],
                    "cache_epoch": int(state.get("cache_epoch", 0)),
                    "message_count": len(state.get("messages", [])),
                    "cold_refs": len(state.get("cold_event_refs", [])),
                    "governance": state.get("last_governance", {}),
                }
            )
        if stopped_reason:
            break

        final_state = snapshots[-1] if snapshots else {}
        ledger = final_state.get("token_ledger", {})
        retained = all(
            str(term) in final_answer for term in case.get("retention_terms", [])
        )
        superseded_absent = all(
            str(term) not in final_answer for term in case.get("superseded_terms", [])
        )
        compression_input = int(ledger.get("compression_input_tokens", 0))
        compression_output = int(ledger.get("compression_output_tokens", 0))
        agent_input = int(ledger.get("input_tokens", 0))
        agent_output = int(ledger.get("output_tokens", 0))
        summary_attempts = int(ledger.get("incremental_summary_attempts", 0))
        summary_successes = int(ledger.get("incremental_summary_successes", 0))
        summary_failures = int(ledger.get("incremental_summary_failures", 0))
        mode_cases.append(
            {
                "case_id": case["case_id"],
                "retained": retained and superseded_absent,
                "required_terms_retained": retained,
                "superseded_terms_absent": superseded_absent,
                "final_answer": final_answer,
                "tool_names": tool_names,
                "trace_runs": trace_runs,
                "turn_metrics": turn_metrics,
                "prefix_stability": _prefix_stability(snapshots),
                "cache_epoch_rolls": max(
                    (int(item.get("cache_epoch", 0)) for item in snapshots),
                    default=0,
                ),
                "tool_offload_count": max(
                    (len(item.get("cold_event_refs", [])) for item in snapshots),
                    default=0,
                ),
                "compression_calls": int(ledger.get("compression_calls", 0)),
                "incremental_summary_attempts": summary_attempts,
                "incremental_summary_successes": summary_successes,
                "incremental_summary_failures": summary_failures,
                "baseline_consolidation_calls": int(
                    ledger.get("baseline_consolidation_calls", 0)
                ),
                "compression_input_tokens": compression_input,
                "compression_output_tokens": compression_output,
                "agent_model_calls": int(ledger.get("model_calls", 0)),
                "agent_input_tokens": agent_input,
                "agent_output_tokens": agent_output,
                "agent_cached_input_tokens": int(
                    ledger.get("cached_input_tokens", 0)
                ),
                "compression_ratio": (
                    round(compression_output / compression_input, 6)
                    if compression_input
                    else 0.0
                ),
            }
        )

    usage = shared_budget.snapshot()
    retention_rate = (
        sum(bool(item["retained"]) for item in mode_cases) / len(mode_cases)
        if mode_cases
        else 0.0
    )
    return (
        {
            "usage": usage,
            "retention_rate": round(retention_rate, 6),
            "prefix_stability": min(
                (float(item["prefix_stability"]) for item in mode_cases),
                default=1.0,
            ),
            "compression_calls": sum(
                int(item["compression_calls"]) for item in mode_cases
            ),
            "incremental_summary_attempts": sum(
                int(item["incremental_summary_attempts"]) for item in mode_cases
            ),
            "incremental_summary_successes": sum(
                int(item["incremental_summary_successes"]) for item in mode_cases
            ),
            "incremental_summary_failures": sum(
                int(item["incremental_summary_failures"]) for item in mode_cases
            ),
            "baseline_consolidation_calls": sum(
                int(item["baseline_consolidation_calls"]) for item in mode_cases
            ),
            "tool_offload_count": sum(
                int(item["tool_offload_count"]) for item in mode_cases
            ),
            "cache_epoch_rolls": sum(
                int(item["cache_epoch_rolls"]) for item in mode_cases
            ),
            "agent_usage": {
                "model_calls": sum(
                    int(item["agent_model_calls"]) for item in mode_cases
                ),
                "input_tokens": sum(
                    int(item["agent_input_tokens"]) for item in mode_cases
                ),
                "output_tokens": sum(
                    int(item["agent_output_tokens"]) for item in mode_cases
                ),
                "cached_input_tokens": sum(
                    int(item["agent_cached_input_tokens"]) for item in mode_cases
                ),
            },
            "cases": mode_cases,
        },
        stopped_reason,
    )


async def run_context_live(
    root: Path,
    *,
    max_model_requests: int,
    max_total_tokens: int,
    runtime_root: Path | None = None,
) -> dict[str, Any]:
    """用相同长会话依次运行 off、deterministic 和 full。"""

    dataset_path = root / "eval/evidence/context_cases.json"
    cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    started_at = utc_now()
    started = time.perf_counter()
    if max_model_requests <= 0 or max_total_tokens <= 0:
        reason = "max_model_requests" if max_model_requests <= 0 else "max_total_tokens"
        return _empty_report(
            root,
            dataset_path,
            cases,
            started_at=started_at,
            started=started,
            reason=reason,
        )

    modes: dict[str, Any] = {}
    total_requests = 0
    total_tokens = 0
    stopped_reason: str | None = None
    temporary_parent = runtime_root or root / "eval/reports/evidence/runtime"
    temporary_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="globex-context-evidence-",
        dir=temporary_parent,
        ignore_cleanup_errors=True,
    ) as directory:
        temporary = Path(directory)
        base = Settings.from_env()
        for mode in ("off", "deterministic", "full"):
            remaining_requests = max_model_requests - total_requests
            remaining_tokens = max_total_tokens - total_tokens
            if remaining_requests <= 0 or remaining_tokens <= 0:
                stopped_reason = (
                    "max_model_requests"
                    if remaining_requests <= 0
                    else "max_total_tokens"
                )
                break
            mode_result, stopped_reason = await _run_mode(
                base=base,
                mode=mode,
                cases=cases,
                temporary=temporary,
                max_model_requests=remaining_requests,
                max_total_tokens=remaining_tokens,
            )
            modes[mode] = mode_result
            total_requests += int(mode_result["usage"]["started_requests"])
            total_tokens += int(mode_result["usage"]["observed_tokens"])
            if stopped_reason:
                break

        from langfuse import get_client

        get_client().flush()

    off_input = int(
        modes.get("off", {}).get("agent_usage", {}).get("input_tokens", 0)
    )
    full_input = int(
        modes.get("full", {}).get("agent_usage", {}).get("input_tokens", 0)
    )
    reduction = (
        (off_input - full_input) / off_input if off_input and full_input else 0.0
    )
    full = modes.get("full", {})
    full_cases = full.get("cases", [])
    incremental_summary_verified = _incremental_summary_verified(full)
    passed = (
        stopped_reason is None
        and len(modes) == 3
        and reduction >= 0.25
        and float(full.get("retention_rate", 0)) >= 0.95
        and float(full.get("prefix_stability", 0)) == 1.0
        and incremental_summary_verified
    )
    limitations = [
        "供应商返回 cached_input_tokens 时才可声明真实 Prompt Cache 命中率；否则只报告稳定前缀率。",
        "实验使用固定小窗口加速触发，不代表生产环境的上下文窗口大小。",
        "受控 evidence_blob 只隔离验证治理策略，不替代完整购物 Agent 端到端评测。",
        "input_token_reduction 只比较主 AgentLoop 输入；摘要模型开销单独计入 usage 与压缩字段。",
        "无净 Token 收益的摘要会被拒绝并计数，不会删除原始候选事件。",
    ]
    if full_input == 0 or off_input == 0:
        limitations.append("模型未返回输入 Token 明细，Token 降幅无法验证。")
    return {
        "claim_id": "CTX-001",
        "capability": "Cache-aware 会话上下文治理",
        "status": "verified" if passed else "code_verified",
        "started_at": started_at,
        "duration_seconds": round(time.perf_counter() - started, 6),
        "command": "uv run python -m scripts.evidence run --suite live-context",
        "environment": runtime_environment(root),
        "dataset": {
            "path": "eval/evidence/context_cases.json",
            "sha256": sha256_file(dataset_path),
            "cases": len(cases),
            "modes": ["off", "deterministic", "full"],
        },
        "runtime_policy": {
            "model_max_concurrency": base.model_max_concurrency,
            "model_min_interval_seconds": base.model_min_interval_seconds,
            "model_max_retries": base.model_max_retries,
            "shared_request_budget": max_model_requests,
            "shared_token_budget": max_total_tokens,
        },
        "metrics": {
            "input_token_reduction": round(reduction, 6),
            "input_token_reduction_scope": "agent_loop_only",
            "total_model_requests": total_requests,
            "total_observed_tokens": total_tokens,
            "stopped_reason": stopped_reason,
            "incremental_summary_verified": incremental_summary_verified,
            "rejected_summary_attempts": int(
                full.get("incremental_summary_failures", 0)
            ),
        },
        "modes": modes,
        "failures": [
            {"mode": mode, "case_id": item["case_id"]}
            for mode, result in modes.items()
            for item in result["cases"]
            if not item["retained"]
        ]
        + [
            {
                "mode": "full",
                "case_id": item["case_id"],
                "reason": "incremental_summary_not_verified",
            }
            for item in full_cases
            if not (
                int(item.get("incremental_summary_successes", 0)) > 0
                and int(item.get("compression_input_tokens", 0))
                > int(item.get("compression_output_tokens", 0))
                > 0
            )
        ],
        "limitations": limitations,
        "passed": passed,
    }
