"""从 Langfuse 回读并生成不含 Prompt/用户内容的 OBS-001 摘要。"""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .utils import runtime_environment, utc_now


def _as_mapping(value: Any) -> dict[str, Any]:
    """兼容 Langfuse Pydantic 响应与测试替身。"""

    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        payload = model_dump()
        return dict(payload) if isinstance(payload, Mapping) else {}
    return {}


def _safe_usage(value: Any) -> dict[str, int | float]:
    """只保留数值 Token/成本字段，不复制供应商响应内容。"""

    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if isinstance(item, (int, float)) and not isinstance(item, bool)
    }


def _observation_summary(observation: Mapping[str, Any]) -> dict[str, Any]:
    """保留调用拓扑和性能字段，明确排除 input/output/metadata。"""

    return {
        "observation_id": str(observation.get("id", "")),
        "parent_observation_id": (
            str(observation["parent_observation_id"])
            if observation.get("parent_observation_id")
            else None
        ),
        "type": str(observation.get("type", "")),
        "name": str(observation.get("name", "")),
        "start_time": str(observation.get("start_time", "")),
        "end_time": str(observation.get("end_time", "")),
        "model": str(observation.get("model", "")) or None,
        "latency_seconds": observation.get("latency"),
        "time_to_first_token_seconds": observation.get("time_to_first_token"),
        "usage": _safe_usage(observation.get("usage_details")),
        "level": str(observation.get("level", "")) or None,
    }


def _trace_summary(trace: Any, *, run_id: str, trace_id: str) -> dict[str, Any]:
    """把一条 Langfuse Trace 转为可公开、可验证的拓扑摘要。"""

    payload = _as_mapping(trace)
    observations = sorted(
        (
            _observation_summary(_as_mapping(item))
            for item in payload.get("observations", [])
        ),
        key=lambda item: item["start_time"],
    )
    agent_names = [
        item["name"] for item in observations if item["type"] == "AGENT"
    ]
    tool_names = [
        item["name"] for item in observations if item["type"] == "TOOL"
    ]
    generation_count = sum(item["type"] == "GENERATION" for item in observations)
    input_tokens = sum(
        float(item["usage"].get("input", 0))
        for item in observations
        if item["type"] == "GENERATION"
    )
    output_tokens = sum(
        float(item["usage"].get("output", 0))
        for item in observations
        if item["type"] == "GENERATION"
    )
    has_main = "globex-main-agent" in agent_names
    has_child = "globex-forked-sub-agent" in agent_names
    return {
        "run_id": run_id,
        "trace_id": trace_id,
        "langfuse_trace_id": str(payload.get("id", "")),
        "trace_name": str(payload.get("name", "")),
        "timestamp": str(payload.get("timestamp", "")),
        "latency_seconds": payload.get("latency"),
        "total_cost": payload.get("total_cost"),
        "observation_count": len(observations),
        "agent_names": agent_names,
        "tool_names": tool_names,
        "generation_count": generation_count,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "has_main_agent": has_main,
        "has_sub_agent": has_child,
        "complete_main_sub_trace": (
            has_main and has_child and generation_count > 0 and bool(tool_names)
        ),
        "observations": observations,
    }


def _candidate_runs(live_report: Mapping[str, Any], limit: int) -> list[dict[str, str]]:
    """优先选择包含 fork 的成功运行，再补充普通运行，并去除重复 Trace。"""

    cases = [item for item in live_report.get("cases", []) if isinstance(item, Mapping)]
    ordered = sorted(
        cases,
        key=lambda item: (
            "fork_sub_agents" not in item.get("tool_names", []),
            not bool(item.get("passed")),
            str(item.get("case_id", "")),
        ),
    )
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in ordered:
        trace_id = str(item.get("trace_id", ""))
        if not trace_id or trace_id in seen:
            continue
        seen.add(trace_id)
        selected.append(
            {
                "case_id": str(item.get("case_id", "")),
                "run_id": str(item.get("run_id", "")),
                "trace_id": trace_id,
            }
        )
        if len(selected) >= limit:
            break
    return selected


def collect_langfuse_trace_evidence(
    root: Path,
    live_report: Mapping[str, Any],
    *,
    context_report: Mapping[str, Any] | None = None,
    client: Any | None = None,
    trace_limit: int = 3,
) -> dict[str, Any]:
    """回读少量 Trace；至少一条完整主/子链路才把 OBS-001 标为 verified。"""

    started_at = utc_now()
    started = time.perf_counter()
    if client is None:
        from langfuse import get_client

        client = get_client()
    traces: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for candidate in _candidate_runs(live_report, trace_limit):
        try:
            trace = client.api.trace.get(
                candidate["trace_id"],
                request_options={"timeout_in_seconds": 20, "max_retries": 1},
            )
        except Exception as exc:  # noqa: BLE001 - 外部观测后端失败必须进入报告。
            failures.append(
                {
                    **candidate,
                    "error_type": type(exc).__name__,
                }
            )
            continue
        summary = _trace_summary(
            trace,
            run_id=candidate["run_id"],
            trace_id=candidate["trace_id"],
        )
        summary["case_id"] = candidate["case_id"]
        traces.append(summary)

    complete = [item for item in traces if item["complete_main_sub_trace"]]
    mappings_ok = all(item["trace_id"] == item["langfuse_trace_id"] for item in traces)
    live_cases = [
        item for item in live_report.get("cases", []) if isinstance(item, Mapping)
    ]
    full_context = (
        context_report.get("modes", {}).get("full", {})
        if isinstance(context_report, Mapping)
        else {}
    )
    event_coverage = {
        "fork": any(
            "fork_sub_agents" in item.get("tool_names", [])
            or "agent.dispatch" in item.get("event_types", [])
            for item in live_cases
        ),
        "compression": int(full_context.get("compression_calls", 0)) > 0,
        "final_result": any(
            "final.result" in item.get("event_types", []) for item in live_cases
        ),
    }
    passed = bool(complete) and mappings_ok and all(event_coverage.values())
    return {
        "claim_id": "OBS-001",
        "capability": "Langfuse 主/子 Agent 全链路 Trace 摘要",
        "status": "verified" if passed else "code_verified",
        "started_at": started_at,
        "duration_seconds": round(time.perf_counter() - started, 6),
        "command": "uv run python -m scripts.evidence run --suite live",
        "environment": runtime_environment(root),
        "metrics": {
            "requested_traces": len(_candidate_runs(live_report, trace_limit)),
            "retrieved_traces": len(traces),
            "complete_main_sub_traces": len(complete),
            "run_trace_mapping_rate": (
                round(
                    sum(
                        item["trace_id"] == item["langfuse_trace_id"]
                        for item in traces
                    )
                    / len(traces),
                    6,
                )
                if traces
                else 0.0
            ),
            "event_coverage": event_coverage,
        },
        "traces": traces,
        "failures": failures,
        "limitations": [
            "公开摘要不保存用户原始输入、完整 Prompt、模型输出、密钥或私有 Langfuse 地址。",
            "只抽样最多三条 Trace；它证明链路可回读，不代表所有遥测事件都成功上传。",
            "Fork/最终结果来自运行事件摘要，压缩事件来自 CTX-001 的 full 模式计数；公开报告不复制事件载荷。",
        ],
        "passed": passed,
    }
