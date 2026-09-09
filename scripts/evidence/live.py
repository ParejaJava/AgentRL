"""使用真实模型与 Langfuse 的受控端到端证据套件。"""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from app.application.agents import RunAgentCommand
from app.application.runtime import ShoppingContextSnapshot
from app.composition import build_container
from app.infrastructure.langchain.reliability_middleware import (
    EvidenceBudgetExceeded,
)
from app.infrastructure.settings import Settings

from .utils import runtime_environment, sha256_file, utc_now


def _load_cases(path: Path) -> tuple[dict[str, Any], ...]:
    """加载固定端到端 JSONL，并拒绝重复 ID。"""

    cases = tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    ids = [str(case["case_id"]) for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("live evidence case_id 不能重复")
    return cases


def _evaluate_case(case: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    """使用事件和结构化工具参数确定性评测一次真实 Agent Run。"""

    event_types = [str(event.get("type", "")) for event in events]
    invocations = [
        event.get("payload", {})
        for event in events
        if event.get("type") in {"tool.invoke", "agent.dispatch"}
    ]
    tool_names = [str(item.get("tool_name", "")) for item in invocations]
    expected_tools = set(map(str, case.get("expected_tools", [])))
    forbidden_tools = set(map(str, case.get("forbidden_tools", [])))
    tools_ok = expected_tools.issubset(tool_names) and not forbidden_tools.intersection(
        tool_names
    )

    expected_event = case.get("expected_event")
    event_ok = (
        str(expected_event) in event_types
        if expected_event
        else "final.result" in event_types and "error" not in event_types
    )
    argument_ok = True
    expected_argument = case.get("expected_argument")
    if isinstance(expected_argument, dict):
        argument_ok = any(
            item.get("tool_name") == expected_argument.get("tool")
            and isinstance(item.get("arguments"), dict)
            and item["arguments"].get(expected_argument.get("key"))
            == expected_argument.get("value")
            for item in invocations
        )

    final_text = "".join(
        str(event.get("payload", {}).get("content", ""))
        for event in events
        if event.get("type") == "final.result"
    )
    terms_ok = all(str(term) in final_text for term in case.get("expected_terms", []))
    expected_any_terms = [str(term) for term in case.get("expected_any_terms", [])]
    any_terms_ok = (
        any(term in final_text for term in expected_any_terms)
        if expected_any_terms
        else True
    )
    run_id = next(
        (
            str(event.get("payload", {}).get("run_id", ""))
            for event in events
            if event.get("type") == "run.started"
        ),
        "",
    )
    passed = tools_ok and event_ok and argument_ok and terms_ok and any_terms_ok
    return {
        "case_id": case["case_id"],
        "tags": case.get("tags", []),
        "message": case["message"],
        "passed": passed,
        "tools_ok": tools_ok,
        "event_ok": event_ok,
        "argument_ok": argument_ok,
        "terms_ok": terms_ok,
        "any_terms_ok": any_terms_ok,
        "tool_names": tool_names,
        "event_types": event_types,
        "final_answer": final_text,
        "run_id": run_id,
        "trace_id": run_id.replace("-", "") if run_id else "",
    }


async def run_live_suite(
    root: Path,
    *,
    repetitions: int = 3,
    max_model_requests: int = 250,
    max_total_tokens: int = 1_000_000,
    runtime_root: Path | None = None,
) -> dict[str, Any]:
    """运行真实 Kimi 用例，并在请求或 Token 达到上限时停止。"""

    case_path = root / "eval/evidence/live_cases.jsonl"
    cases = _load_cases(case_path)
    started_at = utc_now()
    started = time.perf_counter()
    evaluated: list[dict[str, Any]] = []
    stopped_reason: str | None = None
    expected_runs = len(cases) * repetitions
    temporary_parent = runtime_root or root / "eval/reports/evidence/runtime"
    temporary_parent.mkdir(parents=True, exist_ok=True)
    # Windows 上 SQLite/WAL 句柄可能比 AgentLoop 晚片刻释放；忽略清理错误，
    # 残留也只会留在 Git 忽略的原始报告目录，不会进入正式快照。
    with tempfile.TemporaryDirectory(
        prefix="globex-live-evidence-",
        dir=temporary_parent,
        ignore_cleanup_errors=True,
    ) as directory:
        temporary = Path(directory)
        base = Settings.from_env()
        settings = replace(
            base,
            model_max_concurrency=10,
            sub_agent_max_concurrency=8,
            model_min_interval_seconds=0.75,
            # 正式 E2E 证据关注模型实际决策次数；外层重试会随机挤占与
            # 上下文实验共享的 250 次硬预算。重试能力由 offline 故障矩阵独立证明。
            model_max_retries=0,
            token_budget_total=30_000,
            model_run_max_requests=max_model_requests,
            model_run_max_observed_tokens=max_total_tokens,
            context_llm_enabled=False,
            semantic_cache_enabled=False,
            embedding_cache_enabled=False,
            redis_enabled=False,
            langfuse_enabled=True,
            checkpoint_backend="memory",
            preference_database_path=temporary / "preferences.db",
            order_database_path=temporary / "orders.db",
            governance=replace(base.governance, session_root=temporary / "sessions"),
        )
        container = build_container(settings)
        for repetition in range(1, repetitions + 1):
            buyer_id = f"evidence-buyer-{repetition}"
            for case_position, case in enumerate(cases, start=1):
                usage = container.model_gateway.usage_snapshot()
                if usage["started_requests"] >= max_model_requests:
                    stopped_reason = "max_model_requests"
                    break
                if usage["observed_tokens"] >= max_total_tokens:
                    stopped_reason = "max_total_tokens"
                    break
                case_started = time.perf_counter()
                current_run = (repetition - 1) * len(cases) + case_position
                print(
                    f"[live {current_run:>2}/{expected_runs}] "
                    f"case={case['case_id']} repeat={repetition}",
                    flush=True,
                )
                command = RunAgentCommand(
                    message=str(case["message"]),
                    shopping=ShoppingContextSnapshot(
                        shopping_session_id=f"evidence-session-{repetition}",
                        buyer_id=buyer_id,
                    ),
                    thread_id=f"{case['case_id']}-repeat-{repetition}",
                    session_dir=str(temporary / "sessions" / str(case["case_id"])),
                )
                try:
                    events = [
                        event async for event in container.run_agent.execute(command)
                    ]
                except EvidenceBudgetExceeded as exc:
                    stopped_reason = type(exc).__name__
                    break
                except Exception as exc:  # noqa: BLE001 - 失败样本必须进入证据报告。
                    evaluated.append(
                        {
                            "case_id": case["case_id"],
                            "repetition": repetition,
                            "passed": False,
                            "error": type(exc).__name__,
                            "duration_seconds": round(
                                time.perf_counter() - case_started,
                                6,
                            ),
                        }
                    )
                    continue
                result = _evaluate_case(case, events)
                result["repetition"] = repetition
                result["duration_seconds"] = round(
                    time.perf_counter() - case_started,
                    6,
                )
                evaluated.append(result)
            if stopped_reason:
                break

        usage = container.model_gateway.usage_snapshot()

        if settings.langfuse_enabled:
            from langfuse import get_client

            get_client().flush()

    passed_cases = sum(bool(item["passed"]) for item in evaluated)
    pass_rate = passed_cases / len(evaluated) if evaluated else 0.0
    within_budget = (
        usage["started_requests"] <= max_model_requests
        and usage["observed_tokens"] <= max_total_tokens
    )
    usage_observable = usage["observed_tokens"] > 0
    completed_all = (
        len(evaluated) == expected_runs
        and stopped_reason is None
        and within_budget
    )
    passed = completed_all and usage_observable and pass_rate >= 0.90
    limitations = [
        "真实模型结果具有随机性，每条用例固定重复三次并保留全部失败结果。",
        "该套件验证 Agent 路由和工具参数，不使用另一个 LLM 充当裁判。",
    ]
    if usage["observed_tokens"] == 0:
        limitations.append("Kimi 响应未提供可识别 usage，无法证明总 Token 未超过预算。")
    return {
        "claim_id": "E2E-001",
        "capability": "真实 Kimi 多 Agent 端到端执行",
        "status": "verified" if passed else "code_verified",
        "started_at": started_at,
        "duration_seconds": round(time.perf_counter() - started, 6),
        "command": "uv run python -m scripts.evidence run --suite live-e2e",
        "environment": runtime_environment(root),
        "dataset": {
            "path": "eval/evidence/live_cases.jsonl",
            "sha256": sha256_file(case_path),
            "cases": len(cases),
            "repetitions": repetitions,
        },
        "runtime_policy": {
            "model_max_concurrency": settings.model_max_concurrency,
            "sub_agent_max_concurrency": settings.sub_agent_max_concurrency,
            "model_min_interval_seconds": settings.model_min_interval_seconds,
            "model_max_retries": settings.model_max_retries,
            "retry_evidence": "REL-001",
            "reason": "避免网络重试随机挤占 live 套件共享的模型请求硬预算",
        },
        "metrics": {
            "expected_runs": expected_runs,
            "completed_runs": len(evaluated),
            "pass_rate": round(pass_rate, 6),
            **usage,
            "stopped_reason": stopped_reason,
            "within_budget": within_budget,
            "usage_observable": usage_observable,
        },
        "cases": evaluated,
        "failures": [item for item in evaluated if not item["passed"]],
        "limitations": limitations,
        "passed": passed,
    }
