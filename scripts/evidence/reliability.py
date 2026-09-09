"""确定性故障注入矩阵及逐用例证据报告。"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .utils import python_command, run_command, runtime_environment, utc_now

_FAULT_CASES = (
    (
        "model_request_cap",
        "tests/test_platform_extensions.py::test_model_gateway_enforces_evidence_request_limit",
    ),
    (
        "model_429_retry",
        "tests/test_platform_extensions.py::test_model_gateway_retries_transient_failure_and_records_usage",
    ),
    (
        "model_timeout_and_temporary_retry",
        "tests/test_platform_extensions.py::test_model_gateway_retries_timeout_and_temporary_errors",
    ),
    (
        "tool_timeout_no_side_effect",
        "tests/test_platform_extensions.py::test_tool_resilience_returns_structured_timeout_without_side_effect",
    ),
    (
        "tool_exception_circuit_breaker",
        "tests/test_platform_extensions.py::test_tool_exception_opens_circuit_and_prevents_second_execution",
    ),
    (
        "redis_breaker_unavailable",
        "tests/test_platform_extensions.py::test_redis_shared_breaker_fails_open_when_redis_is_unavailable",
    ),
    (
        "order_without_search_evidence",
        "tests/test_platform_extensions.py::test_tool_harness_rejects_order_without_search_evidence",
    ),
    (
        "product_reranker_failure",
        "tests/test_item_search.py::test_item_search_falls_back_to_embedding_order_when_reranker_fails",
    ),
    (
        "product_embedding_failure",
        "tests/test_item_search.py::test_item_search_falls_back_to_keyword_when_embedding_fails",
    ),
    (
        "category_reranker_failure",
        "tests/test_opensearch_category_insight.py::test_search_keeps_hybrid_order_when_reranker_fails",
    ),
    (
        "category_embedding_failure",
        "tests/test_opensearch_category_insight.py::test_search_uses_local_keyword_fallback_when_embedding_fails",
    ),
    (
        "opensearch_unreachable",
        "tests/test_evidence_infrastructure.py::test_opensearch_probe_reports_unreachable_without_false_positive",
    ),
    (
        "context_compressor_failure_boundary",
        "tests/test_context_governance.py::test_governance_keeps_original_events_when_compressor_fails",
    ),
    (
        "context_overflow_emergency_projection",
        "tests/test_context_governance.py::test_context_overflow_retries_once_with_emergency_projection",
    ),
    (
        "large_tool_result_offload",
        "tests/test_context_governance.py::test_tool_middleware_offloads_large_result_without_changing_call_id",
    ),
    (
        "child_context_isolation",
        "tests/test_fork_sub_agents.py::test_forked_agent_loop_uses_independent_threads",
    ),
    (
        "single_worker_failure_settled",
        "tests/test_tasking.py::test_deterministic_dispatch_claims_and_writes_back_results",
    ),
)


def run_fault_injection_matrix(root: Path, run_directory: Path) -> dict[str, Any]:
    """运行固定故障节点并从 JUnit XML 提取每个场景的真实结果。"""

    started_at = utc_now()
    started = time.perf_counter()
    junit_path = run_directory / "reliability-junit.xml"
    node_ids = [node_id for _, node_id in _FAULT_CASES]
    result = run_command(
        python_command(
            "-m",
            "pytest",
            "-q",
            f"--junitxml={junit_path}",
            *node_ids,
        )
    )
    observed: dict[str, dict[str, Any]] = {}
    if junit_path.exists():
        tree = ET.parse(junit_path)
        for testcase in tree.findall(".//testcase"):
            name = str(testcase.attrib.get("name", ""))
            failed = testcase.find("failure") is not None
            errored = testcase.find("error") is not None
            skipped = testcase.find("skipped") is not None
            observed[name] = {
                "passed": not failed and not errored and not skipped,
                "duration_seconds": round(
                    float(testcase.attrib.get("time", "0") or 0),
                    6,
                ),
                "outcome": (
                    "failed"
                    if failed
                    else "error"
                    if errored
                    else "skipped"
                    if skipped
                    else "passed"
                ),
            }
    matrix: list[dict[str, Any]] = []
    for scenario, node_id in _FAULT_CASES:
        test_name = node_id.rsplit("::", 1)[-1]
        details = observed.get(
            test_name,
            {"passed": False, "duration_seconds": 0.0, "outcome": "missing"},
        )
        matrix.append({"scenario": scenario, "test": node_id, **details})
    passed = result["returncode"] == 0 and all(item["passed"] for item in matrix)
    return {
        "claim_id": "REL-001",
        "capability": "检索降级、模型限额、Redis/OpenSearch 故障与上下文保护",
        "status": "verified" if passed else "code_verified",
        "started_at": started_at,
        "duration_seconds": round(time.perf_counter() - started, 6),
        "command": result["command"],
        "environment": runtime_environment(root),
        "metrics": {
            "scenarios": len(matrix),
            "passed_scenarios": sum(bool(item["passed"]) for item in matrix),
            "pass_rate": round(
                sum(bool(item["passed"]) for item in matrix) / len(matrix),
                6,
            ),
        },
        "matrix": matrix,
        "failures": [item for item in matrix if not item["passed"]],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "limitations": [
            "故障注入使用确定性替身，不等价于公网服务的生产故障率。",
            "Redis 故障场景验证共享熔断 fail-open；事件总线本身不承诺自动切换后端。",
        ],
        "passed": passed,
    }
