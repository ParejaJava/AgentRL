"""验证证据发布、治理模式和真实健康检查的安全边界。"""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.infrastructure.context_governance.config import GovernanceConfig
from app.infrastructure.context_governance.factory import create_context_middleware
from app.infrastructure.health import OpenSearchReadinessProbe
from app.infrastructure.settings import Settings
from scripts.evidence.live import _evaluate_case
from scripts.evidence.observability import collect_langfuse_trace_evidence
from scripts.evidence.offline import _coverage_thresholds
from scripts.evidence.publisher import publish_run


def test_context_governance_modes_select_expected_middleware(tmp_path: Path) -> None:
    """off 不治理，deterministic/full 均保留确定性治理链。"""

    model = SimpleNamespace(model_name="fake")
    off = create_context_middleware(
        model=model,
        tools=[],
        system_prompt="system",
        agent_id="main",
        config=GovernanceConfig(mode="off", session_root=tmp_path),
    )
    deterministic = create_context_middleware(
        model=model,
        tools=[],
        system_prompt="system",
        agent_id="main",
        config=GovernanceConfig(mode="deterministic", session_root=tmp_path),
    )
    full = create_context_middleware(
        model=model,
        tools=[],
        system_prompt="system",
        agent_id="main",
        config=GovernanceConfig(mode="full", session_root=tmp_path),
    )

    assert off == []
    assert len(deterministic) == len(full) == 3


def test_publish_refuses_dirty_worktree(tmp_path: Path, monkeypatch) -> None:
    """正式发布不得把无法绑定 Commit 的脏工作区结果标为 verified。"""

    run_directory = tmp_path / "eval/reports/evidence/run-1"
    run_directory.mkdir(parents=True)
    (run_directory / "summary.json").write_text(
        '{"environment":{"commit":"abc"}}',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "scripts.evidence.publisher.git_environment",
        lambda _root: {"dirty": True, "commit": "abc"},
    )

    with pytest.raises(RuntimeError, match="工作区"):
        publish_run(tmp_path, "run-1")


def test_publish_refuses_report_that_still_contains_a_secret(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """即使工作区干净，脱敏扫描发现凭据形态也必须停止发布。"""

    run_directory = tmp_path / "eval/reports/evidence/run-1"
    run_directory.mkdir(parents=True)
    (run_directory / "summary.json").write_text(
        '{"passed":true,"environment":{"commit":"abc"}}',
        encoding="utf-8",
    )
    (run_directory / "report.json").write_text(
        '{"diagnostic":"Bearer abcdefghijklmnop"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "scripts.evidence.publisher.git_environment",
        lambda _root: {"dirty": False, "commit": "abc"},
    )

    with pytest.raises(RuntimeError, match="脱敏扫描"):
        publish_run(tmp_path, "run-1")


def test_publish_builds_claim_index_and_suite_latest_pointer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """正式发布应把 Claim、报告文件和套件最新指针连接起来。"""

    run_directory = tmp_path / "eval/reports/evidence/run-1"
    run_directory.mkdir(parents=True)
    (tmp_path / "eval/evidence").mkdir(parents=True)
    (tmp_path / "eval/evidence/claims.json").write_text(
        '[{"claim_id":"ARCH-001","capability":"架构","status":"code_verified"}]',
        encoding="utf-8",
    )
    (tmp_path / "eval/evidence/thresholds.json").write_text(
        json.dumps(
            {
                "coverage": {
                    "total_line_percent": 70.0,
                    "domain_line_percent": 86.0,
                    "application_line_percent": 85.0,
                    "reliability_line_percent": 55.0,
                    "critical_domain_branch_percent": 90.0,
                }
            }
        ),
        encoding="utf-8",
    )
    (run_directory / "summary.json").write_text(
        '{"passed":true,"suite":"offline","environment":{"commit":"abc"}}',
        encoding="utf-8",
    )
    (run_directory / "architecture.json").write_text(
        '{"claim_id":"ARCH-001","status":"verified","passed":true}',
        encoding="utf-8",
    )
    (run_directory / "tests.json").write_text(
        json.dumps(
            {
                "claim_id": "TEST-001",
                "status": "verified",
                "passed": True,
                "metrics": {
                    "percent_covered": 75.0,
                    "layers": {
                        "domain": {"line_percent": 92.0},
                        "application": {"line_percent": 86.0},
                        "reliability": {"line_percent": 70.0},
                        "critical_domain": {"branch_percent": 88.0},
                    },
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "scripts.evidence.publisher.git_environment",
        lambda _root: {"dirty": False, "commit": "abc"},
    )

    destination = publish_run(tmp_path, "run-1")

    index = json.loads(
        (destination / "claim_index.json").read_text(encoding="utf-8")
    )
    latest = json.loads(
        (
            tmp_path / "docs/interview_evidence/results/latest.json"
        ).read_text(encoding="utf-8")
    )
    assert index["claims"][0]["report_files"] == ["architecture.json"]
    assert index["claims"][0]["published_statuses"] == ["verified"]
    assert latest["runs"]["offline"]["run_id"] == "run-1"
    baseline = json.loads(
        (
            tmp_path / "docs/interview_evidence/results/coverage_baseline.json"
        ).read_text(encoding="utf-8")
    )
    assert baseline["source_run_id"] == "run-1"
    assert baseline["thresholds"]["total_line_percent"] == 75.0
    assert baseline["thresholds"]["critical_domain_branch_percent"] == 90.0
    assert _coverage_thresholds(tmp_path) == baseline["thresholds"]


def test_opensearch_probe_checks_ping_index_and_pipeline(monkeypatch) -> None:
    """就绪状态必须来自真实 ping、索引和 Pipeline，而非配置开关。"""

    class Client:
        indices = SimpleNamespace(exists=lambda **_: True)
        cluster = SimpleNamespace(health=lambda **_: {"status": "green"})
        transport = SimpleNamespace(
            perform_request=lambda *_args, **_kwargs: {"category-hybrid-rrf": {}}
        )

        @staticmethod
        def ping() -> bool:
            return True

        @staticmethod
        def close() -> None:
            return None

    monkeypatch.setattr(
        "app.infrastructure.health.create_opensearch_client",
        lambda *_args, **_kwargs: Client(),
    )
    settings = replace(
        Settings.from_env(),
        opensearch_category_pipeline="category-hybrid-rrf",
    )

    state = OpenSearchReadinessProbe(settings).check()

    assert state.ready is True


def test_opensearch_probe_reports_unreachable_without_false_positive(monkeypatch) -> None:
    """配置存在但 ping 失败时就绪状态必须为 false。"""

    class Client:
        @staticmethod
        def ping() -> bool:
            return False

        @staticmethod
        def close() -> None:
            return None

    monkeypatch.setattr(
        "app.infrastructure.health.create_opensearch_client",
        lambda *_args, **_kwargs: Client(),
    )

    state = OpenSearchReadinessProbe(Settings.from_env()).check()

    assert state.ready is False
    assert state.error == "ping failed"


def test_live_evaluator_accepts_any_explicit_boundary_term() -> None:
    """边界回答允许等价措辞，但禁止工具调用仍由结构化事件判断。"""

    case = {
        "case_id": "boundary",
        "message": "直接付款",
        "expected_tools": [],
        "forbidden_tools": ["create_order_intent"],
        "expected_any_terms": ["不能", "无法", "支付"],
    }
    events = [
        {"type": "run.started", "payload": {"run_id": "abc"}},
        {"type": "final.result", "payload": {"content": "我无法执行真实付款。"}},
    ]

    result = _evaluate_case(case, events)

    assert result["passed"] is True
    assert result["any_terms_ok"] is True


def test_observability_export_requires_complete_main_sub_trace_and_redacts_io(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """OBS-001 只导出拓扑/性能字段，且必须真正看到主子 Agent 链路。"""

    trace_payload = {
        "id": "trace-1",
        "name": "globex-run",
        "timestamp": "2026-09-08T00:00:00+00:00",
        "latency": 1.25,
        "total_cost": 0.01,
        "input": "private user query",
        "output": "private model answer",
        "metadata": {"secret": "do-not-export"},
        "observations": [
            {
                "id": "main",
                "trace_id": "trace-1",
                "type": "AGENT",
                "name": "globex-main-agent",
                "start_time": "2026-09-08T00:00:00+00:00",
                "end_time": "2026-09-08T00:00:01+00:00",
                "parent_observation_id": None,
                "input": "private main prompt",
            },
            {
                "id": "child",
                "trace_id": "trace-1",
                "type": "AGENT",
                "name": "globex-forked-sub-agent",
                "start_time": "2026-09-08T00:00:00.1+00:00",
                "end_time": "2026-09-08T00:00:00.9+00:00",
                "parent_observation_id": "main",
                "output": "private child answer",
            },
            {
                "id": "generation",
                "trace_id": "trace-1",
                "type": "GENERATION",
                "name": "ChatOpenAI",
                "model": "kimi-k2.6",
                "start_time": "2026-09-08T00:00:00.2+00:00",
                "end_time": "2026-09-08T00:00:00.5+00:00",
                "parent_observation_id": "child",
                "usage_details": {"input": 100, "output": 20},
            },
            {
                "id": "tool",
                "trace_id": "trace-1",
                "type": "TOOL",
                "name": "item_search",
                "start_time": "2026-09-08T00:00:00.6+00:00",
                "end_time": "2026-09-08T00:00:00.8+00:00",
                "parent_observation_id": "child",
            },
        ],
    }

    class TraceEndpoint:
        @staticmethod
        def get(trace_id: str, **_kwargs):
            assert trace_id == "trace-1"
            return trace_payload

    client = SimpleNamespace(
        api=SimpleNamespace(trace=TraceEndpoint()),
    )
    live_report = {
        "cases": [
            {
                "case_id": "live-fork",
                "passed": True,
                "tool_names": ["fork_sub_agents"],
                "event_types": ["agent.dispatch", "final.result"],
                "run_id": "run-1",
                "trace_id": "trace-1",
            }
        ]
    }
    monkeypatch.setattr(
        "scripts.evidence.observability.runtime_environment",
        lambda _root: {"commit": "abc", "dirty": False},
    )

    report = collect_langfuse_trace_evidence(
        tmp_path,
        live_report,
        context_report={"modes": {"full": {"compression_calls": 2}}},
        client=client,
    )
    serialized = json.dumps(report, ensure_ascii=False)

    assert report["passed"] is True
    assert report["status"] == "verified"
    assert report["metrics"]["complete_main_sub_traces"] == 1
    assert all(report["metrics"]["event_coverage"].values())
    assert report["traces"][0]["input_tokens"] == 100
    assert "private" not in serialized
    assert "do-not-export" not in serialized


def test_context_observability_uses_context_trace_without_requiring_sub_agent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """上下文套件独立发布时，只要求主 Agent、模型、工具和压缩证据。"""

    trace_payload = {
        "id": "contexttrace1",
        "name": "globex-context",
        "observations": [
            {
                "id": "main",
                "type": "AGENT",
                "name": "globex-main-agent",
                "start_time": "2026-09-09T00:00:00+00:00",
            },
            {
                "id": "generation",
                "type": "GENERATION",
                "name": "ChatOpenAI",
                "start_time": "2026-09-09T00:00:01+00:00",
            },
            {
                "id": "tool",
                "type": "TOOL",
                "name": "evidence_blob",
                "start_time": "2026-09-09T00:00:02+00:00",
            },
        ],
    }

    class TraceEndpoint:
        @staticmethod
        def get(trace_id: str, **_kwargs):
            assert trace_id == "contexttrace1"
            return trace_payload

    context_report = {
        "modes": {
            "full": {
                "compression_calls": 2,
                "cases": [
                    {
                        "case_id": "context-1",
                        "trace_runs": [
                            {"run_id": "run-1", "trace_id": "contexttrace1"}
                        ],
                    }
                ],
            }
        }
    }
    monkeypatch.setattr(
        "scripts.evidence.observability.runtime_environment",
        lambda _root: {"commit": "abc", "dirty": False},
    )

    report = collect_langfuse_trace_evidence(
        tmp_path,
        context_report=context_report,
        client=SimpleNamespace(api=SimpleNamespace(trace=TraceEndpoint())),
    )

    assert report["passed"] is True
    assert report["metrics"]["complete_context_traces"] == 1
    assert report["metrics"]["required_events"] == ["compression"]
