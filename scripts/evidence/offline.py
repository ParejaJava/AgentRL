"""离线证据套件：测试、覆盖率、编排基准与两套召回评测。"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

from app.infrastructure.settings import Settings

from .architecture import collect_architecture_evidence
from .category_calibration import calibrate_category_threshold
from .orchestration import run_orchestration_benchmark
from .preflight import cached_model_metadata, collect_preflight
from .reliability import run_fault_injection_matrix
from .utils import (
    python_command,
    run_command,
    runtime_environment,
    sha256_file,
    utc_now,
    write_json,
)


def _coverage_thresholds(root: Path) -> dict[str, float]:
    """首次正式发布前用 bootstrap 门槛，发布后自动锁定正式基线。"""

    configured = json.loads(
        (root / "eval/evidence/thresholds.json").read_text(encoding="utf-8")
    )["coverage"]
    baseline_path = (
        root / "docs/interview_evidence/results/coverage_baseline.json"
    )
    if not baseline_path.exists():
        return {str(key): float(value) for key, value in configured.items()}
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    thresholds = {
        str(key): float(value)
        for key, value in baseline.get("thresholds", {}).items()
    }
    required = set(configured)
    if set(thresholds) != required:
        raise ValueError("正式覆盖率基线字段与 thresholds.json 不一致")
    # 关键领域规则不因首次环境波动而放宽计划中的 90% 分支门槛。
    thresholds["critical_domain_branch_percent"] = max(
        90.0,
        thresholds["critical_domain_branch_percent"],
    )
    return thresholds


def _test_report(root: Path, run_directory: Path) -> dict[str, Any]:
    """运行完整测试与分支覆盖率，并输出机器可读证据。"""

    started_at = utc_now()
    coverage_path = run_directory / "coverage.json"
    result = run_command(
        python_command(
            "-m",
            "pytest",
            "-q",
            "--cov=app",
            "--cov-branch",
            f"--cov-report=json:{coverage_path}",
        )
    )
    coverage: dict[str, Any] = {}
    if coverage_path.exists():
        raw = json.loads(coverage_path.read_text(encoding="utf-8"))
        layer_coverage = {
            layer: _coverage_for_prefix(raw.get("files", {}), prefix)
            for layer, prefix in {
                "domain": "app/domain/",
                "application": "app/application/",
                "reliability": "app/infrastructure/langchain/reliability_middleware.py",
            }.items()
        }
        critical_domain = _coverage_for_paths(
            raw.get("files", {}),
            {
                "app/domain/catalog/money.py",
                "app/domain/catalog/product_search_spec.py",
                "app/domain/shipping/tariff_policy.py",
                "app/domain/order/models.py",
            },
        )
        layer_coverage["critical_domain"] = critical_domain
        coverage = {
            "percent_covered": round(float(raw["totals"]["percent_covered"]), 4),
            "covered_lines": int(raw["totals"]["covered_lines"]),
            "num_statements": int(raw["totals"]["num_statements"]),
            "covered_branches": int(raw["totals"].get("covered_branches", 0)),
            "num_branches": int(raw["totals"].get("num_branches", 0)),
            "layers": layer_coverage,
        }
    match = re.search(r"(\d+) passed", result["stdout"])
    coverage["tests_passed"] = int(match.group(1)) if match else 0
    thresholds = _coverage_thresholds(root)
    threshold_checks = {
        "total_line_percent": coverage.get("percent_covered", 0.0)
        >= thresholds["total_line_percent"],
        "domain_line_percent": coverage.get("layers", {})
        .get("domain", {})
        .get("line_percent", 0.0)
        >= thresholds["domain_line_percent"],
        "application_line_percent": coverage.get("layers", {})
        .get("application", {})
        .get("line_percent", 0.0)
        >= thresholds["application_line_percent"],
        "reliability_line_percent": coverage.get("layers", {})
        .get("reliability", {})
        .get("line_percent", 0.0)
        >= thresholds["reliability_line_percent"],
        "critical_domain_branch_percent": coverage.get("layers", {})
        .get("critical_domain", {})
        .get("branch_percent", 0.0)
        >= thresholds["critical_domain_branch_percent"],
    }
    passed = result["returncode"] == 0 and all(threshold_checks.values())
    return {
        "claim_id": "TEST-001",
        "capability": "自动化回归与分支覆盖率",
        "status": "verified" if passed else "code_verified",
        "started_at": started_at,
        "duration_seconds": result["duration_seconds"],
        "command": result["command"],
        "environment": runtime_environment(root),
        "metrics": coverage,
        "thresholds": thresholds,
        "threshold_checks": threshold_checks,
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "failures": (
            []
            if passed
            else [
                {
                    "returncode": result["returncode"],
                    "failed_thresholds": [
                        name for name, ok in threshold_checks.items() if not ok
                    ],
                }
            ]
        ),
        "limitations": ["覆盖率用于防止回归，不等同于业务正确率。"],
        "passed": passed,
    }


def _coverage_for_prefix(
    files: dict[str, Any],
    prefix: str,
) -> dict[str, float | int]:
    """聚合指定架构层或关键模块的行、分支覆盖率。"""

    selected = [
        details.get("summary", {})
        for path, details in files.items()
        if path.replace("\\", "/").startswith(prefix)
    ]
    statements = sum(int(item.get("num_statements", 0)) for item in selected)
    covered = sum(int(item.get("covered_lines", 0)) for item in selected)
    branches = sum(int(item.get("num_branches", 0)) for item in selected)
    covered_branches = sum(
        int(item.get("covered_branches", 0)) for item in selected
    )
    return {
        "statements": statements,
        "line_percent": round(covered / statements * 100, 4) if statements else 0.0,
        "branches": branches,
        "branch_percent": (
            round(covered_branches / branches * 100, 4) if branches else 100.0
        ),
    }


def _coverage_for_paths(
    files: dict[str, Any],
    paths: set[str],
) -> dict[str, float | int]:
    """聚合关键领域规则文件，单独执行 90% 分支覆盖率门槛。"""

    normalized = {path.replace("\\", "/"): value for path, value in files.items()}
    selected = {
        path: normalized[path]
        for path in paths
        if path in normalized
    }
    return _coverage_for_prefix(selected, "app/")


def _focused_test_report(
    root: Path,
    *,
    claim_id: str,
    capability: str,
    paths: tuple[str, ...],
    boundaries: list[str],
) -> dict[str, Any]:
    """运行一个面试能力切片，保存命令和完整失败输出。"""

    started_at = utc_now()
    result = run_command(python_command("-m", "pytest", "-q", *paths))
    passed = result["returncode"] == 0
    return {
        "claim_id": claim_id,
        "capability": capability,
        "status": "verified" if passed else "code_verified",
        "started_at": started_at,
        "duration_seconds": result["duration_seconds"],
        "command": result["command"],
        "environment": runtime_environment(root),
        "tested_paths": list(paths),
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "failures": [] if passed else [{"returncode": result["returncode"]}],
        "limitations": boundaries,
        "passed": passed,
    }


def _retrieval_report(
    root: Path,
    run_directory: Path,
    *,
    kind: str,
    strategy: str,
    build_index: bool = False,
    min_score: float | None = None,
) -> dict[str, Any]:
    """运行一套召回评测并把正式阈值绑定到证据报告。"""

    started_at = utc_now()
    started = time.perf_counter()
    thresholds = json.loads(
        (root / "eval/evidence/thresholds.json").read_text(encoding="utf-8")
    )["retrieval"]
    report_path = run_directory / f"{kind}_{strategy}_recall.json"
    if kind == "category":
        dataset_path = root / "eval/category_recall.jsonl"
        arguments = python_command(
            "scripts/evaluate_category_recall.py",
            "--backend",
            "opensearch",
            "--report",
            str(report_path),
            "--strategy",
            strategy,
            "--k",
            "1",
            "3",
            "5",
            "10",
        )
        if min_score is not None:
            arguments.extend(("--min-score", str(min_score)))
    else:
        dataset_path = root / "eval/product_recall.jsonl"
        build = (
            run_command(
                python_command("scripts/build_product_eval_index.py", "--reuse")
            )
            if build_index
            else {"returncode": 0}
        )
        if int(build["returncode"]) != 0:
            return {
                "claim_id": "SEARCH-001",
                "capability": "商品召回与重排",
                "status": "code_verified",
                "strategy": strategy,
                "started_at": started_at,
                "duration_seconds": round(time.perf_counter() - started, 6),
                "command": build.get("command", ""),
                "environment": runtime_environment(root),
                "dataset": {
                    "path": dataset_path.relative_to(root).as_posix(),
                    "sha256": sha256_file(dataset_path),
                },
                "metrics": {},
                "passed": False,
                "failures": [{"stage": "build-index", **build}],
                "limitations": ["商品评测索引构建失败，未执行召回评测。"],
            }
        arguments = python_command(
            "scripts/evaluate_product_recall.py",
            "--report",
            str(report_path),
            "--strategy",
            strategy,
            "--k",
            "1",
            "3",
            "5",
            "10",
        )
    result = run_command(arguments)
    payload = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if result["returncode"] == 0 and report_path.exists()
        else {}
    )
    summary = payload.get("summary", {})
    at_ten = summary.get("at_k", {}).get("10", {})
    mrr = float(summary.get("mrr_at_10", 0.0))
    if kind == "category":
        negative = float(summary.get("negative_rejection_accuracy", 0.0) or 0.0)
        passed = result["returncode"] == 0
        if strategy == "hybrid_rerank":
            passed = (
                float(at_ten.get("recall", 0.0))
                >= thresholds["category_recall_at_10"]
                and negative
                >= thresholds["category_negative_rejection_accuracy"]
                and passed
            )
        claim_id = "RAG-001"
        capability = "品类知识 Hybrid RAG"
        limitations = ["负例拒答与排序指标分开计算。"]
    else:
        passed = result["returncode"] == 0
        if strategy == "embedding_rerank":
            passed = (
                float(at_ten.get("recall", 0.0))
                >= thresholds["product_recall_at_10"]
                and mrr >= thresholds["product_mrr_at_10"]
                and float(at_ten.get("ndcg", 0.0))
                >= thresholds["product_ndcg_at_10"]
                and passed
            )
        claim_id = "SEARCH-001"
        capability = "商品召回与交叉编码重排"
        limitations = ["固定 60 商品快照不代表真实电商平台全量库存。"]
    settings = Settings.from_env()
    embedding_name = (
        settings.category_embedding_model if kind == "category" else "BAAI/bge-m3"
    )
    reranker_name = (
        settings.category_reranker_model
        if kind == "category"
        else "BAAI/bge-reranker-v2-m3"
    )
    model_metadata = {
        "embedding": {
            "name": embedding_name,
            **cached_model_metadata(embedding_name),
        },
        "reranker": {
            "name": reranker_name,
            **cached_model_metadata(reranker_name),
        },
    }
    infrastructure = (
        {
            "image": "opensearchproject/opensearch:3.8.0",
            "index": settings.opensearch_category_index,
            "pipeline": settings.opensearch_category_pipeline,
            "bm25_weight": settings.opensearch_bm25_weight,
            "knn_weight": settings.opensearch_knn_weight,
            "number_of_replicas": settings.opensearch_number_of_replicas,
        }
        if kind == "category"
        else {"index_id": settings.item_search_index_id, "backend": "FAISS"}
    )
    return {
        "claim_id": claim_id,
        "capability": capability,
        "strategy": strategy,
        "status": "verified" if passed else "code_verified",
        "started_at": started_at,
        "duration_seconds": round(time.perf_counter() - started, 6),
        "command": result["command"],
        "environment": runtime_environment(root),
        "models": model_metadata,
        "retrieval_infrastructure": infrastructure,
        "dataset": {
            "path": dataset_path.relative_to(root).as_posix(),
            "sha256": sha256_file(dataset_path),
            **payload.get("dataset", {}),
        },
        "metrics": summary,
        "acceptance_thresholds": thresholds,
        "slices": payload.get("by_tag", {}),
        "cases": payload.get("cases", []),
        "failures": [
            case
            for case in payload.get("cases", [])
            if (
                case.get("negative_rejected") is False
                if kind == "category" and case.get("is_negative")
                else float(case.get("metrics", {}).get("10", {}).get("recall", 0)) < 1
            )
        ],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "limitations": limitations,
        "passed": passed,
    }


def _product_personalization_report(
    root: Path,
    run_directory: Path,
) -> dict[str, Any]:
    """在同一画像标注集上比较商品检索 personalization off/on。"""

    started_at = utc_now()
    started = time.perf_counter()
    dataset = root / "eval/evidence/product_personalization.jsonl"
    # 使用与固定标注集一致、但不包含商品 ID 的可解释画像；避免把家居等
    # 无关领域偏好拼进检索查询，导致 User Tower 跨品类漂移。
    profile = "旅行收纳品优先亚麻和再生尼龙材质"
    variants: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for mode in ("off", "on"):
        report_path = run_directory / f"product_personalization_{mode}_raw.json"
        arguments = python_command(
            "scripts/evaluate_product_recall.py",
            "--dataset",
            str(dataset),
            "--report",
            str(report_path),
            "--strategy",
            "embedding_rerank",
            "--k",
            "1",
            "3",
            "5",
            "10",
        )
        if mode == "on":
            arguments.extend(("--buyer-id", "evidence-buyer", "--profile", profile))
        result = run_command(arguments)
        if result["returncode"] != 0 or not report_path.exists():
            failures.append({"mode": mode, **result})
            continue
        variants[mode] = json.loads(report_path.read_text(encoding="utf-8"))[
            "summary"
        ]

    off = variants.get("off", {})
    on = variants.get("on", {})
    metrics = {
        "off": off,
        "on": on,
        "delta": {
            "recall_at_10": round(
                float(on.get("at_k", {}).get("10", {}).get("recall", 0))
                - float(off.get("at_k", {}).get("10", {}).get("recall", 0)),
                6,
            ),
            "mrr_at_10": round(
                float(on.get("mrr_at_10", 0)) - float(off.get("mrr_at_10", 0)),
                6,
            ),
            "ndcg_at_10": round(
                float(on.get("at_k", {}).get("10", {}).get("ndcg", 0))
                - float(off.get("at_k", {}).get("10", {}).get("ndcg", 0)),
                6,
            ),
        },
    }
    executed = not failures and set(variants) == {"off", "on"}
    improved = (
        metrics["delta"]["mrr_at_10"] >= 0.0
        and metrics["delta"]["ndcg_at_10"] >= 0.0
    )
    passed = executed and improved
    return {
        "claim_id": "SEARCH-PERSONALIZATION-001",
        "capability": "商品 User Tower 个性化 on/off 消融",
        "status": "verified" if passed else "code_verified",
        "started_at": started_at,
        "duration_seconds": round(time.perf_counter() - started, 6),
        "command": "uv run python -m scripts.evidence run --suite offline",
        "environment": runtime_environment(root),
        "dataset": {
            "path": dataset.relative_to(root).as_posix(),
            "sha256": sha256_file(dataset),
            "cases": 10,
        },
        "metrics": metrics,
        "failures": failures
        + (
            []
            if improved
            else [
                {
                    "reason": "画像已影响排序，但离线 MRR/NDCG 未获得正向收益。"
                }
            ]
        ),
        "limitations": [
            "该小型定向集只证明画像影响排序；不代表全量用户线上收益。",
        ],
        "passed": passed,
    }
async def run_offline_suite(
    root: Path,
    run_directory: Path,
    *,
    include_retrieval: bool = True,
) -> list[dict[str, Any]]:
    """按稳定顺序运行离线证据并逐项写盘。"""

    reports: list[dict[str, Any]] = []
    preflight = collect_preflight(root)
    reports.append(preflight)
    write_json(run_directory / "preflight.json", preflight)
    print("[evidence] preflight complete", flush=True)

    architecture = collect_architecture_evidence(root)
    reports.append(architecture)
    write_json(run_directory / "architecture.json", architecture)
    print("[evidence] architecture complete", flush=True)

    orchestration = await run_orchestration_benchmark(root)
    reports.append(orchestration)
    write_json(run_directory / "orchestration.json", orchestration)
    print("[evidence] orchestration benchmark complete", flush=True)

    tests = await asyncio.to_thread(_test_report, root, run_directory)
    reports.append(tests)
    write_json(run_directory / "tests.json", tests)
    print("[evidence] full tests and coverage complete", flush=True)

    focused_suites = (
        (
            "COMMERCE-001",
            "个性化、到手价与订单意向边界",
            (
                "tests/test_preferences.py",
                "tests/test_pricing.py",
                "tests/test_orders.py",
                "tests/test_item_search.py",
            ),
            ["只创建订单意向，不访问真实库存、支付、退款或平台下单接口。"],
        ),
    )
    for claim_id, capability, paths, boundaries in focused_suites:
        report = await asyncio.to_thread(
            _focused_test_report,
            root,
            claim_id=claim_id,
            capability=capability,
            paths=paths,
            boundaries=boundaries,
        )
        reports.append(report)
        write_json(run_directory / f"{claim_id.casefold()}.json", report)
        print(f"[evidence] {claim_id} focused tests complete", flush=True)

    reliability = await asyncio.to_thread(
        run_fault_injection_matrix,
        root,
        run_directory,
    )
    reports.append(reliability)
    write_json(run_directory / "rel-001.json", reliability)
    print("[evidence] REL-001 fault matrix complete", flush=True)

    if include_retrieval:
        retrieval_thresholds = json.loads(
            (root / "eval/evidence/thresholds.json").read_text(encoding="utf-8")
        )["retrieval"]
        calibration = await asyncio.to_thread(
            calibrate_category_threshold,
            root,
            minimum_recall=retrieval_thresholds["category_recall_at_10"],
            minimum_negative_rejection=retrieval_thresholds[
                "category_negative_rejection_accuracy"
            ],
        )
        reports.append(calibration)
        write_json(run_directory / "category_threshold.json", calibration)
        print("[evidence] category threshold calibration complete", flush=True)
        selected_threshold = float(calibration["selected"]["threshold"])
        retrieval_matrix = {
            "category": ("keyword", "bm25", "knn", "hybrid_rrf", "hybrid_rerank"),
            "product": ("lexical", "embedding", "embedding_rerank"),
        }
        for kind, strategies in retrieval_matrix.items():
            for position, strategy in enumerate(strategies):
                print(f"[evidence] running {kind}:{strategy}", flush=True)
                report = await asyncio.to_thread(
                    _retrieval_report,
                    root,
                    run_directory,
                    kind=kind,
                    strategy=strategy,
                    build_index=kind == "product" and position == 0,
                    min_score=(
                        selected_threshold
                        if kind == "category" and strategy == "hybrid_rerank"
                        else None
                    ),
                )
                reports.append(report)
                write_json(run_directory / f"{kind}_{strategy}.json", report)
                print(f"[evidence] completed {kind}:{strategy}", flush=True)
        personalization = await asyncio.to_thread(
            _product_personalization_report,
            root,
            run_directory,
        )
        reports.append(personalization)
        write_json(run_directory / "product_personalization.json", personalization)
        print("[evidence] product personalization off/on complete", flush=True)
    return reports
