"""把原始证据脱敏为可提交 Git 的版本化快照。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .utils import (
    git_environment,
    redact,
    sensitive_findings,
    utc_now,
    write_json,
)


def publish_run(root: Path, run_id: str) -> Path:
    """只允许从干净工作区发布全部通过且绑定当前 Commit 的证据。"""

    git = git_environment(root)
    if git["dirty"]:
        raise RuntimeError("正式证据发布要求 Git 工作区干净")
    source = root / "eval/reports/evidence" / run_id
    summary_path = source / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"证据运行不存在：{run_id}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("passed"):
        raise RuntimeError("证据套件未全部通过，不能发布 verified 快照")
    run_commit = summary.get("environment", {}).get("commit")
    if run_commit != git["commit"]:
        raise RuntimeError("证据运行 Commit 与当前 Commit 不一致")

    sanitized_reports: dict[str, Any] = {}
    findings: dict[str, list[str]] = {}
    for path in sorted(source.glob("*.json")):
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        sanitized = redact(payload)
        sanitized_reports[path.name] = sanitized
        report_findings = sensitive_findings(sanitized)
        if report_findings:
            findings[path.name] = report_findings
    if findings:
        raise RuntimeError(f"脱敏扫描未通过：{findings}")

    destination = root / "docs/interview_evidence/results" / run_id
    if destination.exists():
        raise FileExistsError(f"正式证据快照已存在：{run_id}")
    destination.mkdir(parents=True)
    for filename, payload in sanitized_reports.items():
        write_json(destination / filename, payload)
    definitions_path = root / "eval/evidence/claims.json"
    definitions = (
        json.loads(definitions_path.read_text(encoding="utf-8"))
        if definitions_path.exists()
        else []
    )
    reports_by_claim: dict[str, list[str]] = {}
    statuses_by_claim: dict[str, set[str]] = {}
    for filename, payload in sanitized_reports.items():
        if not isinstance(payload, dict) or not payload.get("claim_id"):
            continue
        claim_id = str(payload["claim_id"])
        reports_by_claim.setdefault(claim_id, []).append(filename)
        if payload.get("status"):
            statuses_by_claim.setdefault(claim_id, set()).add(
                str(payload["status"])
            )
    write_json(
        destination / "claim_index.json",
        {
            "run_id": run_id,
            "commit": git["commit"],
            "claims": [
                {
                    **definition,
                    "published_statuses": sorted(
                        statuses_by_claim.get(str(definition["claim_id"]), set())
                    ),
                    "report_files": sorted(
                        reports_by_claim.get(str(definition["claim_id"]), [])
                    ),
                }
                for definition in definitions
            ],
        },
    )
    write_json(
        destination / "publication.json",
        {
            "run_id": run_id,
            "commit": git["commit"],
            "published_at": utc_now(),
            "secret_scan": "passed",
            "source_reports": sorted(sanitized_reports),
        },
    )
    baseline_path = root / "docs/interview_evidence/results/coverage_baseline.json"
    if (
        str(summary.get("suite", "")) == "offline"
        and not baseline_path.exists()
        and isinstance(sanitized_reports.get("tests.json"), dict)
    ):
        metrics = sanitized_reports["tests.json"].get("metrics", {})
        layers = metrics.get("layers", {})
        write_json(
            baseline_path,
            {
                "source_run_id": run_id,
                "source_commit": git["commit"],
                "thresholds": {
                    "total_line_percent": float(
                        metrics.get("percent_covered", 0.0)
                    ),
                    "domain_line_percent": float(
                        layers.get("domain", {}).get("line_percent", 0.0)
                    ),
                    "application_line_percent": float(
                        layers.get("application", {}).get("line_percent", 0.0)
                    ),
                    "reliability_line_percent": float(
                        layers.get("reliability", {}).get("line_percent", 0.0)
                    ),
                    "critical_domain_branch_percent": max(
                        90.0,
                        float(
                            layers.get("critical_domain", {}).get(
                                "branch_percent",
                                0.0,
                            )
                        ),
                    ),
                },
            },
        )
    latest_path = root / "docs/interview_evidence/results/latest.json"
    latest = (
        json.loads(latest_path.read_text(encoding="utf-8"))
        if latest_path.exists()
        else {"runs": {}}
    )
    runs = latest.get("runs", {}) if isinstance(latest, dict) else {}
    suite = str(summary.get("suite", "unknown"))
    runs[suite] = {
        "run_id": run_id,
        "commit": git["commit"],
        "published_at": utc_now(),
        "path": f"docs/interview_evidence/results/{run_id}",
    }
    write_json(latest_path, {"runs": runs})
    # 删除可能由异常中断留下的空目录，正式快照本身保持不可覆盖。
    for directory in destination.rglob("*"):
        if directory.is_dir() and not any(directory.iterdir()):
            shutil.rmtree(directory)
    return destination
