"""`python -m scripts.evidence` 统一命令行入口。"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .context_live import run_context_live
from .live import run_live_suite
from .observability import collect_langfuse_trace_evidence
from .offline import run_offline_suite
from .preflight import collect_preflight
from .publisher import publish_run
from .utils import runtime_environment, write_json

ROOT = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    """定义 preflight、run 和 publish 三个稳定子命令。"""

    parser = argparse.ArgumentParser(description="Globex Agent 面试证据链")
    subcommands = parser.add_subparsers(dest="command", required=True)
    preflight = subcommands.add_parser("preflight", help="检查证据依赖")
    preflight.add_argument("--live", action="store_true", help="同时要求 Kimi/Langfuse 配置")

    run = subcommands.add_parser("run", help="运行证据套件")
    run.add_argument("--suite", choices=("offline", "live"), required=True)
    run.add_argument("--skip-retrieval", action="store_true")
    run.add_argument("--repetitions", type=int, default=3)
    run.add_argument("--max-requests", type=int, default=250)
    run.add_argument("--max-total-tokens", type=int, default=1_000_000)

    publish = subcommands.add_parser("publish", help="发布脱敏证据快照")
    publish.add_argument("--run-id", required=True)
    return parser


async def _run(args: argparse.Namespace) -> int:
    """运行指定套件并生成总清单。"""

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{args.suite}"
    directory = ROOT / "eval/reports/evidence" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    if args.suite == "offline":
        reports = await run_offline_suite(
            ROOT,
            directory,
            include_retrieval=not args.skip_retrieval,
        )
    else:
        preflight = collect_preflight(ROOT, live=True)
        write_json(directory / "preflight.json", preflight)
        if not preflight["passed"]:
            reports = [preflight]
        else:
            live = await run_live_suite(
                ROOT,
                repetitions=args.repetitions,
                max_model_requests=args.max_requests,
                max_total_tokens=args.max_total_tokens,
                runtime_root=directory / "runtime",
            )
            write_json(directory / "live.json", live)
            remaining_requests = max(0, args.max_requests - int(live["metrics"]["started_requests"]))
            remaining_tokens = max(0, args.max_total_tokens - int(live["metrics"]["observed_tokens"]))
            context = await run_context_live(
                ROOT,
                max_model_requests=remaining_requests,
                max_total_tokens=remaining_tokens,
                runtime_root=directory / "runtime",
            )
            write_json(directory / "context.json", context)
            observability = collect_langfuse_trace_evidence(
                ROOT,
                live,
                context_report=context,
            )
            write_json(directory / "observability.json", observability)
            reports = [preflight, live, context, observability]
    summary: dict[str, Any] = {
        "run_id": run_id,
        "suite": args.suite,
        "environment": runtime_environment(ROOT),
        "passed": all(bool(report.get("passed")) for report in reports),
        "reports": [
            {
                "claim_id": report.get("claim_id"),
                "status": report.get("status"),
                "passed": report.get("passed"),
            }
            for report in reports
        ],
    }
    write_json(directory / "summary.json", summary)
    print(f"run_id={run_id}")
    print(f"passed={summary['passed']}")
    print(f"report={directory.relative_to(ROOT)}")
    return 0 if summary["passed"] else 1


def main() -> None:
    """解析命令并返回适合 CI 使用的退出码。"""

    args = _parser().parse_args()
    if args.command == "preflight":
        report = collect_preflight(ROOT, live=args.live)
        print("PASS" if report["passed"] else "FAIL")
        for item in report["checks"]:
            print(f"[{('ok' if item['ok'] else 'fail')}] {item['name']}")
        raise SystemExit(0 if report["passed"] else 1)
    if args.command == "publish":
        destination = publish_run(ROOT, args.run_id)
        print(destination.relative_to(ROOT))
        return
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
