"""验证正式证据运行所需的本地与外部依赖。"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from app.infrastructure.health import OpenSearchReadinessProbe
from app.infrastructure.settings import Settings

from .utils import run_command, runtime_environment, sha256_file, utc_now


def _check(name: str, ok: bool, *, required: bool, details: Any = None) -> dict[str, Any]:
    """构造统一的依赖检查项。"""

    return {
        "name": name,
        "ok": bool(ok),
        "required": required,
        "details": details,
    }


def cached_model_metadata(model_name: str) -> dict[str, Any]:
    """只检查本地 Hugging Face 缓存，不向公网发送探测请求。"""

    path = Path(model_name)
    if path.exists():
        return {"available": True, "source": "local_path", "revision": path.name}
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(model_name, "config.json")
        if isinstance(cached, str):
            resolved = Path(cached)
            revision = resolved.parent.name if resolved.parent.name else None
            return {
                "available": True,
                "source": "huggingface_cache",
                "revision": revision,
            }
    except Exception:  # noqa: BLE001, S110 - preflight 必须汇总而不是中断。
        pass
    return {"available": False, "source": None, "revision": None}


def collect_preflight(root: Path, *, live: bool = False) -> dict[str, Any]:
    """收集证据运行前置条件，不加载大型检索模型。"""

    started_at = utc_now()
    started = time.perf_counter()
    settings = Settings.from_env()
    checks: list[dict[str, Any]] = []
    required_files = [
        root / "eval/category_recall.jsonl",
        root / "eval/product_recall.jsonl",
        root / "eval/evidence/category_threshold_dev.jsonl",
        root / "eval/evidence/product_personalization.jsonl",
        root / "eval/evidence/live_cases.jsonl",
        root / "eval/evidence/context_cases.json",
        root / "eval/evidence/claims.json",
        root / "eval/evidence/thresholds.json",
        root / "pyproject.toml",
        root / "uv.lock",
    ]
    for path in required_files:
        checks.append(
            _check(
                path.relative_to(root).as_posix(),
                path.exists(),
                required=True,
                details={
                    "sha256": sha256_file(path) if path.exists() else None,
                    "bytes": path.stat().st_size if path.exists() else None,
                },
            )
        )

    for package in (
        "langchain",
        "langgraph",
        "pytest",
        "coverage",
        "FlagEmbedding",
        "faiss",
        "opensearchpy",
        "redis",
    ):
        checks.append(
            _check(
                f"python-package:{package}",
                importlib.util.find_spec(package) is not None,
                required=True,
            )
        )

    for label, model in (
        ("embedding-model", settings.category_embedding_model),
        ("reranker-model", settings.category_reranker_model),
    ):
        details = cached_model_metadata(model)
        checks.append(
            _check(label, details["available"], required=True, details=details)
        )

    try:
        docker = run_command(
            [
                "docker",
                "compose",
                "-f",
                str(root / "docker/docker-compose.yml"),
                "ps",
                "--format",
                "json",
            ],
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        docker = {
            "returncode": -1,
            "stdout": "",
            "stderr": type(exc).__name__,
        }
    docker_output = docker["stdout"].casefold()
    docker_ok = (
        docker["returncode"] == 0
        and '"service":"redis"' in docker_output
        and '"service":"opensearch"' in docker_output
        and docker_output.count('"health":"healthy"') >= 2
    )
    checks.append(
        _check(
            "docker-services",
            docker_ok,
            required=True,
            details={
                "compose_file": "docker/docker-compose.yml",
                "redis_healthy": '"service":"redis"' in docker_output,
                "opensearch_healthy": '"service":"opensearch"' in docker_output,
            },
        )
    )

    opensearch_required = settings.category_retriever_backend == "opensearch"
    if opensearch_required:
        state = OpenSearchReadinessProbe(settings).check()
        checks.append(
            _check(
                "opensearch",
                state.ready,
                required=True,
                details=state.to_dict(),
            )
        )
    else:
        checks.append(_check("opensearch", True, required=False, details="disabled"))

    redis_ok = True
    redis_details: Any = "disabled"
    if settings.redis_enabled:
        try:
            import redis

            client = redis.Redis.from_url(settings.redis_url)
            redis_ok = bool(client.ping())
            redis_details = {"reachable": redis_ok}
        except Exception as exc:  # noqa: BLE001 - 前置检查需要汇总全部失败项。
            redis_ok = False
            redis_details = {"reachable": False, "error": type(exc).__name__}
    checks.append(_check("redis", redis_ok, required=settings.redis_enabled, details=redis_details))

    langfuse_keys = bool(
        os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
    )
    checks.append(
        _check(
            "langfuse-config",
            langfuse_keys if live else True,
            required=live,
            details={"enabled": settings.langfuse_enabled, "credentials_present": langfuse_keys},
        )
    )
    if live and langfuse_keys:
        try:
            from langfuse import Langfuse

            langfuse_reachable = bool(Langfuse().auth_check())
            langfuse_error = None
        except Exception as exc:  # noqa: BLE001 - 只报告脱敏异常类型。
            langfuse_reachable = False
            langfuse_error = type(exc).__name__
        checks.append(
            _check(
                "langfuse-connectivity",
                langfuse_reachable,
                required=True,
                details={"error": langfuse_error},
            )
        )
    llm_configured = bool(settings.llm_api_key and settings.llm_base_url)
    checks.append(
        _check(
            "llm-config",
            llm_configured if live else True,
            required=live,
            details={"model": settings.llm_model_name, "configured": llm_configured},
        )
    )

    passed = all(item["ok"] for item in checks if item["required"])
    return {
        "claim_id": "ENV-001",
        "capability": "可复现运行环境",
        "status": "verified" if passed else "planned",
        "started_at": started_at,
        "duration_seconds": round(time.perf_counter() - started, 6),
        "command": (
            "uv run python -m scripts.evidence preflight --live"
            if live
            else "uv run python -m scripts.evidence preflight"
        ),
        "passed": passed,
        "environment": runtime_environment(root),
        "checks": checks,
    }
