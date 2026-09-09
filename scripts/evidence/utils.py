"""证据脚本共享的哈希、环境、子进程和脱敏工具。"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SECRET_KEY = re.compile(
    r"(api[_-]?key|secret|password|authorization|token)$",
    re.IGNORECASE,
)
_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
_WINDOWS_PATH = re.compile(r"[A-Za-z]:\\[^\s\"']+")
_EXPOSED_SECRET = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{12,}|Bearer\s+[A-Za-z0-9._~+/-]{12,})",
    re.IGNORECASE,
)


def utc_now() -> str:
    """返回 ISO 8601 UTC 时间。"""

    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    """计算文件内容摘要，不把本地绝对路径写入报告。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: Sequence[float], ratio: float) -> float:
    """使用线性插值计算稳定的百分位数。"""

    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * ratio
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 6)


def run_command(arguments: Sequence[str], *, timeout: int = 1800) -> dict[str, Any]:
    """运行无 shell 子进程并捕获可归档结果。"""

    started = datetime.now(UTC)
    completed = subprocess.run(
        list(arguments),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
        env=os.environ.copy(),
    )
    elapsed = (datetime.now(UTC) - started).total_seconds()
    return {
        "command": subprocess.list2cmdline(list(arguments)),
        "returncode": completed.returncode,
        "duration_seconds": round(elapsed, 6),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def git_environment(root: Path) -> dict[str, Any]:
    """读取 Commit 和工作区状态，为报告建立版本锚点。"""

    revision = run_command(["git", "rev-parse", "HEAD"], timeout=30)
    status = run_command(["git", "status", "--porcelain"], timeout=30)
    return {
        "commit": revision["stdout"].strip() if revision["returncode"] == 0 else None,
        "dirty": bool(status["stdout"].strip()),
        "dirty_entries": len(status["stdout"].splitlines()),
        "repository": root.name,
    }


def runtime_environment(root: Path) -> dict[str, Any]:
    """收集不包含用户名、密钥和绝对路径的运行环境。"""

    try:
        uv = run_command(["uv", "--version"], timeout=30)
        uv_version = uv["stdout"].strip() if uv["returncode"] == 0 else None
    except OSError:
        uv_version = None
    environment: dict[str, Any] = {
        **git_environment(root),
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "uv": uv_version,
        "pyproject_sha256": (
            sha256_file(root / "pyproject.toml")
            if (root / "pyproject.toml").exists()
            else None
        ),
        "uv_lock_sha256": (
            sha256_file(root / "uv.lock") if (root / "uv.lock").exists() else None
        ),
    }
    try:
        import torch

        environment["torch"] = str(torch.__version__)
        environment["cuda_available"] = bool(torch.cuda.is_available())
        environment["cuda"] = str(torch.version.cuda) if torch.version.cuda else None
        environment["gpu"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except ImportError:
        environment.update(
            {"torch": None, "cuda_available": False, "cuda": None, "gpu": None}
        )
    return environment


def write_json(path: Path, payload: Any) -> None:
    """原子写入 UTF-8 JSON 报告。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def redact(value: Any, *, key: str = "") -> Any:
    """递归删除凭据、原始 Prompt、个人邮箱和本地绝对路径。"""

    if _SECRET_KEY.search(key):
        return "[REDACTED]"
    if key.casefold() in {
        "query",
        "message",
        "prompt",
        "raw_input",
        "final_answer",
    }:
        text = str(value)
        return {"sha256": hashlib.sha256(text.encode()).hexdigest(), "chars": len(text)}
    if isinstance(value, dict):
        return {str(item_key): redact(item, key=str(item_key)) for item_key, item in value.items()}
    if isinstance(value, list):
        return [redact(item, key=key) for item in value]
    if isinstance(value, tuple):
        return [redact(item, key=key) for item in value]
    if isinstance(value, str):
        return _WINDOWS_PATH.sub("[LOCAL_PATH]", _EMAIL.sub("[EMAIL]", value))
    return value


def sensitive_findings(value: Any) -> list[str]:
    """检查脱敏结果中是否仍含密钥、邮箱或本机绝对路径。"""

    serialized = json.dumps(value, ensure_ascii=False)
    findings: list[str] = []
    if _EXPOSED_SECRET.search(serialized):
        findings.append("credential-pattern")
    if _EMAIL.search(serialized):
        findings.append("email")
    if _WINDOWS_PATH.search(serialized):
        findings.append("local-windows-path")
    for variable in (
        "LLM_API_KEY",
        "OPENAI_API_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_PUBLIC_KEY",
        "OPENSEARCH_PASSWORD",
        "REDIS_PASSWORD",
    ):
        secret = os.getenv(variable, "")
        if len(secret) >= 8 and secret in serialized:
            findings.append(f"environment-secret:{variable}")
    return sorted(set(findings))


def python_command(*arguments: str) -> list[str]:
    """返回使用当前 uv 环境解释器的命令。"""

    return [sys.executable, *arguments]
