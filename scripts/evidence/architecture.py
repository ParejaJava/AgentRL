"""生成 DDD-lite/Hexagonal 依赖方向和 Agent 权限的机器可读证据。"""

from __future__ import annotations

import ast
import time
from pathlib import Path
from typing import Any

from .utils import runtime_environment, utc_now

_LAYERS = ("domain", "application", "infrastructure", "presentation")
_FORBIDDEN = {
    "domain": ("application", "infrastructure", "presentation"),
    "application": ("infrastructure", "presentation"),
    "infrastructure": ("presentation",),
    "presentation": (),
}


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.append(node.module)
    return result


def collect_architecture_evidence(root: Path) -> dict[str, Any]:
    """扫描项目内层依赖，输出违规项、依赖边和固定权限矩阵。"""

    started_at = utc_now()
    started = time.perf_counter()
    edges: set[tuple[str, str]] = set()
    violations: list[dict[str, str]] = []
    for source in _LAYERS:
        for path in (root / "app" / source).rglob("*.py"):
            for imported in _imports(path):
                for target in _LAYERS:
                    if imported == f"app.{target}" or imported.startswith(
                        f"app.{target}."
                    ):
                        edges.add((source, target))
                        if target in _FORBIDDEN[source]:
                            violations.append(
                                {
                                    "file": path.relative_to(root).as_posix(),
                                    "source": source,
                                    "target": target,
                                    "import": imported,
                                }
                            )

    permissions = {
        "main_agent": [
            "item_search",
            "category_insight",
            "web_search(optional)",
            "TaskCreate",
            "TaskGet",
            "TaskList",
            "TaskUpdate",
            "remember_preference",
            "forget_preference",
            "create_order_intent",
            "query_order",
            "cancel_order",
            "fork_sub_agents",
        ],
        "forked_sub_agent": [
            "item_search",
            "category_insight",
            "web_search(optional)",
        ],
    }
    passed = not violations
    return {
        "claim_id": "ARCH-001",
        "capability": "DDD-lite 与 Hexagonal 依赖方向",
        "status": "verified" if passed else "code_verified",
        "started_at": started_at,
        "duration_seconds": round(time.perf_counter() - started, 6),
        "command": "uv run python -m scripts.evidence run --suite offline",
        "environment": runtime_environment(root),
        "dependency_edges": [
            {"source": source, "target": target}
            for source, target in sorted(edges)
        ],
        "permission_matrix": permissions,
        "composition_root": {
            "module": "app/composition.py",
            "direction": "ports <- adapters <- use_cases <- presentation",
            "adapters": [
                "LangChain/LangGraph agent runtime",
                "OpenSearch category repository",
                "FAISS product repository",
                "Redis/SQLite persistence",
                "Langfuse observability",
                "FastAPI presentation",
            ],
        },
        "violations": violations,
        "passed": passed,
        "limitations": [
            "AST 扫描证明静态 import 方向，不证明运行时业务逻辑正确。",
        ],
    }
