"""防止框架依赖重新渗透到 Domain 和 Application。"""

import ast
from pathlib import Path

APP_ROOT = Path("app")
DOMAIN_ROOT = APP_ROOT / "domain"
APPLICATION_ROOT = APP_ROOT / "application"

DOMAIN_FORBIDDEN_ROOTS = {
    "fastapi",
    "langchain",
    "langchain_core",
    "langchain_openai",
    "langgraph",
    "numpy",
    "openai",
    "pydantic",
    "redis",
    "sqlite3",
}
DOMAIN_FORBIDDEN_APP_PREFIXES = (
    "app.application",
    "app.infrastructure",
    "app.presentation",
)
APPLICATION_FORBIDDEN_ROOTS = {
    "fastapi",
    "langchain",
    "langchain_core",
    "langchain_openai",
    "langgraph",
    "openai",
    "pydantic",
    "redis",
    "sqlite3",
}
APPLICATION_FORBIDDEN_APP_PREFIXES = (
    "app.infrastructure",
    "app.presentation",
)


def _imports(path: Path) -> list[str]:
    """读取一个模块的绝对导入目标。"""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports


def _assert_layer_is_clean(
    root: Path,
    *,
    forbidden_roots: set[str],
    forbidden_app_prefixes: tuple[str, ...],
) -> None:
    """检查一个内层架构目录没有导入外层实现。"""

    violations: list[str] = []
    for path in root.rglob("*.py"):
        for imported in _imports(path):
            package_root = imported.split(".", maxsplit=1)[0]
            if package_root in forbidden_roots or imported.startswith(
                forbidden_app_prefixes
            ):
                violations.append(f"{path}: {imported}")
    assert not violations, "架构依赖方向违规：\n" + "\n".join(violations)


def test_domain_is_framework_independent() -> None:
    """Domain 只能依赖标准库和自身领域模块。"""

    _assert_layer_is_clean(
        DOMAIN_ROOT,
        forbidden_roots=DOMAIN_FORBIDDEN_ROOTS,
        forbidden_app_prefixes=DOMAIN_FORBIDDEN_APP_PREFIXES,
    )


def test_application_does_not_depend_on_concrete_adapters() -> None:
    """Application 只依赖 Domain 和端口。"""

    _assert_layer_is_clean(
        APPLICATION_ROOT,
        forbidden_roots=APPLICATION_FORBIDDEN_ROOTS,
        forbidden_app_prefixes=APPLICATION_FORBIDDEN_APP_PREFIXES,
    )


def test_agent_platform_modules_do_not_return_to_domain() -> None:
    """平台运行机制属于 Application，不能再次伪装成电商 Domain。"""

    forbidden_packages = {"context", "orchestration", "runtime", "session", "tools"}
    current_packages = {
        path.name for path in DOMAIN_ROOT.iterdir() if path.is_dir()
    }

    assert not current_packages & forbidden_packages
