"""会话内 D4 Artifact 文件存储。"""

from __future__ import annotations

import asyncio
import hashlib
import re
from pathlib import Path
from uuid import uuid4

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


class FileArtifactStore:
    """把大型工具结果写入当前会话的 artifacts 目录。"""

    def __init__(self, session_dir: Path) -> None:
        self._session_dir = session_dir
        self._artifact_dir = session_dir / "artifacts"

    async def put_text(
        self,
        content: str,
        *,
        prefix: str = "artifact",
        suffix: str = ".txt",
    ) -> str:
        """原子写入文本并返回相对当前会话目录的引用。"""

        safe_prefix = _SAFE_NAME.sub("_", prefix).strip("._") or "artifact"
        safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        filename = f"{safe_prefix}-{digest}{safe_suffix}"
        path = self._artifact_dir / filename
        await asyncio.to_thread(self._write_if_missing, path, content)
        return path.relative_to(self._session_dir).as_posix()

    async def get_text(self, artifact_ref: str) -> str:
        """安全读取当前会话目录中的文本 Artifact。"""

        path = (self._session_dir / artifact_ref).resolve()
        root = self._session_dir.resolve()
        if path != root and root not in path.parents:
            raise ValueError("artifact_ref 超出当前会话目录")
        return await asyncio.to_thread(path.read_text, encoding="utf-8")

    @staticmethod
    def _write_if_missing(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(content, encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
