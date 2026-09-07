"""Tavily 异步网页检索适配器。"""

from __future__ import annotations

from typing import Any


class TavilyWebSearch:
    """使用 Tavily Search API 返回带 URL 的精简证据。"""

    def __init__(self, api_key: str) -> None:
        try:
            from tavily import AsyncTavilyClient
        except ImportError as exc:
            raise RuntimeError(
                "已配置 Tavily；请运行 `uv sync --extra platform`"
            ) from exc
        self._client = AsyncTavilyClient(api_key=api_key)

    async def search(self, query: str, max_results: int) -> list[dict[str, object]]:
        """执行基础深度搜索，不把网页原文整段注入 Agent 上下文。"""

        response: dict[str, Any] = await self._client.search(
            query=query,
            search_depth="basic",
            max_results=max_results,
            include_answer=False,
            include_raw_content=False,
        )
        return [
            {
                "title": str(item.get("title", "")),
                "url": str(item.get("url", "")),
                "content": str(item.get("content", ""))[:1000],
                "score": float(item.get("score", 0.0)),
            }
            for item in response.get("results", [])
        ]
