"""将时效性网页检索适配为 LangChain 工具。"""

from __future__ import annotations

import json
from typing import Annotated

from langchain_core.tools import BaseTool, tool
from pydantic import Field

from app.application.web_search import WebSearch


def create_web_search_tool(search: WebSearch) -> BaseTool:
    """创建仅在供应商已配置时注册的网页检索工具。"""

    @tool
    async def web_search(
        query: Annotated[str, Field(min_length=2, max_length=500)],
        max_results: Annotated[int, Field(ge=1, le=10)] = 5,
    ) -> str:
        """查询可能随时间变化的跨境政策、法规和平台规则。

        query 应包含国家、平台、主题和时间范围；max_results 控制返回证据数。
        商品库存和价格仍应使用 item_search，不要用网页结果冒充可购买商品事实。
        """

        results = await search.search(query, max_results)
        return json.dumps({"query": query, "results": results}, ensure_ascii=False)

    return web_search
