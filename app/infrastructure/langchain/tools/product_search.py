"""把商品搜索用例适配为 LangChain 工具。"""

import asyncio
import json

from langchain_core.tools import BaseTool, tool

from app.application.catalog.models import ItemSearchRequest
from app.application.catalog.search_catalog import ItemSearchService


def create_item_search_tool(service: ItemSearchService) -> BaseTool:
    """创建注入商品搜索用例的 LangChain 工具。"""

    @tool
    async def item_search(
        query: str,
        index_id: str,
        user_id: str | None = None,
        top_k: int = 10,
    ) -> str:
        """在指定商品索引域中执行语义召回和重排。

        query 是用户的商品需求；index_id 是 Orchestrator 已选择的平台或商品域；
        user_id 用于读取域内买家偏好；top_k 是最终返回的商品数量。
        """

        response = await asyncio.to_thread(
            service.search,
            ItemSearchRequest(
                query=query,
                index_id=index_id,
                user_id=user_id,
                top_k=top_k,
            ),
        )
        return json.dumps(response.to_dict(), ensure_ascii=False)

    return item_search
