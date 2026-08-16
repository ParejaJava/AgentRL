"""LangChain tool facade for ItemSearch."""

import asyncio
import json

from langchain_core.tools import BaseTool, tool

from .schemas import ItemSearchRequest
from .service import ItemSearchService


def create_item_search_tool(service: ItemSearchService) -> BaseTool:
    """Inject an ItemSearch service into the Agent-facing decorated tool."""

    @tool
    async def item_search(
        query: str,
        index_id: str,
        user_id: str | None = None,
        top_k: int = 10,
    ) -> str:
        """在已确定的商品检索域中执行语义召回和重排。

        index_id 必须由当前子 Agent 根据 fork 后的任务范围确定，例如平台或商品域。
        本工具不执行平台、类目、价格等元数据过滤；它只在指定向量索引中召回
        Top-100 候选，并用 BGE cross-encoder 返回重排后的 Top-K 商品。
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

