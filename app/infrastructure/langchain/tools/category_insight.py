"""把品类洞察应用用例适配为 LangChain 工具。"""

import asyncio
import json
from typing import Literal

from langchain_core.tools import BaseTool, tool

from app.application.catalog import (
    CategoryInsightNotFound,
    CategoryInsightRequest,
    GetCategoryInsight,
)


def create_category_insight_tool(service: GetCategoryInsight) -> BaseTool:
    """创建注入品类知识用例的 LangChain 工具。"""

    @tool
    async def category_insight(
        category: str,
        depth: Literal["quick", "deep"] = "quick",
    ) -> str:
        """获取一个品类的结构化常识，不查询具体商品或实时价格。

        category 是标准化品类名，例如“旅行三件套”“咖啡杯”或“威士忌酒杯”。
        depth 为 quick 时返回典型组件、爆款、价格档和避坑点；需要详细的材质、
        参数或属性选择口径时使用 deep。工具返回已压缩知识，不包含 RAG 原文。
        """

        try:
            result = await asyncio.to_thread(
                service.execute_with_metadata,
                CategoryInsightRequest(category=category, depth=depth),
            )
        except CategoryInsightNotFound as exc:
            return json.dumps(
                {
                    "status": "not_found",
                    "category": category,
                    "depth": depth,
                    "message": str(exc),
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "status": "ok",
                "depth": depth,
                **result.to_dict(),
            },
            ensure_ascii=False,
        )

    return category_insight
