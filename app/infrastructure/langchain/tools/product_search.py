"""把商品搜索用例适配为 LangChain 工具。"""

import asyncio
import json
from typing import Annotated

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.application.catalog.models import ItemSearchCommand
from app.application.catalog.search_catalog import ItemSearchService
from app.domain.catalog import ProductSearchSpec
from app.infrastructure.context import ShoppingContext


class ProductSearchToolInput(BaseModel):
    """LLM 可见的商品搜索参数及调用前校验规则。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    normalized_query: Annotated[
        str,
        Field(min_length=1, max_length=500),
    ]
    category: Annotated[str | None, Field(max_length=100)] = None
    ship_to: Annotated[
        str | None,
        Field(pattern=r"^[A-Za-z]{2}$"),
    ] = None
    top_k: Annotated[int, Field(ge=1, le=50)] = 5
    price_max_major: Annotated[float | None, Field(ge=0)] = None
    target_currency: Annotated[
        str,
        Field(pattern=r"^[A-Za-z]{3}$"),
    ] = "CNY"

    @field_validator("ship_to", "target_currency")
    @classmethod
    def normalize_codes(cls, value: str | None) -> str | None:
        """把国家/地区与币种代码统一为大写形式。"""

        return value.upper() if value is not None else None


def create_item_search_tool(
    service: ItemSearchService,
    *,
    index_id: str = "evaluation-products",
) -> BaseTool:
    """创建商品搜索工具，并把内部索引域作为运行时依赖注入。"""

    @tool(args_schema=ProductSearchToolInput)
    async def item_search(
        normalized_query: str,
        category: str | None = None,
        ship_to: str | None = None,
        top_k: int = 5,
        price_max_major: float | None = None,
        target_currency: str = "CNY",
    ) -> str:
        """检索商品，并执行语义召回、重排和结构化硬约束过滤。

        Args:
            normalized_query: 标准化检索词，保留品类词和关键属性词。
            category: 标准化品类名；传入时限制为该品类。
            ship_to: 两位收货国家或地区代码；传入时过滤不可配送商品并估算到手价。
            top_k: 最终返回的候选商品数量，范围为 1 到 50。
            price_max_major: 目标币种主单位的价格上限；有预算硬约束时传入。
            target_currency: 三位价格币种代码，默认 CNY。

        Returns:
            JSON 字符串，包含排序商品卡、召回策略、硬过滤摘要；传入收货地时商品卡
            还包含商品小计、运费、关税、到手总价及规则版本。
        """

        # buyer_id 属于执行上下文，不应让模型猜测或作为工具参数传入。
        shopping = ShoppingContext.current()
        user_id = shopping.buyer_id if shopping is not None else None
        try:
            spec = ProductSearchSpec(
                normalized_query=normalized_query,
                category=category,
                ship_to=ship_to,
                locale=shopping.locale if shopping is not None else "zh-CN",
                top_k=top_k,
                price_max_major=price_max_major,
                target_currency=target_currency,
            )
            response = await asyncio.to_thread(
                service.search,
                ItemSearchCommand(
                    spec=spec,
                    index_id=index_id,
                    buyer_id=user_id,
                ),
            )
        except (TypeError, ValueError) as exc:
            return json.dumps(
                {
                    "status": "invalid_request",
                    "message": str(exc),
                },
                ensure_ascii=False,
            )
        return json.dumps(response.to_dict(), ensure_ascii=False)

    return item_search
