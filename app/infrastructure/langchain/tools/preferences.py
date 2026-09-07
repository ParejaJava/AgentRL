"""将长期偏好用例适配为 LangChain 工具。"""

from __future__ import annotations

import json
from typing import Annotated, Literal

from langchain_core.tools import BaseTool, tool
from pydantic import Field

from app.application.memory import PreferenceStore
from app.domain.buyer import BuyerPreference, PreferenceKind
from app.infrastructure.context import ShoppingContext


def create_preference_tools(store: PreferenceStore) -> tuple[BaseTool, BaseTool]:
    """创建只允许主 Agent 使用的记住和撤回偏好工具。"""

    @tool
    async def remember_preference(
        kind: Literal["like", "dislike"],
        statement: Annotated[str, Field(min_length=1, max_length=200)],
    ) -> str:
        """记住一条跨会话稳定偏好。

        仅当用户表达长期适用的品牌、材质、风格、预算习惯或明确黑名单时使用；
        本次临时要求不要保存。kind 为 like 或 dislike，statement 是简短偏好原文。
        """

        shopping = ShoppingContext.require_current()
        preference = await store.save(
            BuyerPreference(
                buyer_id=shopping.buyer_id,
                kind=PreferenceKind(kind),
                statement=statement,
            )
        )
        return json.dumps(
            {"status": "ok", "preference": preference.to_dict()},
            ensure_ascii=False,
        )

    @tool
    async def forget_preference(
        statement: Annotated[str, Field(min_length=1, max_length=200)],
    ) -> str:
        """精确撤回一条跨会话偏好。

        仅当用户明确说明某条历史偏好不再适用时使用。statement 必须使用原始完整文本；
        未命中时工具会返回当前偏好，供模型选择正确原文后重试。
        """

        shopping = ShoppingContext.require_current()
        deleted = await store.delete_exact(shopping.buyer_id, statement)
        remaining = await store.list_by_buyer(shopping.buyer_id)
        return json.dumps(
            {
                "status": "ok",
                "deleted": deleted,
                "statement": statement,
                "remaining": [item.to_dict() for item in remaining],
            },
            ensure_ascii=False,
        )

    return remember_preference, forget_preference
