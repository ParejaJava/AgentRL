"""品类知识查询使用的出站端口。"""

from collections.abc import Sequence
from typing import Protocol

from .category_insight_models import RetrievedCategoryCard


class CategoryKnowledgeRetriever(Protocol):
    """检索结构化知识卡片；未来 OpenSearch 适配器实现此端口。"""

    def search(self, category: str, limit: int) -> Sequence[RetrievedCategoryCard]: ...
