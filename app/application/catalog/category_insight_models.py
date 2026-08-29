"""品类洞察用例的输入和检索结果模型。"""

from dataclasses import dataclass
from typing import Literal

from app.domain.catalog import CategoryCard, CategoryInsight

from .models import RecallStrategy

InsightDepth = Literal["quick", "deep"]


@dataclass(frozen=True, slots=True)
class CategoryInsightRequest:
    """请求一个标准化品类的结构化常识。"""

    category: str
    depth: InsightDepth = "quick"


@dataclass(frozen=True, slots=True)
class RetrievedCategoryCard:
    """检索端口返回的知识卡片及归一化相关度。"""

    card: CategoryCard
    score: float
    recall_strategy: RecallStrategy = "embedding_rerank"

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("retrieval score 必须位于 0 到 1 之间")


@dataclass(frozen=True, slots=True)
class CategoryInsightResult:
    """品类洞察及本次检索实际采用的技术策略。"""

    insight: CategoryInsight
    recall_strategy: RecallStrategy

    def to_dict(self) -> dict[str, object]:
        """合并稳定领域输出和应用层检索元数据。"""

        return {
            **self.insight.to_dict(),
            "recall_strategy": self.recall_strategy,
        }
