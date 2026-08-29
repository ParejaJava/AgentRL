"""品类洞察用例的框架无关配置。"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CategoryInsightConfig:
    """控制召回规模、置信度下限和返回规模。"""

    quick_recall_k: int = 8
    deep_recall_k: int = 20
    min_confidence: float = 0.6
    max_bestsellers: int = 8
    max_attributes: int = 12
    max_price_tiers: int = 9

    def __post_init__(self) -> None:
        if self.quick_recall_k < 1:
            raise ValueError("quick_recall_k 必须大于 0")
        if self.deep_recall_k < self.quick_recall_k:
            raise ValueError("deep_recall_k 不能小于 quick_recall_k")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence 必须位于 0 到 1 之间")
