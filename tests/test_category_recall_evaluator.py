"""验证 CategoryInsight 离线数据和排序指标。"""

from collections.abc import Sequence
from pathlib import Path

import pytest

from app.application.catalog.category_insight_models import RetrievedCategoryCard
from app.domain.catalog import (
    Bestseller,
    CategoryCard,
    CategoryCardType,
    PriceTier,
    PriceTierName,
)
from scripts.category_recall_evaluator import (
    CategoryRecallCase,
    RelevanceLabel,
    evaluate_category_recall,
    load_category_recall_cases,
)


def _card(
    source: str,
    card_type: CategoryCardType,
    suffix: str,
) -> CategoryCard:
    """构建覆盖不同评测键的合法知识卡。"""

    common = {
        "card_id": f"{source}-{card_type.value}-{suffix}",
        "category": "测试品类",
        "card_type": card_type,
        "summary": suffix,
        "raw_evidence": (suffix,),
        "last_updated": "2026-08-27T00:00:00+00:00",
        "confidence": 0.9,
        "source_document": source,
        "source_hash": "hash",
    }
    if card_type is CategoryCardType.BESTSELLER:
        return CategoryCard(
            bestsellers=(Bestseller(name=suffix, confidence=0.9),),
            **common,
        )
    if card_type is CategoryCardType.PRICE_RANGE:
        return CategoryCard(
            price_tiers=(
                PriceTier(
                    tier=PriceTierName.ENTRY,
                    currency="CNY",
                    representative_products=(suffix,),
                    description=suffix,
                    confidence=0.9,
                ),
            ),
            **common,
        )
    raise AssertionError("测试只需要 bestseller 和 price_range 卡片")


class FakeRetriever:
    """按查询返回固定排名，用于精确验证指标公式。"""

    def search(self, category: str, limit: int) -> Sequence[RetrievedCategoryCard]:
        if category == "无答案":
            return ()
        cards = (
            _card("noise.md", CategoryCardType.BESTSELLER, "无关"),
            _card("b.md", CategoryCardType.PRICE_RANGE, "次相关"),
            _card("a.md", CategoryCardType.BESTSELLER, "核心"),
            _card("a.md", CategoryCardType.BESTSELLER, "重复核心"),
        )
        return tuple(
            RetrievedCategoryCard(card=card, score=0.8)
            for card in cards[:limit]
        )


def test_dataset_contains_100_valid_and_balanced_cases() -> None:
    """固定数据规模，并保证正例与无答案负例都存在。"""

    cases = load_category_recall_cases(Path("eval/category_recall.jsonl"))

    assert len(cases) == 100
    assert sum(not case.is_negative for case in cases) == 90
    assert sum(case.is_negative for case in cases) == 10
    assert len({case.case_id for case in cases}) == 100


def test_evaluator_computes_precision_recall_mrr_ndcg_and_rejection() -> None:
    """重复卡片被折叠，四类排序指标与负例拒答分别汇总。"""

    positive = CategoryRecallCase(
        case_id="positive",
        query="测试",
        expected_category="测试品类",
        relevance=(
            RelevanceLabel("a.md", "bestseller", "核心", 3),
            RelevanceLabel("b.md", "price_range", "补充", 1),
        ),
        tags=("test",),
    )
    negative = CategoryRecallCase(
        case_id="negative",
        query="无答案",
        expected_category=None,
        relevance=(),
        tags=("negative",),
    )

    report = evaluate_category_recall(FakeRetriever(), (positive, negative), (1, 3))

    at_one = report["summary"]["at_k"]["1"]
    at_three = report["summary"]["at_k"]["3"]
    assert at_one == {"recall": 0.0, "precision": 0.0, "ndcg": 0.0}
    assert at_three["recall"] == 1.0
    assert at_three["precision"] == pytest.approx(2 / 3)
    assert 0.0 < at_three["ndcg"] < 1.0
    assert report["summary"]["mrr_at_3"] == 0.5
    assert report["summary"]["negative_rejection_accuracy"] == 1.0


def test_category_evaluator_reports_each_completed_case() -> None:
    """进度回调按数据顺序逐条执行，并携带当前序号和总数。"""

    cases = (
        CategoryRecallCase(
            case_id="first",
            query="测试",
            expected_category="测试品类",
            relevance=(RelevanceLabel("a.md", "bestseller", "核心", 3),),
            tags=(),
        ),
        CategoryRecallCase(
            case_id="second",
            query="无答案",
            expected_category=None,
            relevance=(),
            tags=(),
        ),
    )
    progress: list[tuple[int, int, str]] = []

    evaluate_category_recall(
        FakeRetriever(),
        cases,
        (3,),
        progress=lambda current, total, item: progress.append(
            (current, total, item.case_id)
        ),
    )

    assert progress == [(1, 2, "first"), (2, 2, "second")]
