"""验证商品评测数据适配与四项排序指标。"""

import runpy
from pathlib import Path
from typing import Any

import pytest

from app.application.catalog.models import (
    ItemSearchRequest,
    ItemSearchResponse,
    RankedItem,
)
from app.domain.catalog import Product
from scripts.product_recall_evaluator import (
    ProductRecallCase,
    ProductRelevanceLabel,
    evaluate_product_recall,
    load_product_recall_cases,
)


def _ranked(item_id: str, rank: int) -> RankedItem:
    """创建最终商品排名测试项。"""

    return RankedItem(
        rank=rank,
        product=Product(item_id=item_id, title=item_id),
        retrieval_score=0.8,
        rerank_score=1.0 / rank,
    )


class FakeItemSearchService:
    """返回固定的二阶段重排结果。"""

    def search(self, request: ItemSearchRequest) -> ItemSearchResponse:
        assert request.index_id == "evaluation-products"
        items = (
            _ranked("noise", 1),
            _ranked("B", 2),
            _ranked("A", 3),
        )[: request.spec.top_k]
        return ItemSearchResponse(
            query=request.spec.normalized_query,
            index_id=request.index_id,
            recall_count=60,
            items=items,
        )


def test_product_dataset_and_seed_products_match_current_project() -> None:
    """所有相关 ID 都存在，旧模型和旧过滤字段均已清除。"""

    cases = load_product_recall_cases(Path("eval/product_recall.jsonl"))
    namespace: dict[str, Any] = runpy.run_path("eval/seed_products.py")
    products = namespace["build_seed_products"]()
    product_ids = {item.item_id for item in products}
    relevant_ids = {
        label.item_id for case in cases for label in case.relevance
    }

    assert len(cases) == 67
    assert len(products) == 60
    assert all(isinstance(item, Product) for item in products)
    assert relevant_ids <= product_ids
    assert {case.index_id for case in cases} == {"evaluation-products"}


def test_product_evaluator_computes_four_ranking_metrics() -> None:
    """工具最终排名用于计算 Recall、Precision、MRR 和分级 NDCG。"""

    case = ProductRecallCase(
        case_id="product-test",
        query="测试商品",
        index_id="evaluation-products",
        kind="semantic",
        relevance=(
            ProductRelevanceLabel("A", 3),
            ProductRelevanceLabel("B", 1),
        ),
        tags=("semantic",),
        note="",
    )

    report = evaluate_product_recall(FakeItemSearchService(), (case,), (1, 3))

    at_one = report["summary"]["at_k"]["1"]
    at_three = report["summary"]["at_k"]["3"]
    assert at_one == {"recall": 0.0, "precision": 0.0, "ndcg": 0.0}
    assert at_three["recall"] == 1.0
    assert at_three["precision"] == pytest.approx(2 / 3)
    assert 0.0 < at_three["ndcg"] < 1.0
    assert report["summary"]["mrr_at_3"] == 0.5


def test_product_evaluator_reports_each_completed_case() -> None:
    """商品评测在每条查询完成后触发一次进度回调。"""

    cases = tuple(
        ProductRecallCase(
            case_id=f"case-{number}",
            query="测试商品",
            index_id="evaluation-products",
            kind="semantic",
            relevance=(ProductRelevanceLabel("A", 3),),
            tags=(),
            note="",
        )
        for number in (1, 2)
    )
    progress: list[tuple[int, int, str]] = []

    evaluate_product_recall(
        FakeItemSearchService(),
        cases,
        (3,),
        progress=lambda current, total, item: progress.append(
            (current, total, item.case_id)
        ),
    )

    assert progress == [(1, 2, "case-1"), (2, 2, "case-2")]
