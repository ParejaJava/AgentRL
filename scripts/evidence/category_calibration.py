"""在独立开发集上校准品类拒答阈值，冻结测试集不参与调参。"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from app.application.catalog.category_insight_models import RetrievedCategoryCard
from app.infrastructure.retrieval.category_insight import (
    OpenSearchCategoryCardRepository,
)
from app.infrastructure.settings import Settings
from scripts.category_recall_evaluator import (
    CategoryRecallCase,
    evaluate_category_recall,
    load_category_recall_cases,
)

from .preflight import cached_model_metadata
from .utils import runtime_environment, sha256_file, utc_now

_THRESHOLDS = (
    0.0,
    0.001,
    0.002,
    0.003,
    0.005,
    0.0075,
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.1,
    0.15,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
)


class _ThresholdRetriever:
    """把一次性计算的重排分数投影成不同阈值下的检索结果。"""

    def __init__(
        self,
        raw_results: dict[str, Sequence[RetrievedCategoryCard]],
        threshold: float,
    ) -> None:
        self._raw_results = raw_results
        self._threshold = threshold

    def search(self, category: str, limit: int) -> Sequence[RetrievedCategoryCard]:
        """用最高分做查询级域门控，入域后保留完整重排列表。"""

        raw = self._raw_results[category]
        if not raw or raw[0].score < self._threshold:
            return ()
        return tuple(raw[:limit])


def calibrate_category_threshold(
    root: Path,
    *,
    minimum_recall: float = 0.93,
    minimum_negative_rejection: float = 0.80,
) -> dict[str, Any]:
    """只用开发集选阈值；满足双门槛后优先召回、NDCG 和较低阈值。"""

    from app.composition import build_category_retriever

    dataset = root / "eval/evidence/category_threshold_dev.jsonl"
    cases: tuple[CategoryRecallCase, ...] = load_category_recall_cases(dataset)
    settings = replace(
        Settings.from_env(),
        category_retriever_backend="opensearch",
        category_min_relevance_score=0.0,
    )
    retriever = build_category_retriever(settings)
    if not isinstance(retriever, OpenSearchCategoryCardRepository):
        raise TypeError("品类阈值校准必须使用 OpenSearch 检索器")

    started_at = utc_now()
    started = time.perf_counter()
    raw_results: dict[str, Sequence[RetrievedCategoryCard]] = {}
    for position, case in enumerate(cases, start=1):
        raw_results[case.query] = retriever.search_with_strategy(
            case.query,
            settings.category_hybrid_recall_k,
            "hybrid_rerank",
        )
        print(
            f"[category-threshold {position:>2}/{len(cases)}] {case.case_id}",
            flush=True,
        )
    candidates: list[dict[str, Any]] = []
    for threshold in _THRESHOLDS:
        evaluation = evaluate_category_recall(
            _ThresholdRetriever(raw_results, threshold),
            cases,
            (1, 3, 5, 10),
        )
        summary = evaluation["summary"]
        candidates.append(
            {
                "threshold": threshold,
                "recall_at_10": summary["at_k"]["10"]["recall"],
                "mrr_at_10": summary["mrr_at_10"],
                "ndcg_at_10": summary["at_k"]["10"]["ndcg"],
                "negative_rejection_accuracy": summary[
                    "negative_rejection_accuracy"
                ],
                "failed_case_ids": [
                    item["case_id"]
                    for item in evaluation["cases"]
                    if (
                        item.get("negative_rejected") is False
                        if item.get("is_negative")
                        else float(
                            item.get("metrics", {})
                            .get("10", {})
                            .get("recall", 0.0)
                        )
                        < 1.0
                    )
                ],
            }
        )
    eligible = [
        item
        for item in candidates
        if item["recall_at_10"] >= minimum_recall
        and item["negative_rejection_accuracy"] >= minimum_negative_rejection
    ]
    selected = max(
        eligible or candidates,
        key=lambda item: (
            item["recall_at_10"],
            item["ndcg_at_10"],
            -item["threshold"],
            item["negative_rejection_accuracy"],
        ),
    )
    return {
        "claim_id": "RAG-THRESHOLD-001",
        "capability": "品类域外拒答阈值校准",
        "started_at": started_at,
        "duration_seconds": round(time.perf_counter() - started, 6),
        "command": "uv run python -m scripts.evidence run --suite offline",
        "environment": runtime_environment(root),
        "models": {
            "embedding": {
                "name": settings.category_embedding_model,
                **cached_model_metadata(settings.category_embedding_model),
            },
            "reranker": {
                "name": settings.category_reranker_model,
                **cached_model_metadata(settings.category_reranker_model),
            },
        },
        "retrieval_infrastructure": {
            "image": "opensearchproject/opensearch:3.8.0",
            "index": settings.opensearch_category_index,
            "pipeline": settings.opensearch_category_pipeline,
        },
        "dataset": {
            "path": dataset.relative_to(root).as_posix(),
            "sha256": sha256_file(dataset),
            "cases": len(cases),
        },
        "minimum_recall_at_10": minimum_recall,
        "minimum_negative_rejection_accuracy": minimum_negative_rejection,
        "candidates": candidates,
        "selected": selected,
        "observations": [
            {
                "case_id": case.case_id,
                "is_negative": case.is_negative,
                "top_score": round(
                    max(
                        (
                            item.score
                            for item in raw_results[case.query]
                        ),
                        default=0.0,
                    ),
                    6,
                ),
                "retrieved": [
                    {
                        "source_document": item.card.source_document,
                        "card_type": item.card.card_type.value,
                        "score": item.score,
                    }
                    for item in raw_results[case.query]
                ],
            }
            for case in cases
        ],
        "status": "verified" if eligible else "code_verified",
        "passed": bool(eligible),
        "limitations": [
            "阈值只在独立开发集上选择；冻结测试集仅用于一次最终评估。",
        ],
    }
