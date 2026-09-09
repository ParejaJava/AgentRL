"""CategoryInsight 卡片检索的离线排序评测。"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from app.application.catalog.category_insight_models import RetrievedCategoryCard
from app.application.catalog.category_insight_ports import CategoryKnowledgeRetriever

EvaluationKey = tuple[str, str]
CategoryProgressCallback = Callable[[int, int, "EvaluatedCase"], None]


@dataclass(frozen=True, slots=True)
class RelevanceLabel:
    """一个稳定的相关单元；不依赖可能随重新摄取变化的 card_id。"""

    source_document: str
    card_type: str
    topic: str
    grade: int

    @property
    def key(self) -> EvaluationKey:
        return (self.source_document, self.card_type)


@dataclass(frozen=True, slots=True)
class CategoryRecallCase:
    """一条检索评测样本。relevance 为空表示知识库应拒答。"""

    case_id: str
    query: str
    expected_category: str | None
    relevance: tuple[RelevanceLabel, ...]
    tags: tuple[str, ...]

    @property
    def is_negative(self) -> bool:
        return not self.relevance


@dataclass(frozen=True, slots=True)
class RankingAtK:
    """单条正例在指定 K 下的排序指标。"""

    recall: float
    precision: float
    ndcg: float


@dataclass(frozen=True, slots=True)
class EvaluatedCase:
    """单条用例的命中详情与指标。"""

    case_id: str
    query: str
    tags: tuple[str, ...]
    is_negative: bool
    reciprocal_rank: float
    negative_rejected: bool | None
    metrics: dict[int, RankingAtK]
    expected: tuple[EvaluationKey, ...]
    retrieved: tuple[EvaluationKey, ...]
    latency_ms: float = 0.0


def load_category_recall_cases(path: Path) -> tuple[CategoryRecallCase, ...]:
    """加载并严格校验 JSONL，尽早发现重复 ID 或无效相关性等级。"""

    cases: list[CategoryRecallCase] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            case = _parse_case(raw)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"无效评测数据：{path}:{line_number}: {exc}") from exc
        if case.case_id in seen_ids:
            raise ValueError(f"重复 case_id：{case.case_id}")
        seen_ids.add(case.case_id)
        cases.append(case)
    if not cases:
        raise ValueError(f"评测数据为空：{path}")
    return tuple(cases)


def evaluate_category_recall(
    retriever: CategoryKnowledgeRetriever,
    cases: Sequence[CategoryRecallCase],
    ks: Sequence[int],
    *,
    progress: CategoryProgressCallback | None = None,
) -> dict[str, Any]:
    """逐条运行检索，并可在每条完成后回调输出进度。"""

    normalized_ks = tuple(sorted(set(ks)))
    if not normalized_ks or normalized_ks[0] < 1:
        raise ValueError("K 必须至少包含一个正整数")
    max_k = normalized_ks[-1]
    evaluated: list[EvaluatedCase] = []
    total = len(cases)
    for current, case in enumerate(cases, start=1):
        started = time.perf_counter()
        retrieved = retriever.search(case.query, max_k)
        latency_ms = (time.perf_counter() - started) * 1000
        item = _evaluate_case(
            case,
            retrieved,
            normalized_ks,
        )
        item = replace(item, latency_ms=round(latency_ms, 3))
        evaluated.append(item)
        if progress is not None:
            progress(current, total, item)
    positives = [item for item in evaluated if not item.is_negative]
    negatives = [item for item in evaluated if item.is_negative]
    if not positives:
        raise ValueError("评测集至少需要一条正例")

    summary = {
        str(k): {
            "recall": _mean(item.metrics[k].recall for item in positives),
            "precision": _mean(item.metrics[k].precision for item in positives),
            "ndcg": _mean(item.metrics[k].ndcg for item in positives),
        }
        for k in normalized_ks
    }
    report = {
        "dataset": {
            "cases": len(evaluated),
            "positive_cases": len(positives),
            "negative_cases": len(negatives),
        },
        "summary": {
            "at_k": summary,
            f"mrr_at_{max_k}": _mean(
                item.reciprocal_rank for item in positives
            ),
            "negative_rejection_accuracy": (
                _mean(float(item.negative_rejected) for item in negatives)
                if negatives
                else None
            ),
            "latency_ms": {
                "p50": _percentile([item.latency_ms for item in evaluated], 50),
                "p95": _percentile([item.latency_ms for item in evaluated], 95),
            },
        },
        "by_tag": _aggregate_by_tag(evaluated, max_k),
        "cases": [_case_to_dict(item) for item in evaluated],
    }
    return report


def write_evaluation_report(report: dict[str, Any], path: Path) -> None:
    """把完整结果原子写入 JSON，便于 CI 保存和版本间比较。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_case(raw: dict[str, Any]) -> CategoryRecallCase:
    """把一行 JSON 转成强类型样本。"""

    case_id = str(raw["case_id"]).strip()
    query = str(raw["query"]).strip()
    if not case_id or not query:
        raise ValueError("case_id 和 query 不能为空")
    relevance = tuple(
        RelevanceLabel(
            source_document=str(item["source_document"]),
            card_type=str(item["card_type"]),
            topic=str(item["topic"]),
            grade=int(item["grade"]),
        )
        for item in raw.get("relevance", [])
    )
    allowed_card_types = {"bestseller", "attribute", "price_range"}
    for label in relevance:
        if label.card_type not in allowed_card_types:
            raise ValueError(f"未知 card_type：{label.card_type}")
        if label.grade not in {1, 2, 3}:
            raise ValueError("relevance.grade 只能是 1、2 或 3")
    keys = [label.key for label in relevance]
    if len(keys) != len(set(keys)):
        raise ValueError("同一用例不能重复标注 source_document + card_type")
    expected_category = raw.get("expected_category")
    return CategoryRecallCase(
        case_id=case_id,
        query=query,
        expected_category=(
            str(expected_category) if expected_category is not None else None
        ),
        relevance=relevance,
        tags=tuple(str(tag) for tag in raw.get("tags", [])),
    )


def _evaluate_case(
    case: CategoryRecallCase,
    retrieved: Sequence[RetrievedCategoryCard],
    ks: Sequence[int],
) -> EvaluatedCase:
    """按稳定评测键折叠重复卡片并计算一条样本。"""

    ranked_keys = _collapse_retrieved(retrieved)
    grades = {label.key: label.grade for label in case.relevance}
    reciprocal_rank = 0.0
    for rank, key in enumerate(ranked_keys, start=1):
        if key in grades:
            reciprocal_rank = 1.0 / rank
            break
    metrics = {
        k: _ranking_at_k(ranked_keys, grades, k)
        for k in ks
    }
    return EvaluatedCase(
        case_id=case.case_id,
        query=case.query,
        tags=case.tags,
        is_negative=case.is_negative,
        reciprocal_rank=reciprocal_rank,
        negative_rejected=(not ranked_keys if case.is_negative else None),
        metrics=metrics,
        expected=tuple(grades),
        retrieved=ranked_keys,
    )


def _collapse_retrieved(
    retrieved: Sequence[RetrievedCategoryCard],
) -> tuple[EvaluationKey, ...]:
    """同一来源同一类型只保留排名最高的卡，防止重复命中虚增指标。"""

    values: list[EvaluationKey] = []
    seen: set[EvaluationKey] = set()
    for item in retrieved:
        key = (item.card.source_document, item.card.card_type.value)
        if key not in seen:
            seen.add(key)
            values.append(key)
    return tuple(values)


def _ranking_at_k(
    ranked_keys: Sequence[EvaluationKey],
    grades: dict[EvaluationKey, int],
    k: int,
) -> RankingAtK:
    """计算标准二元 Recall/Precision 和分级 NDCG。"""

    if not grades:
        return RankingAtK(recall=0.0, precision=0.0, ndcg=0.0)
    top_k = ranked_keys[:k]
    relevant_hits = sum(key in grades for key in top_k)
    recall = relevant_hits / len(grades)
    precision = relevant_hits / k
    dcg = sum(
        (2 ** grades.get(key, 0) - 1) / math.log2(rank + 1)
        for rank, key in enumerate(top_k, start=1)
    )
    ideal_grades = sorted(grades.values(), reverse=True)[:k]
    ideal_dcg = sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal_grades, start=1)
    )
    return RankingAtK(
        recall=recall,
        precision=precision,
        ndcg=dcg / ideal_dcg if ideal_dcg else 0.0,
    )


def _aggregate_by_tag(
    evaluated: Sequence[EvaluatedCase],
    k: int,
) -> dict[str, dict[str, float | int]]:
    """按数据标签输出最大 K 的切片指标，快速定位退化场景。"""

    groups: dict[str, list[EvaluatedCase]] = defaultdict(list)
    for item in evaluated:
        if not item.is_negative:
            for tag in item.tags:
                groups[tag].append(item)
    return {
        tag: {
            "cases": len(items),
            "recall": _mean(item.metrics[k].recall for item in items),
            "precision": _mean(item.metrics[k].precision for item in items),
            "ndcg": _mean(item.metrics[k].ndcg for item in items),
            "mrr": _mean(item.reciprocal_rank for item in items),
        }
        for tag, items in sorted(groups.items())
    }


def _case_to_dict(item: EvaluatedCase) -> dict[str, Any]:
    """把元组键转换成 JSON 友好的对象。"""

    return {
        "case_id": item.case_id,
        "query": item.query,
        "tags": list(item.tags),
        "is_negative": item.is_negative,
        "reciprocal_rank": item.reciprocal_rank,
        "negative_rejected": item.negative_rejected,
        "latency_ms": item.latency_ms,
        "metrics": {
            str(k): asdict(value) for k, value in item.metrics.items()
        },
        "expected": [
            {"source_document": source, "card_type": card_type}
            for source, card_type in item.expected
        ],
        "retrieved": [
            {"source_document": source, "card_type": card_type}
            for source, card_type in item.retrieved
        ],
    }


def _mean(values: Iterable[float]) -> float:
    """计算稳定到六位小数的算术平均值。"""

    materialized = tuple(values)
    if not materialized:
        return 0.0
    return round(sum(materialized) / len(materialized), 6)


def _percentile(values: Sequence[float], percentile: int) -> float:
    """用线性插值计算小样本延迟分位数。"""

    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 3)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 3)
