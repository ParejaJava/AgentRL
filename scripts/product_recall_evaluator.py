"""商品搜索工具最终返回排名的离线评测。"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.application.catalog.models import ItemSearchCommand
from app.application.catalog.search_catalog import ItemSearchService
from app.domain.catalog import ProductSearchSpec

ProductProgressCallback = Callable[[int, int, "EvaluatedProductCase"], None]


@dataclass(frozen=True, slots=True)
class ProductRelevanceLabel:
    """一个商品相关性标注，grade=3/2/1 分别表示核心/补充/弱相关。"""

    item_id: str
    grade: int


@dataclass(frozen=True, slots=True)
class ProductRecallCase:
    """一条商品检索评测样本。"""

    case_id: str
    query: str
    index_id: str
    kind: str
    relevance: tuple[ProductRelevanceLabel, ...]
    tags: tuple[str, ...]
    note: str
    buyer_id: str | None = None


@dataclass(frozen=True, slots=True)
class RankingAtK:
    """单条商品查询在指定 K 下的四项指标中的前三项。"""

    recall: float
    precision: float
    ndcg: float


@dataclass(frozen=True, slots=True)
class EvaluatedProductCase:
    """单条商品查询的完整排名和评测结果。"""

    case_id: str
    query: str
    index_id: str
    kind: str
    tags: tuple[str, ...]
    reciprocal_rank: float
    metrics: dict[int, RankingAtK]
    expected: tuple[str, ...]
    retrieved: tuple[str, ...]
    latency_ms: float = 0.0


def load_product_recall_cases(path: Path) -> tuple[ProductRecallCase, ...]:
    """加载并严格校验商品评测 JSONL。"""

    cases: list[ProductRecallCase] = []
    seen_case_ids: set[str] = set()
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
            raise ValueError(f"无效商品评测数据：{path}:{line_number}: {exc}") from exc
        if case.case_id in seen_case_ids:
            raise ValueError(f"重复 case_id：{case.case_id}")
        seen_case_ids.add(case.case_id)
        cases.append(case)
    if not cases:
        raise ValueError(f"商品评测数据为空：{path}")
    return tuple(cases)


def evaluate_product_recall(
    service: ItemSearchService,
    cases: Sequence[ProductRecallCase],
    ks: Sequence[int],
    *,
    progress: ProductProgressCallback | None = None,
) -> dict[str, Any]:
    """逐条评测完整 ItemSearch，并可在每条完成后回调输出进度。"""

    normalized_ks = tuple(sorted(set(ks)))
    if not normalized_ks or normalized_ks[0] < 1:
        raise ValueError("K 必须至少包含一个正整数")
    max_k = normalized_ks[-1]
    evaluated: list[EvaluatedProductCase] = []
    total = len(cases)
    for current, case in enumerate(cases, start=1):
        started = time.perf_counter()
        response = service.search(
            ItemSearchCommand(
                spec=ProductSearchSpec(
                    normalized_query=case.query,
                    top_k=max_k,
                ),
                index_id=case.index_id,
                buyer_id=case.buyer_id,
            )
        )
        ranked_ids = tuple(item.product.item_id for item in response.items)
        grades = {item.item_id: item.grade for item in case.relevance}
        item = EvaluatedProductCase(
            case_id=case.case_id,
            query=case.query,
            index_id=case.index_id,
            kind=case.kind,
            tags=case.tags,
            reciprocal_rank=_reciprocal_rank(ranked_ids, grades),
            metrics={
                k: _ranking_at_k(ranked_ids, grades, k)
                for k in normalized_ks
            },
            expected=tuple(grades),
            retrieved=ranked_ids,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        evaluated.append(item)
        if progress is not None:
            progress(current, total, item)

    report = {
        "dataset": {
            "cases": len(evaluated),
            "indexes": sorted({item.index_id for item in evaluated}),
        },
        "summary": {
            "at_k": {
                str(k): {
                    "recall": _mean(item.metrics[k].recall for item in evaluated),
                    "precision": _mean(
                        item.metrics[k].precision for item in evaluated
                    ),
                    "ndcg": _mean(item.metrics[k].ndcg for item in evaluated),
                }
                for k in normalized_ks
            },
            f"mrr_at_{max_k}": _mean(
                item.reciprocal_rank for item in evaluated
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


def write_product_evaluation_report(report: dict[str, Any], path: Path) -> None:
    """原子写入可供 CI 比较的商品评测报告。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_case(raw: dict[str, Any]) -> ProductRecallCase:
    """把单行 JSON 转换为强类型样本，并拒绝旧版不兼容字段。"""

    unsupported = {"price_max_major", "ship_to"} & set(raw)
    if unsupported:
        raise ValueError(
            f"当前 item_search 不支持独立过滤字段：{sorted(unsupported)}"
        )
    labels = tuple(
        ProductRelevanceLabel(
            item_id=str(item["item_id"]),
            grade=int(item["grade"]),
        )
        for item in raw["relevance"]
    )
    if not labels:
        raise ValueError("商品召回用例至少需要一个相关商品")
    if any(label.grade not in {1, 2, 3} for label in labels):
        raise ValueError("relevance.grade 只能是 1、2 或 3")
    item_ids = [label.item_id for label in labels]
    if any(not item_id.strip() for item_id in item_ids):
        raise ValueError("relevance.item_id 不能为空")
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("同一用例不能重复标注 item_id")
    case_id = str(raw["case_id"]).strip()
    query = str(raw["query"]).strip()
    index_id = str(raw["index_id"]).strip()
    kind = str(raw["kind"]).strip()
    if not all((case_id, query, index_id, kind)):
        raise ValueError("case_id/query/index_id/kind 不能为空")
    return ProductRecallCase(
        case_id=case_id,
        query=query,
        index_id=index_id,
        kind=kind,
        relevance=labels,
        tags=tuple(str(tag) for tag in raw.get("tags", [])),
        note=str(raw.get("note", "")),
        buyer_id=(
            str(raw["buyer_id"]).strip() if raw.get("buyer_id") else None
        ),
    )


def _ranking_at_k(
    ranked_ids: Sequence[str],
    grades: dict[str, int],
    k: int,
) -> RankingAtK:
    """计算商品级 Recall@K、Precision@K 与分级 NDCG@K。"""

    top_k = ranked_ids[:k]
    relevant_hits = sum(item_id in grades for item_id in top_k)
    recall = relevant_hits / len(grades)
    precision = relevant_hits / k
    dcg = sum(
        (2 ** grades.get(item_id, 0) - 1) / math.log2(rank + 1)
        for rank, item_id in enumerate(top_k, start=1)
    )
    ideal = sorted(grades.values(), reverse=True)[:k]
    ideal_dcg = sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal, start=1)
    )
    return RankingAtK(recall, precision, dcg / ideal_dcg if ideal_dcg else 0.0)


def _reciprocal_rank(
    ranked_ids: Sequence[str],
    grades: dict[str, int],
) -> float:
    """返回首个相关商品排名的倒数。"""

    for rank, item_id in enumerate(ranked_ids, start=1):
        if item_id in grades:
            return 1.0 / rank
    return 0.0


def _aggregate_by_tag(
    evaluated: Sequence[EvaluatedProductCase],
    k: int,
) -> dict[str, dict[str, float | int]]:
    """按 lexical/semantic/constraint 等标签定位质量退化。"""

    groups: dict[str, list[EvaluatedProductCase]] = defaultdict(list)
    for item in evaluated:
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


def _case_to_dict(item: EvaluatedProductCase) -> dict[str, Any]:
    """序列化逐条结果，用于人工检查错误排名。"""

    return {
        "case_id": item.case_id,
        "query": item.query,
        "index_id": item.index_id,
        "kind": item.kind,
        "tags": list(item.tags),
        "reciprocal_rank": item.reciprocal_rank,
        "latency_ms": item.latency_ms,
        "metrics": {
            str(k): asdict(value) for k, value in item.metrics.items()
        },
        "expected": list(item.expected),
        "retrieved": list(item.retrieved),
    }


def _mean(values: Iterable[float]) -> float:
    """计算稳定到六位小数的算术平均值。"""

    materialized = tuple(values)
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
