"""运行商品搜索工具的离线检索评测。"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.composition import build_item_search_service
from app.infrastructure.settings import Settings
from scripts.product_recall_evaluator import (
    EvaluatedProductCase,
    evaluate_product_recall,
    load_product_recall_cases,
    write_product_evaluation_report,
)


def _one_line(text: str, limit: int = 80) -> str:
    """折叠查询中的空白，保证一个用例只打印一行。"""

    normalized = " ".join(text.split())
    return normalized if len(normalized) <= limit else f"{normalized[:limit - 1]}…"


def _print_progress(
    current: int,
    total: int,
    item: EvaluatedProductCase,
    *,
    max_k: int,
) -> None:
    """立即打印单条商品召回指标，避免长任务看起来没有响应。"""

    metrics = item.metrics[max_k]
    print(
        f"[{current:>3}/{total}] case={item.case_id} "
        f'query="{_one_line(item.query)}" retrieved={len(item.retrieved)} '
        f"R@{max_k}={metrics.recall:.4f} "
        f"P@{max_k}={metrics.precision:.4f} "
        f"MRR={item.reciprocal_rank:.4f} NDCG@{max_k}={metrics.ndcg:.4f}",
        flush=True,
    )


def _parse_args() -> argparse.Namespace:
    """解析数据集、K 值和报告路径。"""

    parser = argparse.ArgumentParser(description="评测 ItemSearch 商品排名")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("eval/product_recall.jsonl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("eval/reports/product_recall.json"),
    )
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
    return parser.parse_args()


def main() -> None:
    """通过 Composition Root 使用线上同款 ItemSearchService 运行评测。"""

    args = _parse_args()
    cases = load_product_recall_cases(args.dataset)
    service = build_item_search_service(Settings.from_env())
    max_k = max(args.k)
    print(f"开始商品召回评测：{len(cases)} 条，max_k={max_k}", flush=True)
    report = evaluate_product_recall(
        service,
        cases,
        args.k,
        progress=lambda current, total, item: _print_progress(
            current,
            total,
            item,
            max_k=max_k,
        ),
    )
    write_product_evaluation_report(report, args.report)

    summary = report["summary"]
    print(f"cases={report['dataset']['cases']}")
    print("K\tRecall\tPrecision\tNDCG")
    for k in sorted(summary["at_k"], key=int):
        metrics = summary["at_k"][k]
        print(
            f"{k}\t{metrics['recall']:.4f}\t"
            f"{metrics['precision']:.4f}\t{metrics['ndcg']:.4f}"
        )
    print(f"MRR@{max_k}\t{summary[f'mrr_at_{max_k}']:.4f}")
    print(f"report\t{args.report}")


if __name__ == "__main__":
    main()
