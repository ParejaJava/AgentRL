"""运行商品搜索工具的离线检索评测。"""

from __future__ import annotations

import argparse
from pathlib import Path

from product_recall_evaluator import (
    evaluate_product_recall,
    load_product_recall_cases,
    write_product_evaluation_report,
)

from app.composition import build_item_search_service
from app.infrastructure.settings import Settings


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
    report = evaluate_product_recall(service, cases, args.k)
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
    max_k = max(args.k)
    print(f"MRR@{max_k}\t{summary[f'mrr_at_{max_k}']:.4f}")
    print(f"report\t{args.report}")


if __name__ == "__main__":
    main()
