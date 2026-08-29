"""运行 CategoryInsight 离线检索评测并生成 JSON 报告。"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from category_recall_evaluator import (
    evaluate_category_recall,
    load_category_recall_cases,
    write_evaluation_report,
)

from app.composition import build_category_retriever
from app.infrastructure.settings import Settings


def _parse_args() -> argparse.Namespace:
    """解析数据集、后端、K 值和报告路径。"""

    parser = argparse.ArgumentParser(description="评测 CategoryInsight 卡片召回")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("eval/category_recall.jsonl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("eval/reports/category_recall.json"),
    )
    parser.add_argument(
        "--backend",
        choices=("local", "opensearch"),
        help="覆盖 CATEGORY_RETRIEVER_BACKEND",
    )
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
    return parser.parse_args()


def main() -> None:
    """通过 Composition Root 使用与线上相同的 retriever 运行评测。"""

    args = _parse_args()
    settings = Settings.from_env()
    if args.backend is not None:
        settings = replace(settings, category_retriever_backend=args.backend)
    cases = load_category_recall_cases(args.dataset)
    retriever = build_category_retriever(settings)
    report = evaluate_category_recall(retriever, cases, args.k)
    write_evaluation_report(report, args.report)

    dataset = report["dataset"]
    summary = report["summary"]
    print(
        f"cases={dataset['cases']} positive={dataset['positive_cases']} "
        f"negative={dataset['negative_cases']}"
    )
    print("K\tRecall\tPrecision\tNDCG")
    for k in sorted(summary["at_k"], key=int):
        metrics = summary["at_k"][k]
        print(
            f"{k}\t{metrics['recall']:.4f}\t"
            f"{metrics['precision']:.4f}\t{metrics['ndcg']:.4f}"
        )
    max_k = max(args.k)
    print(f"MRR@{max_k}\t{summary[f'mrr_at_{max_k}']:.4f}")
    rejection = summary["negative_rejection_accuracy"]
    if rejection is not None:
        print(f"Negative rejection\t{rejection:.4f}")
    print(f"report\t{args.report}")


if __name__ == "__main__":
    main()
