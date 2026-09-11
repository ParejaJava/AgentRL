"""Validate one or more JSONL splits before training or evaluation."""

import argparse
import json
from pathlib import Path

from training.data import read_jsonl, validate_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument(
        "--tokenizer", help="Optional local tokenizer for supervision checks"
    )
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_length < 1:
        parser.error("--max-length must be positive")
    rows = [row for path in args.paths for row in read_jsonl(path)]
    report = validate_splits(rows)
    if args.tokenizer:
        from transformers import AutoTokenizer

        from training.modeling import encode_decision

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        encoded = [encode_decision(tokenizer, row, args.max_length) for row in rows]
        report["tokenization"] = {
            "tokenizer": args.tokenizer,
            "max_length": args.max_length,
            "cases": [
                {
                    "case_id": row["case_id"],
                    "prompt_tokens": item["prompt_tokens"],
                    "target_tokens": item["target_tokens"],
                    "sequence_tokens": len(item["input_ids"]),
                }
                for row, item in zip(rows, encoded, strict=True)
            ],
            "prefix_and_target_mask_checks_passed": True,
        }
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
