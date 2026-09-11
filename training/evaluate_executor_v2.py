"""Evaluate base or adapter on frozen decisions with per-case raw evidence."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

from training.data import digest, read_jsonl, validate_splits
from training.modeling import load_model, parse_output, render_prompt
from training.provenance import file_hash, model_manifest


def score(row: dict, answer: dict) -> dict:
    from app.application.catalog.recommendations import validate_selection
    from training.evaluate_executor import score as base_score

    result = base_score(row, answer)
    rubric = row["rubric"]
    if "selection_ids" in rubric:
        ids, errors = validate_selection(
            answer.get("content", ""), rubric["products"], rubric["top_k"]
        )
        valid = (
            not errors
            and ids == rubric["selection_ids"]
            and not answer.get("tool_calls")
        )
        result.update(selection_ok=valid, selection_errors=errors)
        result["passed"] = result["passed"] and valid
    return result


def wilson(passed: int, total: int) -> list[float]:
    if not total:
        return [0, 1]
    z = 1.96
    p = passed / total
    divisor = 1 + z * z / total
    center = (p + z * z / (2 * total)) / divisor
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / divisor
    return [center - radius, center + radius]


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/models/Qwen3-1.7B")
    parser.add_argument("--adapter")
    parser.add_argument("--cases", default="eval/executor_v2/test.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--context-window", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new evaluation output directory")
    rows = read_jsonl(args.cases)
    validate_splits(rows)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No evaluation cases")
    torch.set_num_threads(4)
    model, tokenizer = load_model(args.model, args.adapter)
    args.output.mkdir(parents=True)
    metadata = {
        "model_manifest": model_manifest(args.model),
        "adapter_manifest": model_manifest(args.adapter) if args.adapter else None,
        "evaluation_source_sha256": file_hash(Path(__file__)),
        "shared_scoring_source_sha256": file_hash(
            Path("training/evaluate_executor.py")
        ),
        "selection_source_sha256": file_hash(
            Path("app/application/catalog/recommendations.py")
        ),
        "modeling_source_sha256": file_hash(Path("training/modeling.py")),
        "model": args.model,
        "adapter": args.adapter,
        "dataset_sha256": digest(rows),
        "decoding": "greedy",
        "max_new_tokens": args.max_new_tokens,
        "context_window": args.context_window,
        "fallback_enabled": False,
        "cache_enabled": False,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "limitations": "Synthetic single-decision benchmark with schema/argument/keyword rubrics. Not an end-to-end business success or semantic correctness proof.",
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    warm = tokenizer("你好", return_tensors="pt").to(model.device)
    with torch.inference_mode():
        model.generate(
            **warm,
            max_new_tokens=2,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    results = []
    for row in rows:
        prompt = render_prompt(tokenizer, row["messages"], row["tools"])
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
            model.device
        )
        n = inputs["input_ids"].shape[1]
        if n + args.max_new_tokens > args.context_window:
            raise ValueError("Evaluation context exceeds configured window")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        generated = outputs[0, n:].tolist()
        raw = tokenizer.decode(generated, skip_special_tokens=True)
        error = None
        truncated = bool(
            generated
            and generated[-1] != tokenizer.eos_token_id
            and len(generated) >= args.max_new_tokens
        )
        try:
            answer = parse_output(raw)
            scores = score(row, answer)
            if truncated:
                scores["passed"] = False
        except (ValueError, TypeError, KeyError) as exc:
            answer = None
            error = str(exc)
            scores = {
                "passed": False,
                "schema_ok": False,
                "parameters_ok": False,
                "tool_choice_ok": False,
                "terms_ok": False,
                "forbidden_ok": False,
            }
        entry = {
            "case_id": row["case_id"],
            "family": row["family"],
            **scores,
            "raw_output": raw,
            "answer": answer,
            "parse_error": error,
            "truncated": truncated,
            "input_tokens": n,
            "output_tokens": len(generated),
            "duration_seconds": elapsed,
        }
        results.append(entry)
        with (args.output / "cases.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(
            json.dumps(
                {
                    "case": entry["case_id"],
                    "passed": entry["passed"],
                    "seconds": round(elapsed, 2),
                }
            ),
            flush=True,
        )
    by_family = defaultdict(list)
    for result in results:
        by_family[result["family"]].append(result)
    passed = sum(r["passed"] for r in results)
    durations = sorted(r["duration_seconds"] for r in results)
    tool_results = [
        r
        for r, row in zip(results, rows, strict=True)
        if row["target"].get("tool_calls")
    ]
    report = {
        "total": len(results),
        "passed": passed,
        "success_rate": passed / len(results),
        "success_wilson_95": wilson(passed, len(results)),
        "tool_target_cases": len(tool_results),
        "tool_schema_rate": sum(
            r["schema_ok"] and r["tool_choice_ok"] for r in tool_results
        )
        / max(1, len(tool_results)),
        "tool_parameter_rate": sum(r["parameters_ok"] for r in tool_results)
        / max(1, len(tool_results)),
        "p50_seconds": statistics.median(durations),
        "p95_seconds": durations[math.ceil(len(durations) * 0.95) - 1],
        "total_generation_seconds": sum(durations),
        "generation_seconds_per_success": sum(durations) / passed if passed else None,
        "families": {
            f: {"passed": sum(x["passed"] for x in rs), "total": len(rs)}
            for f, rs in by_family.items()
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
