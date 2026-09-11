"""Evaluate base or adapter on frozen decisions with per-case raw evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from training.data import arguments, digest, read_jsonl, validate_splits
from training.modeling import load_model, parse_output, render_prompt
from training.provenance import file_hash, model_manifest


def score(row: dict, answer: dict) -> dict:
    expected = row["target"].get("tool_calls", [])
    actual = answer.get("tool_calls", [])
    names_ok = [c["function"]["name"] for c in actual] == [
        c["function"]["name"] for c in expected
    ]
    schema_ok = True
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in row["tools"]}
    parsed = []
    for call in actual:
        try:
            args = arguments(call)
            if call["function"]["name"] == "item_search":
                # Mirror production ProductSearchToolInput's code normalization
                # and whitespace stripping before comparing hard constraints.
                args = {
                    k: v.strip() if isinstance(v, str) else v for k, v in args.items()
                }
                for key in ("ship_to", "target_currency"):
                    if isinstance(args.get(key), str):
                        args[key] = args[key].upper()
            Draft202012Validator(schemas[call["function"]["name"]]).validate(args)
            parsed.append(args)
        except (ValueError, TypeError, KeyError, ValidationError):
            schema_ok = False
    rubric = row["rubric"]
    params_ok = names_ok and schema_ok
    if expected:
        params_ok = (
            params_ok
            and bool(parsed)
            and all(
                parsed[0].get(
                    key,
                    schemas[actual[0]["function"]["name"]]
                    .get("properties", {})
                    .get(key, {})
                    .get("default"),
                )
                == value
                for key, value in rubric["expected_args"].items()
            )
        )
        if params_ok and expected[0]["function"]["name"] == "item_search":
            category = arguments(expected[0]).get("category")
            if category:
                params_ok = (
                    category.casefold()
                    in parsed[0].get("normalized_query", "").casefold()
                )
    text = str(answer.get("content", ""))
    if expected:
        text += json.dumps(parsed, ensure_ascii=False)

    def matches(term: str) -> bool:
        aliases = {
            "EUR": ("EUR", "欧元"),
            "USD": ("USD", "美元"),
            "GBP": ("GBP", "英镑"),
            "JPY": ("JPY", "日元"),
            "CNY": ("CNY", "人民币"),
        }
        if term in aliases:
            return any(alias in text for alias in aliases[term])
        if term.isdigit():
            numbers = re.findall(
                r"(?<![\d.])\d+(?:\.\d+)?(?![\d.])", text.replace(",", "")
            )
            return any(float(n) == float(term) for n in numbers)
        return term in text

    terms_ok = all(
        any(matches(term) for term in choices) for choices in rubric["required_any"]
    )
    forbidden_ok = not any(term in text for term in rubric["forbidden"])
    nonempty = bool(actual) or bool(str(answer.get("content", "")).strip())
    return {
        "passed": names_ok
        and schema_ok
        and params_ok
        and terms_ok
        and forbidden_ok
        and nonempty,
        "tool_choice_ok": names_ok,
        "schema_ok": schema_ok,
        "parameters_ok": params_ok,
        "terms_ok": terms_ok,
        "forbidden_ok": forbidden_ok,
    }


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
    parser.add_argument("--model", default="data/models/Qwen3-0.6B")
    parser.add_argument("--adapter")
    parser.add_argument("--cases", default="eval/executor/test.jsonl")
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
