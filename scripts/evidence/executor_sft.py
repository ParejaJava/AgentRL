"""Build a comparable, artifact-backed executor SFT evidence report.

Usage: python -m scripts.evidence.executor_sft --base DIR --adapter DIR --output FILE
"""

import argparse
import json
from pathlib import Path


def compare(base: Path, adapter: Path) -> dict:
    def load(root, name):
        return json.loads((root / name).read_text(encoding="utf-8"))

    left, right = load(base, "metadata.json"), load(adapter, "metadata.json")
    for key in (
        "dataset_sha256",
        "decoding",
        "max_new_tokens",
        "context_window",
        "evaluation_source_sha256",
        "modeling_source_sha256",
        "model_manifest",
    ):
        if left.get(key) is None or right.get(key) is None:
            raise ValueError(f"Missing experiment provenance: {key}")
        if left.get(key) != right.get(key):
            raise ValueError(f"Non-comparable experiment: {key} differs")
    if left.get("adapter") or not right.get("adapter_manifest"):
        raise ValueError("Expected an unadapted base and a versioned adapter")
    a, b = load(base, "summary.json"), load(adapter, "summary.json")
    base_cases = [
        json.loads(s)
        for s in (base / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    adapted_cases = [
        json.loads(s)
        for s in (adapter / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if [x["case_id"] for x in base_cases] != [x["case_id"] for x in adapted_cases]:
        raise ValueError("Case identities/order differ")
    if len(base_cases) != a["total"] or len(adapted_cases) != b["total"]:
        raise ValueError("Incomplete per-case evidence")
    for summary, cases in ((a, base_cases), (b, adapted_cases)):
        actual_passed = sum(bool(c["passed"]) for c in cases)
        if (
            not cases
            or summary["passed"] != actual_passed
            or abs(summary["success_rate"] - actual_passed / len(cases)) > 1e-9
        ):
            raise ValueError("Summary disagrees with per-case evidence")
    if any(
        m.get("fallback_enabled") is not False or m.get("cache_enabled") is not False
        for m in (left, right)
    ):
        raise ValueError(
            "Comparative model evaluation must disable fallback and response cache"
        )
    return {
        "claim_id": "EXECUTOR-SFT-001",
        "status": "experiment_completed",
        "base_directory": str(base),
        "adapter_directory": str(adapter),
        "dataset_sha256": left["dataset_sha256"],
        "base": a,
        "adapter": b,
        "success_delta_percentage_points": 100
        * (b["success_rate"] - a["success_rate"]),
        "paired_improved": sum(
            not x["passed"] and y["passed"]
            for x, y in zip(base_cases, adapted_cases, strict=True)
        ),
        "paired_regressed": sum(
            x["passed"] and not y["passed"]
            for x, y in zip(base_cases, adapted_cases, strict=True)
        ),
        "p95_relative_change": b["p95_seconds"] / a["p95_seconds"] - 1,
        "limitations": [
            "Small authored synthetic single-decision benchmark, not production traffic.",
            "No paid teacher baseline; no claim of matching the existing production model.",
            "Local wall-clock latency is not a measured monetary cost saving.",
            "Keyword rubrics need semantic review; report interval estimates alongside point results.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.base, args.adapter)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
