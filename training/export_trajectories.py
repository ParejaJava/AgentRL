"""Export explicitly reviewed successful attempts; never auto-label success as correct.

Review JSON is a list containing attempt_id, approved=true, privacy_reviewed=true,
split, group_id, and replacements (sensitive literal -> stable placeholder).
Optional corrected_target replaces an incorrect model answer after human review.
"""

import argparse
import json
from pathlib import Path

from training.data import digest, validate_splits, write_jsonl


def export(root: Path, reviews: list[dict]) -> list[dict]:
    rows = []
    for review in reviews:
        if not review.get("approved"):
            continue
        if review.get("privacy_reviewed") is not True:
            raise ValueError("Approved capture has no privacy review")
        ident = review["attempt_id"]
        if len(ident) != 32 or any(c not in "0123456789abcdef" for c in ident):
            raise ValueError("Invalid attempt ID")
        record = json.loads((root / f"{ident}.json").read_text(encoding="utf-8"))
        if record["status"] != "success" or record["agent_role"] != "executor":
            raise ValueError("Only completed executor attempts are eligible")
        if not record.get("thread_id"):
            raise ValueError(
                "Captured session identity is required for split isolation"
            )
        responses = record["response"]
        if len(responses) != 1:
            raise ValueError("Review requires a single assistant response")
        payload = {
            "messages": record["messages"],
            "tools": record["tools"],
            "target": review.get("corrected_target", responses[0]),
        }

        def redact(value, review=review):
            if isinstance(value, str):
                for old, new in review.get("replacements", {}).items():
                    if not old:
                        raise ValueError("Empty redaction literal")
                    value = value.replace(old, new)
                return value
            if isinstance(value, list):
                return [redact(v) for v in value]
            if isinstance(value, dict):
                return {k: redact(v) for k, v in value.items()}
            return value

        rows.append(
            {
                "case_id": ident,
                "group_id": review["group_id"],
                "split": review["split"],
                "source": "reviewed_runtime",
                "privacy_reviewed": True,
                "capture_model": record["model"],
                "capture_session_hash": record.get("session_group_hash")
                or digest(record["thread_id"]),
                "capture_prompt_hash": record["prompt_hash"],
                "capture_response_hash": digest(responses[0]),
                "review_sha256": digest(review),
                "target_origin": (
                    "review_correction"
                    if "corrected_target" in review
                    else "captured_response"
                ),
                **redact(payload),
            }
        )
    validate_splits(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = export(args.root, json.loads(args.reviews.read_text(encoding="utf-8")))
    write_jsonl(args.output, rows)
    print(f"Exported {len(rows)} explicitly reviewed decisions")


if __name__ == "__main__":
    main()
