"""Framework-independent decision dataset validation and hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def arguments(call: dict[str, Any]) -> dict[str, Any]:
    value = call["function"]["arguments"]
    value = json.loads(value) if isinstance(value, str) else value
    if not isinstance(value, dict):
        raise TypeError("Tool arguments must be a JSON object")
    json.dumps(value, allow_nan=False)
    return value


def validate_decision(row: dict[str, Any]) -> None:
    """Reject malformed, unreviewed, future-leaking or unpaired supervision."""
    for key in (
        "case_id",
        "group_id",
        "split",
        "source",
        "messages",
        "tools",
        "target",
    ):
        if key not in row:
            raise ValueError(f"Missing {key}")
    if row["split"] not in {"train", "dev", "test"}:
        raise ValueError("Unknown split")
    if row["source"] not in {"authored_synthetic", "reviewed_runtime"}:
        raise ValueError("Unreviewed source")
    if row["source"] == "reviewed_runtime" and not row.get("privacy_reviewed"):
        raise ValueError("Runtime samples require explicit privacy review")
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in row["tools"]}
    if len(schemas) != len(row["tools"]):
        raise ValueError("Duplicate tools")
    pending: set[str] = set()
    seen: set[str] = set()
    for message in row["messages"]:
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError("Unsupported message role")
        if role == "tool":
            ident = message.get("tool_call_id")
            if ident not in pending:
                raise ValueError("Orphan or duplicate tool result")
            pending.remove(ident)
        else:
            if pending:
                raise ValueError("Tool results missing before next message")
            for call in message.get("tool_calls", []):
                ident = call.get("id")
                if role != "assistant" or not ident or ident in seen:
                    raise ValueError("Invalid tool call identity")
                seen.add(ident)
                pending.add(ident)
    if pending:
        raise ValueError("Input ends with unanswered tool calls")
    target = row["target"]
    if target.get("role") != "assistant":
        raise ValueError("Only assistant targets can be supervised")
    if not target.get("content") and not target.get("tool_calls"):
        raise ValueError("Empty target")
    for call in target.get("tool_calls", []):
        name = call["function"]["name"]
        if name not in schemas:
            raise ValueError(f"Unavailable target tool: {name}")
        Draft202012Validator(schemas[name]).validate(arguments(call))


def validate_splits(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids: set[str] = set()
    groups: dict[str, str] = {}
    inputs: dict[str, str] = {}
    sessions: dict[str, str] = {}
    counts = {split: 0 for split in ("train", "dev", "test")}
    for row in rows:
        validate_decision(row)
        if row["case_id"] in ids:
            raise ValueError("Duplicate case ID")
        ids.add(row["case_id"])
        split = row["split"]
        group = row["group_id"]
        fingerprint = digest({"messages": row["messages"], "tools": row["tools"]})
        if group in groups and groups[group] != split:
            raise ValueError("Scenario/session group leakage")
        if fingerprint in inputs and inputs[fingerprint] != split:
            raise ValueError("Identical input across splits")
        groups[group] = split
        session = row.get("capture_session_hash")
        if session:
            if session in sessions and sessions[session] != split:
                raise ValueError("Captured session leakage")
            sessions[session] = split
        inputs[fingerprint] = split
        counts[split] += 1
    return {"counts": counts, "groups": len(groups), "dataset_sha256": digest(rows)}
