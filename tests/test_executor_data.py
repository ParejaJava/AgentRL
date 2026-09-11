"""Dataset leakage, supervision, privacy and evaluation failure regressions."""

import copy
import json
from pathlib import Path

import pytest

from training.build_dataset import build
from training.data import validate_decision, validate_splits
from training.evaluate_executor import score
from training.export_trajectories import export
from training.modeling import parse_output


def test_authored_splits_and_targets_are_valid() -> None:
    report = validate_splits(build())
    assert report["counts"] == {"train": 224, "dev": 56, "test": 84}


def test_authored_currency_and_compressed_context_match_domain_schema() -> None:
    from app.domain.catalog.money import Money
    from app.infrastructure.context_governance.schemas import TaskState

    for row in build():
        if row["family"] == "grounded":
            result = json.loads(row["messages"][-1]["content"])
            price = result["items"][0]["product"]["primary_price"]
            assert (
                Money(price["amount_minor"], price["currency"]).major == price["major"]
            )
            assert str(int(price["major"])) in row["target"]["content"]
        if row["family"] == "compressed":
            text = row["messages"][1]["content"]
            payload = json.loads(
                text.removeprefix("<active_context>").removesuffix("</active_context>")
            )
            TaskState.model_validate(payload["task_state"])


def test_dataset_rejects_group_leakage_and_missing_tool_result() -> None:
    rows = build()
    clone = copy.deepcopy(rows[0])
    clone.update(case_id="leaked", split="test")
    with pytest.raises(ValueError, match="leakage"):
        validate_splits([rows[0], clone])
    row = next(r for r in rows if r["family"] == "grounded")
    row["messages"].pop()
    with pytest.raises(ValueError, match="unanswered"):
        validate_decision(row)


def test_dataset_rejects_unavailable_target_tool() -> None:
    row = build()[0]
    row["target"]["tool_calls"][0]["function"]["name"] = "create_order_intent"
    with pytest.raises(ValueError, match="Unavailable"):
        validate_decision(row)


def test_capture_export_requires_review_and_redacts_consistently(
    tmp_path: Path,
) -> None:
    row = build()[0]
    ident = "a" * 32
    row["messages"][0]["content"] += " email=user@example.test"
    capture = {
        "status": "success",
        "agent_role": "executor",
        "model": "local",
        "messages": row["messages"],
        "tools": row["tools"],
        "response": [row["target"]],
        "prompt_hash": "hash",
        "thread_id": "synthetic-session-1",
    }
    (tmp_path / f"{ident}.json").write_text(json.dumps(capture), encoding="utf-8")
    review = {
        "attempt_id": ident,
        "approved": True,
        "group_id": "session-1",
        "split": "train",
    }
    with pytest.raises(ValueError, match="privacy"):
        export(tmp_path, [review])
    review.update(privacy_reviewed=True, replacements={"user@example.test": "EMAIL_1"})
    result = export(tmp_path, [review])
    assert "user@example.test" not in json.dumps(result)
    assert "EMAIL_1" in json.dumps(result)
    assert result[0]["target_origin"] == "captured_response"
    corrected_review = {
        **review,
        "corrected_target": {"role": "assistant", "content": "请确认所需商品数量。"},
    }
    corrected = export(tmp_path, [corrected_review])[0]
    assert corrected["target_origin"] == "review_correction"
    assert corrected["capture_response_hash"] == result[0]["capture_response_hash"]
    assert corrected["review_sha256"] != result[0]["review_sha256"]
    assert corrected["target"]["content"] == "请确认所需商品数量。"
    capture["status"] = "error"
    (tmp_path / f"{ident}.json").write_text(json.dumps(capture), encoding="utf-8")
    with pytest.raises(ValueError, match="completed"):
        export(tmp_path, [review])


def test_malformed_tool_output_is_not_accepted_as_final_text() -> None:
    with pytest.raises(ValueError, match="Unclosed"):
        parse_output('<tool_call>{"name":"item_search"}')
    with pytest.raises(TypeError, match="argument"):
        parse_output('<tool_call>{"name":"item_search","arguments":[]}</tool_call>')
    result = parse_output(
        '<tool_call>{"name":"item_search","arguments":{"normalized_query":"杯子"}}</tool_call>'
    )
    assert result["tool_calls"][0]["function"]["name"] == "item_search"


def test_evaluation_rejects_wrong_query_or_missing_requested_count() -> None:
    row = build()[0]
    answer = copy.deepcopy(row["target"])
    assert score(row, answer)["passed"]
    args = answer["tool_calls"][0]["function"]["arguments"]
    args["ship_to"] = args["ship_to"].lower()
    args["target_currency"] = args["target_currency"].lower()
    assert score(row, answer)["passed"]
    args["normalized_query"] = "unrelated product"
    assert not score(row, answer)["passed"]
    args["normalized_query"] = args["category"]
    args.pop("top_k")
    assert not score(row, answer)["passed"]


def test_grounded_score_accepts_currency_name_but_not_price_substring() -> None:
    row = next(r for r in build() if r["case_id"] == "train-00-grounded")
    answer = copy.deepcopy(row["target"])
    answer["content"] = answer["content"].replace("EUR", "欧元")
    assert score(row, answer)["passed"]
    answer["content"] = answer["content"].replace("29 欧元", "299 欧元")
    assert not score(row, answer)["passed"]
