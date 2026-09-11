"""Reject invalid supervision before freezing or fitting V2."""

from collections import Counter

from training.build_dataset_v2 import build, complete_tasks
from training.data import digest, validate_splits
from training.evaluate_executor_v2 import score


def test_v2_split_integrity_and_all_authored_targets_satisfy_their_rubrics():
    rows = build()
    assert validate_splits(rows)["counts"] == {"train": 1500, "dev": 200, "test": 300}
    assert len({digest(r["messages"]) for r in rows}) == 2000
    assert Counter(r["bucket"] for r in rows if r["split"] == "train") == {
        "normal": 400,
        "clarification": 350,
        "grounding": 300,
        "recovery": 250,
        "multistep": 200,
    }
    for row in rows:
        assert score(row, row["target"])["passed"], row["case_id"]
    tasks = complete_tasks()
    assert len(tasks) == 40
    assert len({r["case_id"] for r in tasks}) == 40


def test_v2_score_does_not_accept_json_substrings_or_extra_prices():
    row = next(r for r in build() if "selection_ids" in r["rubric"])
    assert not score(row, {"content": "Here is JSON: " + row["target"]["content"]})[
        "passed"
    ]
    assert not score(row, {"content": '{"item_ids":[],"price":1}'})["passed"]
