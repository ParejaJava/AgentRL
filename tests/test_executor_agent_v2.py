"""Exercise evaluator plumbing with a scripted double before any model test run."""

import asyncio
import json
import re
from typing import Any
from uuid import uuid4

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from training import evaluate_agent_v2 as evaluation
from training.build_dataset_v2 import complete_tasks


class ExecutorDouble(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "evaluation-plumbing-double"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        system = next(m.text for m in messages if isinstance(m, SystemMessage))
        match = re.search(
            r"<confirmed_shopping_constraints>(.*?)</confirmed_shopping_constraints>",
            system,
            re.DOTALL,
        )
        ledger = json.loads(match.group(1))
        values = dict(ledger["values"])
        results = [m for m in messages if isinstance(m, ToolMessage)]
        if ledger["pending"]:
            message = AIMessage(content="请确认预算币种。")
        elif results and "items" in json.loads(results[-1].text):
            items = json.loads(results[-1].text)["items"]
            message = AIMessage(
                content=json.dumps(
                    {"item_ids": [item["product"]["item_id"] for item in items]}
                )
                if items
                else "当前没有候选"
            )
        else:
            values["normalized_query"] = values["category"]
            message = AIMessage(
                content="",
                tool_calls=[{"name": "item_search", "id": uuid4().hex, "args": values}],
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


@pytest.mark.parametrize(
    "kind",
    ["unit", "landed", "empty-budget", "stock", "update-budget", "clarify-currency"],
)
def test_frozen_task_evaluator_with_scripted_executor(monkeypatch, tmp_path, kind):
    monkeypatch.setattr(evaluation, "ChatOpenAI", lambda **kwargs: ExecutorDouble())
    case = next(c for c in complete_tasks() if c["family"] == kind)
    result = asyncio.run(
        evaluation.run_case(
            case, "http://127.0.0.1:8010/v1", "scripted-test-only", tmp_path / kind
        )
    )
    assert result["guarded_task_passed"], result
    assert result["parent_links_present"]
    assert result["raw_selection_failures"] == 0


class BadSummaryDispatcher(evaluation.Dispatcher):
    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if any(isinstance(m, ToolMessage) for m in messages):
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(content="FAKE 商品只需 1 EUR，已经下单。")
                    )
                ]
            )
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def test_parent_cannot_rewrite_verified_child_prices(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluation, "ChatOpenAI", lambda **kwargs: ExecutorDouble())
    monkeypatch.setattr(evaluation, "Dispatcher", BadSummaryDispatcher)
    case = next(c for c in complete_tasks() if c["family"] == "unit")
    result = asyncio.run(
        evaluation.run_case(
            case,
            "http://127.0.0.1:8010/v1",
            "scripted-test-only",
            tmp_path / "bad-parent",
        )
    )
    assert result["guarded_task_passed"]
    assert "FAKE" not in result["turns"][-1]["final"]
    assert "已下单" not in result["turns"][-1]["final"]
