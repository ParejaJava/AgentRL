"""Exercise real MainAgent -> fork -> local executor -> production item_search.

The planner is deliberately scripted to avoid any paid API. The executor is a
real served model. Retrieval uses a tiny declared synthetic lexical catalog and
the production ItemSearchService filtering/domain logic. This proves integration,
not planner intelligence, retrieval quality or production business success.
Run with the application environment while serve_executor is running separately.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

from app.application.catalog.config import ItemSearchConfig
from app.application.catalog.models import RecallHit
from app.application.catalog.search_catalog import ItemSearchService
from app.application.runtime import AgentExecutionContext, ShoppingContextSnapshot
from app.domain.catalog import Money, Product, Sku
from app.infrastructure.context_governance.config import GovernanceConfig
from app.infrastructure.evidence_budget import EvidenceUsageBudget
from app.infrastructure.langchain.main_agent import MainAgent
from app.infrastructure.langchain.reliability_middleware import (
    ModelAttemptBudgetMiddleware,
    ModelGatewayMiddleware,
)
from app.infrastructure.langchain.tools.product_search import create_item_search_tool
from app.infrastructure.observability.trajectory_recorder import (
    ModelCallScopeMiddleware,
    TrajectoryRecorderMiddleware,
)
from app.infrastructure.retrieval.item_search.lexical import keyword_2gram_score


class ScriptedPlanner(BaseChatModel):
    """Only dispatches a fixed task then exposes the executor's actual result."""

    demand: str

    @property
    def _llm_type(self) -> str:
        return "scripted-integration-planner"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedPlanner:
        return self

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        results = [m for m in messages if isinstance(m, ToolMessage)]
        message = (
            AIMessage(content=results[-1].text)
            if results
            else AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "fork_sub_agents",
                        "id": "dispatch-1",
                        "args": {"reason": "context_isolation", "tasks": [self.demand]},
                    }
                ],
            )
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


class UnusedDenseBackend:
    dimension = 2

    def embed_queries(self, texts: Sequence[str]) -> Any:
        raise AssertionError("Lexical fixture must not load embedding weights")

    def score(self, pairs: Sequence[tuple[str, str]]) -> Any:
        raise AssertionError("Lexical fixture must not load reranker weights")


class FixtureIndex:
    dimension = 2
    index_id = "executor-fixture"

    def __init__(self) -> None:
        self.products = {
            ident: Product(
                item_id=ident,
                title=f"测试水杯 {ident}",
                category="测试水杯",
                ships_to=destinations,
                platform="synthetic-fixture",
                skus=(
                    Sku(
                        f"{ident}-sku", "标准", Money.from_major_units(price, "EUR"), 10
                    ),
                ),
            )
            for ident, price, destinations in (
                ("CUP-OK", 20, ("DE",)),
                ("CUP-OVER", 80, ("DE",)),
                ("CUP-NOSHIP", 10, ("US",)),
            )
        }

    @property
    def size(self) -> int:
        return len(self.products)

    def get_product(self, ident: str) -> Product:
        return self.products[ident]

    def keyword_search(self, query: str, top_k: int) -> tuple[RecallHit, ...]:
        hits = [
            RecallHit(p.item_id, keyword_2gram_score(query, p.to_search_text()))
            for p in self.products.values()
        ]
        return tuple(
            sorted((h for h in hits if h.score > 0), key=lambda h: -h.score)[:top_k]
        )


class FixtureRegistry:
    def __init__(self) -> None:
        self.index = FixtureIndex()

    def get(self, index_id: str) -> FixtureIndex:
        if index_id != self.index.index_id:
            raise KeyError(index_id)
        return self.index


def assess_fixture_answer(final: str) -> dict[str, bool]:
    """Check the one-product fixture answer, beyond successful tool execution."""
    try:
        envelope = json.loads(final)
    except ValueError:
        envelope = None
    if isinstance(envelope, dict) and isinstance(envelope.get("results"), list):
        final = "\n".join(str(item.get("answer", "")) for item in envelope["results"])
    return {
        "single_valid_product": len(re.findall(r"\bCUP-OK\b", final)) == 1,
        "no_filtered_product": "CUP-OVER" not in final and "CUP-NOSHIP" not in final,
        "correct_price": bool(
            re.search(r"(?<![\d.])20(?:\.0+)?\s*(?:欧元|EUR)", final)
        ),
        "price_is_not_checkout_total": "不是到手总价" in final or "非到手总价" in final,
    }


async def run(
    endpoint: str,
    model_name: str,
    output: Path,
    mode: str,
    explicit_category: bool = False,
) -> dict:
    if output.exists():
        raise ValueError("Choose a new integration evidence directory")
    output.mkdir(parents=True)
    dense = UnusedDenseBackend()
    service = ItemSearchService(
        encoder=dense,
        reranker=dense,
        indexes=FixtureRegistry(),
        config=ItemSearchConfig(embedding_dimension=2, retrieval_mode="lexical"),
    )
    search = create_item_search_tool(service, index_id="executor-fixture")
    demand = "只用商品搜索查询测试水杯，配送德国，标价预算50欧元以内，最多2个。根据工具结果给出商品编号和标价，说明不是到手总价。"
    if explicit_category:
        demand += "商品目录中的品类名称为“测试水杯”，搜索时请完整保留该品类名称。"
    budget = EvidenceUsageBudget(max_total_requests=8, max_observed_tokens=20000)
    gateway = ModelGatewayMiddleware(
        max_concurrency=1,
        min_interval_seconds=0,
        max_retries=0,
        evidence_budget=budget,
        external_attempt_accounting=True,
    )
    runtime = MainAgent(
        model=ScriptedPlanner(demand=demand),
        sub_agent_model=ChatOpenAI(
            model=model_name,
            base_url=endpoint,
            api_key="local-no-key",
            max_retries=0,
            temperature=0,
            max_tokens=256,
            timeout=180,
        ),
        tools=[search],
        governance_config=GovernanceConfig(
            mode=mode, context_window_tokens=4096, session_root=output / "sessions"
        ),
        main_model_middleware=(ModelCallScopeMiddleware(), gateway),
        sub_model_middleware=(ModelCallScopeMiddleware(), gateway),
        main_boundary_middleware=(
            ModelAttemptBudgetMiddleware(budget),
            TrajectoryRecorderMiddleware(output / "captures", agent_role="planner"),
        ),
        sub_boundary_middleware=(
            ModelAttemptBudgetMiddleware(budget),
            TrajectoryRecorderMiddleware(output / "captures", agent_role="executor"),
        ),
    )
    context = AgentExecutionContext(
        thread_id=f"integration-{uuid4().hex[:8]}",
        run_id=uuid4().hex,
        shopping=ShoppingContextSnapshot(
            "fixture-session", "fixture-buyer", currency="EUR"
        ),
    )
    events = []
    error = None
    try:
        async for event in runtime.stream(demand, context):
            events.append(event)
            print(json.dumps(event, ensure_ascii=False), flush=True)
    except Exception as exc:  # noqa: BLE001 - persist failed integration evidence.
        error = f"{type(exc).__name__}: {exc}"
    captures = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in (output / "captures").glob("*.json")
    ]
    executor = [r for r in captures if r["agent_role"] == "executor"]
    paired = any(any(m.get("role") == "tool" for m in r["messages"]) for r in executor)
    tool_invocations = []
    consumed_product_ids = []
    for capture in executor:
        for answer in capture.get("response", []):
            for call in answer.get("tool_calls", []):
                if call["function"]["name"] == "item_search":
                    arguments = call["function"]["arguments"]
                    tool_invocations.append(
                        json.loads(arguments)
                        if isinstance(arguments, str)
                        else arguments
                    )
        for message in capture["messages"]:
            if message.get("role") == "tool":
                try:
                    payload = json.loads(message.get("content", ""))
                    if payload.get("index_id") == "executor-fixture":
                        consumed_product_ids.append(
                            [item["product"]["item_id"] for item in payload["items"]]
                        )
                except (ValueError, KeyError, TypeError):
                    continue
    arguments_ok = any(
        args.get("category") == "测试水杯"
        and str(args.get("ship_to", "")).upper() == "DE"
        and str(args.get("target_currency", "")).upper() == "EUR"
        and args.get("price_max_major") == 50
        and args.get("top_k", 5) == 2
        for args in tool_invocations
    )
    final = "\n".join(
        e.get("content", "") for e in events if e.get("type") == "text_message_content"
    )
    result = {
        "model": model_name,
        "governance_mode": mode,
        "explicit_catalog_category": explicit_category,
        "demand": demand,
        "error": error,
        "executor_attempts": len(executor),
        "production_tool_result_consumed": paired,
        "tool_invocations": tool_invocations,
        "executor_arguments_ok": arguments_ok,
        "consumed_product_ids": consumed_product_ids,
        "correct_filtered_result_consumed": ["CUP-OK"] in consumed_product_ids,
        "parent_links_present": bool(executor)
        and all(r["parent_run_id"] == context.run_id for r in executor),
        "final_contains_valid_product": "CUP-OK" in final,
        "final": final,
        "budget": budget.snapshot(),
        "events": events,
        "limitations": "Scripted planner, real local executor, synthetic lexical catalog using production item filtering.",
    }
    result["integration_path_passed"] = (
        error is None
        and paired
        and result["parent_links_present"]
        and result["final_contains_valid_product"]
        and result["executor_arguments_ok"]
        and result["correct_filtered_result_consumed"]
    )
    result["answer_quality"] = assess_fixture_answer(final)
    result["answer_quality_passed"] = all(result["answer_quality"].values())
    result["passed"] = (
        result["integration_path_passed"] and result["answer_quality_passed"]
    )
    (output / "integration.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8010/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--governance", choices=["off", "deterministic"], default="deterministic"
    )
    parser.add_argument(
        "--explicit-category",
        action="store_true",
        help="Also supply the exact catalog category; keeps all outcome checks unchanged.",
    )
    args = parser.parse_args()
    if args.endpoint not in {"http://127.0.0.1:8010/v1", "http://localhost:8010/v1"}:
        raise ValueError(
            "Integration smoke is restricted to the local development server"
        )
    print(
        json.dumps(
            asyncio.run(
                run(
                    args.endpoint,
                    args.model,
                    args.output,
                    args.governance,
                    args.explicit_category,
                )
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
