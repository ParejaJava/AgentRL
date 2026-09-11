"""Run frozen tasks through MainAgent -> fork -> real local model -> real tools.

Planner dispatch, catalog, exchange rates and shipping are explicit deterministic
fixtures. This measures executor integration, not planner intelligence or live
commerce. Never score the deterministic response guard as raw model accuracy.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from collections import Counter
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

from app.application.catalog.category_resolution import CategoryCatalog
from app.application.catalog.config import ItemSearchConfig
from app.application.catalog.models import ItemSearchCommand, RecallHit
from app.application.catalog.search_catalog import ItemSearchService
from app.application.runtime import AgentExecutionContext, ShoppingContextSnapshot
from app.domain.catalog import Money, Product, ProductSearchSpec, Sku
from app.domain.shipping import ShippingQuote
from app.infrastructure.context_governance.config import GovernanceConfig
from app.infrastructure.evidence_budget import EvidenceUsageBudget
from app.infrastructure.langchain.main_agent import MainAgent
from app.infrastructure.langchain.reliability_middleware import (
    ModelAttemptBudgetMiddleware,
)
from app.infrastructure.langchain.shopping_guard import ShoppingGuardMiddleware
from app.infrastructure.langchain.tools.product_search import (
    create_category_resolution_tool,
    create_item_search_tool,
)
from app.infrastructure.observability.trajectory_recorder import (
    TrajectoryRecorderMiddleware,
)
from training.data import digest, read_jsonl
from training.integration_smoke import UnusedDenseBackend
from training.provenance import file_hash


class Dispatcher(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "fixture-dispatcher"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        last_user = max(
            i for i, m in enumerate(messages) if isinstance(m, HumanMessage)
        )
        results = [m for m in messages[last_user + 1 :] if isinstance(m, ToolMessage)]
        message = (
            AIMessage(content=results[-1].text)
            if results
            else AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "fork_sub_agents",
                        "id": uuid4().hex,
                        "args": {
                            "reason": "context_isolation",
                            "tasks": [
                                "按已确认条件完成购物任务。" + messages[last_user].text
                            ],
                        },
                    }
                ],
            )
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


class CatalogFixture:
    dimension = 2
    index_id = "v2-acceptance"

    def __init__(self, case: dict) -> None:
        category, ccy, country = case["category"], case["currency"], case["ship_to"]
        self.products = {
            ident: Product(
                item_id=ident,
                title=f"{category} {ident}",
                category=category,
                brand=brand,
                ships_to=destinations,
                platform="synthetic-fixture",
                metadata={"category_aliases": "出行便携用品"},
                skus=(
                    Sku(
                        f"{ident}-SKU",
                        "标准款",
                        Money.from_major_units(price, ccy),
                        stock,
                    ),
                ),
            )
            for ident, price, brand, stock, destinations in (
                ("ACCEPT-A", 20, "BrandAllowed", 12, (country,)),
                ("ACCEPT-B", 30, "BrandBlocked", 12, (country,)),
                ("ACCEPT-C", 120, "BrandAllowed", 12, (country,)),
                ("ACCEPT-D", 10, "BrandAllowed", 12, ("CN",)),
                ("ACCEPT-E", 5, "BrandAllowed", 0, (country,)),
            )
        }

    @property
    def size(self) -> int:
        return len(self.products)

    def get(self, index_id: str) -> CatalogFixture:
        assert index_id == self.index_id
        return self

    def get_product(self, ident: str) -> Product:
        return self.products[ident]

    def category_catalog(self) -> CategoryCatalog:
        return CategoryCatalog.from_products(self.products.values())

    def keyword_search(self, query: str, top_k: int) -> tuple[RecallHit, ...]:
        # All fixture products share a category. No model/retrieval-quality claim.
        return tuple(
            RecallHit(p.item_id, 1.0)
            for p in self.products.values()
            if p.category in query
        )[:top_k]


class FixturePricing:
    def supported_destinations(self) -> tuple[str, ...]:
        return ("DE", "US", "JP", "GB")

    def convert(self, money: Money, target_currency: str) -> Money:
        if money.currency != target_currency:
            raise ValueError("Fixture does not invent cross-currency rates")
        return money

    def quote(
        self,
        *,
        unit_price: Money,
        category: str,
        ship_to: str,
        quantity: int,
        target_currency: str,
    ) -> ShippingQuote:
        subtotal = self.convert(unit_price, target_currency).multiply(quantity)
        return ShippingQuote(
            ship_to=ship_to,
            quantity=quantity,
            subtotal=subtotal,
            freight=Money.from_major_units(10, target_currency),
            tariff=Money.from_major_units(5, target_currency),
            tariff_rate=Decimal(0),
            de_minimis_applied=False,
            tariff_rule_version="synthetic-fixed-5",
            exchange_rate_version="synthetic-same-currency",
            exchange_rate_as_of="fixture",
        )


class ObservedSearch(ItemSearchService):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.executed: list[dict] = []

    def search(self, command: ItemSearchCommand):
        result = super().search(command)
        self.executed.append({"spec": asdict(command.spec), "result": result.to_dict()})
        return result


def unwrap(text: str) -> str:
    try:
        result = json.loads(text)
        return "\n".join(str(item.get("answer", "")) for item in result["results"])
    except (ValueError, TypeError, KeyError):
        return text


async def run_case(case: dict, endpoint: str, model: str, output: Path) -> dict:
    output.mkdir(parents=True)
    fixture, dense = CatalogFixture(case), UnusedDenseBackend()
    service = ObservedSearch(
        encoder=dense,
        reranker=dense,
        indexes=fixture,
        pricing=FixturePricing(),
        config=ItemSearchConfig(embedding_dimension=2, retrieval_mode="lexical"),
    )
    expected = service.search(
        ItemSearchCommand(
            spec=ProductSearchSpec(**case["expected_args"]), index_id=fixture.index_id
        )
    ).to_dict()
    expected_ids = [item["product"]["item_id"] for item in expected["items"]]
    fixture_ids = {
        "unit": ["ACCEPT-A", "ACCEPT-B"],
        "subtotal": ["ACCEPT-A", "ACCEPT-B"],
        "landed": ["ACCEPT-A"],
        "exclude": ["ACCEPT-A"],
        "required": ["ACCEPT-A"],
        "update-budget": ["ACCEPT-A"],
        "clarify-currency": ["ACCEPT-A", "ACCEPT-B"],
        "empty-budget": [],
        "stock": [],
        "alias": ["ACCEPT-A", "ACCEPT-B"],
    }[case["family"]]
    assert expected_ids == fixture_ids, (
        "Fixture business oracle disagrees with independently specified expected IDs"
    )
    service.executed.clear()
    budget = EvidenceUsageBudget(max_total_requests=20, max_observed_tokens=120000)
    agent = MainAgent(
        model=Dispatcher(),
        sub_agent_model=ChatOpenAI(
            model=model,
            base_url=endpoint,
            api_key="local-no-key",
            temperature=0,
            max_tokens=256,
            max_retries=0,
            timeout=180,
        ),
        tools=[
            create_item_search_tool(service, index_id=fixture.index_id),
            create_category_resolution_tool(service, index_id=fixture.index_id),
        ],
        shared_middleware=(ShoppingGuardMiddleware(output / "guard"),),
        main_boundary_middleware=(
            TrajectoryRecorderMiddleware(output / "captures", agent_role="planner"),
        ),
        sub_boundary_middleware=(
            ModelAttemptBudgetMiddleware(budget),
            TrajectoryRecorderMiddleware(output / "captures", agent_role="executor"),
        ),
        governance_config=GovernanceConfig(
            mode="deterministic",
            context_window_tokens=8192,
            session_root=output / "sessions",
        ),
    )
    context = AgentExecutionContext(
        thread_id=uuid4().hex,
        run_id=uuid4().hex,
        shopping=ShoppingContextSnapshot(
            "fixture-session-" + uuid4().hex, "fixture-buyer", currency=case["currency"]
        ),
    )
    turns, error = [], None
    started = time.perf_counter()
    for user in case["turns"]:
        prior = len(service.executed)
        events = []
        try:
            async for event in agent.stream(user, context):
                events.append(event)
        except Exception as exc:  # noqa: BLE001 - retain failed task evidence.
            error = f"{type(exc).__name__}: {exc}"
        final = unwrap(
            "".join(
                event.get("content", "")
                for event in events
                if event.get("type") == "text_message_content"
            )
        )
        turns.append(
            {
                "user": user,
                "final": final,
                "executions": service.executed[prior:],
                "events": events,
            }
        )
        if error:
            break
    duration = time.perf_counter() - started
    captures = sorted(
        [
            json.loads(path.read_text(encoding="utf-8"))
            for path in (output / "captures").glob("*.json")
        ],
        key=lambda x: x["created_at"],
    )
    executor = [c for c in captures if c["agent_role"] == "executor"]
    guards = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (output / "guard").glob("*.json")
    ]
    positive_guards = [g for g in guards if g.get("has_candidates")]
    final = turns[-1]["final"]
    invoked = turns[-1]["executions"]
    matching = [
        e
        for e in invoked
        if all(
            e["spec"].get(k) == v
            for k, v in case["expected_args"].items()
            if k not in {"normalized_query", "excluded_brands", "category"}
        )
        and e["spec"].get("category") in {case["category"], "出行便携用品"}
        and set(case["expected_args"].get("excluded_brands", [])).issubset(
            e["spec"]["excluded_brands"]
        )
    ]
    actual_ids = (
        [p["product"]["item_id"] for p in matching[-1]["result"]["items"]]
        if matching
        else None
    )
    if expected_ids:
        final_ok = all(f"{ident}（" in final for ident in expected_ids) and all(
            f"{ident}（" not in final
            for ident in fixture.products
            if ident not in expected_ids
        )
        final_ok = (
            final_ok
            and all(final.count(f"{ident}（") == 1 for ident in expected_ids)
            and "估算到手总价" in final
        )
        quantity = case["expected_args"]["quantity"]
        for ident in expected_ids:
            price = 20 if ident == "ACCEPT-A" else 30
            final_ok = (
                final_ok
                and f"SKU {ident}-SKU" in final
                and f"单件标价 {price} {case['currency']}，购买 {quantity} 件" in final
                and f"估算到手总价 {price * quantity + 15} {case['currency']}" in final
            )
    else:
        final_ok = "当前检索范围内没有" in final and not any(
            f"{ident}（" in final for ident in fixture.products
        )
    clarification_ok = case["family"] != "clarify-currency" or (
        not turns[0]["executions"]
        and any(word in turns[0]["final"] for word in ("币种", "货币"))
    )
    blocked = sum(
        1
        for c in executor
        for m in c["messages"]
        if m.get("role") == "tool" and '"constraint_mismatch"' in m.get("content", "")
    )
    result = {
        "case_id": case["case_id"],
        "family": case["family"],
        "error": error,
        "guarded_task_passed": error is None
        and actual_ids == expected_ids
        and final_ok
        and clarification_ok,
        "expected_ids": expected_ids,
        "actual_ids": actual_ids,
        "final_ok": final_ok,
        "clarification_ok": clarification_ok,
        "duration_seconds": duration,
        "executor_model_calls": len(executor),
        "constraint_mismatch_observations": blocked,
        "raw_selection_checks": len(positive_guards),
        "raw_selection_failures": sum(
            not g["raw_selection_valid"] for g in positive_guards
        ),
        "parent_links_present": bool(executor)
        and all(c["parent_run_id"] == context.run_id for c in executor),
        "turns": turns,
        "budget": budget.snapshot(),
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


async def main_async(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise ValueError("Choose a new evidence directory")
    cases = read_jsonl(args.cases)
    args.output.mkdir(parents=True)
    metadata = {
        "model": args.model,
        "cases_sha256": digest(cases),
        "source_sha256": file_hash(Path(__file__)),
        "decoding": "greedy",
        "max_new_tokens": 256,
        "context_window": 8192,
        "guard_enabled": True,
        "limitations": "Frozen synthetic catalog and scripted dispatcher. Real executor model, MainAgent/fork, production filtering and deterministic answer guard. Raw selection failure and guarded task success are separate.",
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    results = []
    for case in cases:
        result = await run_case(
            case, args.endpoint, args.model, args.output / case["case_id"]
        )
        results.append(result)
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "case_id",
                        "guarded_task_passed",
                        "raw_selection_failures",
                        "duration_seconds",
                        "error",
                    )
                }
            ),
            flush=True,
        )
    durations = sorted(r["duration_seconds"] for r in results)
    report = {
        "total": len(results),
        "guarded_passed": sum(r["guarded_task_passed"] for r in results),
        "raw_selection_checks": sum(r["raw_selection_checks"] for r in results),
        "raw_selection_failures": sum(r["raw_selection_failures"] for r in results),
        "executor_model_calls": sum(r["executor_model_calls"] for r in results),
        "p50_seconds": statistics.median(durations),
        "p95_seconds": durations[math.ceil(len(durations) * 0.95) - 1],
        "families_passed": dict(
            Counter(r["family"] for r in results if r["guarded_task_passed"])
        ),
        "failed_cases": [r["case_id"] for r in results if not r["guarded_task_passed"]],
    }
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8010/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", default="eval/executor_v2/complete_tasks.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.endpoint != "http://127.0.0.1:8010/v1":
        raise ValueError("This evaluation only uses the local development server")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
