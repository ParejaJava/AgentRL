"""Business invariants across model mistakes, turns and fork boundaries."""

import asyncio
import json
from types import SimpleNamespace
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.application.catalog.category_resolution import CategoryCatalog
from app.application.catalog.config import ItemSearchConfig
from app.application.catalog.recommendations import validate_selection
from app.application.catalog.requirements import (
    check_search_arguments,
    update_requirements,
)
from app.application.catalog.search_catalog import ItemSearchService
from app.application.runtime import AgentExecutionContext
from app.infrastructure.context import reset_context, set_context
from app.infrastructure.context_governance.config import GovernanceConfig
from app.infrastructure.langchain.main_agent import MainAgent
from app.infrastructure.langchain.shopping_guard import ShoppingGuardMiddleware
from app.infrastructure.langchain.tools.product_search import create_item_search_tool
from training.integration_smoke import FixtureRegistry, UnusedDenseBackend


def test_pending_currency_survives_fork_and_user_can_resolve_it():
    ledger = update_requirements({}, "预算50，配送德国，购买3件，推荐2款。", "u1")
    assert ledger["pending"] == ["target_currency"]
    child = AgentExecutionContext(
        thread_id="child", parent_run_id="parent", confirmed_requirements=ledger
    )
    result = asyncio.run(
        ShoppingGuardMiddleware().abefore_model(
            {"messages": [HumanMessage(content="预算50欧元，配送美国")]},
            SimpleNamespace(context=child),
        )
    )
    assert result["shopping_requirements"]["pending"] == ["target_currency"]
    assert result["shopping_requirements"]["values"]["ship_to"] == "DE"
    resolved = update_requirements(ledger, "按欧元计算，标价不是到手总价。", "u2")
    assert resolved["pending"] == []
    assert resolved["values"]["price_basis"] == "unit"
    args = dict(resolved["values"])
    assert check_search_arguments(resolved, args) is None
    del args["ship_to"]
    assert check_search_arguments(resolved, args)["expected"] == {"ship_to": "DE"}


def test_brand_conflict_requires_explicit_withdrawal_and_budget_cancel_clears_cap():
    ledger = update_requirements({}, "不要品牌甲，只选品牌甲，预算50欧元", "u1")
    assert "brand_conflict" in ledger["pending"]
    ledger = update_requirements(ledger, "撤回品牌甲的排除条件，取消预算", "u2")
    assert not ledger["pending"]
    assert ledger["values"]["excluded_brands"] == []
    assert ledger["values"]["price_max_major"] is None
    assert check_search_arguments(ledger, {**ledger["values"], "price_max_major": 50})


def test_explicit_basis_quantity_and_amount_alternatives_are_not_guessed():
    cases = [
        ("预算50欧元可能指单价，也可能是含税费总价，口径还没定。", "price_basis"),
        ("买2件还是3件尚未确定。", "quantity"),
        ("预算可能是50或100欧元，金额未确定。", "price_max_major"),
    ]
    for text, field in cases:
        ledger = update_requirements({}, text, "ambiguous")
        assert field in ledger["pending"]
        assert field not in ledger["values"]
        assert check_search_arguments(ledger, {})["status"] == "needs_clarification"


def test_category_only_resolves_authoritative_unique_aliases():
    catalog = CategoryCatalog(
        names=("旅行杯", "运动杯"),
        aliases=(("随行杯", "旅行杯"), ("水杯", "旅行杯"), ("水杯", "运动杯")),
    )
    assert catalog.resolve("随行杯")["category"] == "旅行杯"
    assert catalog.resolve("水杯")["status"] == "needs_clarification"
    assert catalog.resolve("杯")["status"] == "needs_clarification"


def test_selection_rejects_duplicates_unknown_ids_and_model_prices():
    products = [{"item_id": "a"}]
    for text in (
        '{"item_ids":["a","a"]}',
        '{"item_ids":["fake"]}',
        '{"item_ids":["a"],"price":1}',
    ):
        assert validate_selection(text, products, 2)[1]
    assert not validate_selection('{"item_ids":["a"]}', products, 2)[1]


def test_malformed_tool_reply_is_recorded_as_failed_evidence():
    async def scenario():
        token = set_context(AgentExecutionContext(thread_id="malformed"))
        try:
            request = SimpleNamespace(
                state={},
                tool_call={"name": "item_search", "args": {}, "id": "bad-result"},
            )

            async def handler(request):
                return ToolMessage(
                    content="invalid provider output", tool_call_id="bad-result"
                )

            result = await ShoppingGuardMiddleware().awrap_tool_call(request, handler)
            evidence = result.update["shopping_evidence"]["bad-result"]["payload"]
            assert evidence["status"] == "error"
            assert "items" not in evidence
        finally:
            reset_context(token)

    asyncio.run(scenario())


class MistakeThenRepair(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "guard-test"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        results = [m for m in messages if isinstance(m, ToolMessage)]
        if len(results) >= 2:
            answer = AIMessage(
                content='{"item_ids":["FAKE","CUP-OK","CUP-OK"],"price":1}'
            )
        else:
            args = {
                "normalized_query": "测试水杯",
                "category": "测试水杯",
                "price_max_major": 50,
                "target_currency": "EUR",
                "top_k": 2,
            }
            if results:
                assert json.loads(results[-1].text)["code"] == "constraint_mismatch"
                args["ship_to"] = "DE"
            answer = AIMessage(
                content="",
                tool_calls=[
                    {"name": "item_search", "id": f"s{len(results)}", "args": args}
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=answer)])


def test_real_agent_blocks_lost_destination_and_renders_only_tool_evidence(tmp_path):
    async def scenario():
        dense = UnusedDenseBackend()
        service = ItemSearchService(
            encoder=dense,
            reranker=dense,
            indexes=FixtureRegistry(),
            config=ItemSearchConfig(embedding_dimension=2, retrieval_mode="lexical"),
        )
        agent = MainAgent(
            model=MistakeThenRepair(),
            tools=[create_item_search_tool(service, index_id="executor-fixture")],
            enable_fork_tool=False,
            shared_middleware=(ShoppingGuardMiddleware(),),
            governance_config=GovernanceConfig(mode="off", session_root=tmp_path),
        )
        events = [
            event
            async for event in agent.stream(
                "找测试水杯，配送德国，单件标价预算50欧元，推荐2款",
                AgentExecutionContext(thread_id="guard-test"),
            )
        ]
        final = "".join(
            e.get("content", "") for e in events if e["type"] == "text_message_content"
        )
        assert (
            "FAKE" not in final
            and "CUP-OVER" not in final
            and "CUP-NOSHIP" not in final
        )
        assert "20 EUR" in final and "不是到手总价" in final
        state = await agent.state_snapshot("guard-test")
        assert len(state["shopping_evidence"]) == 2
        answer = state["messages"][-1]
        assert not answer.additional_kwargs["shopping_guard"]["raw_selection_valid"]

    asyncio.run(scenario())
