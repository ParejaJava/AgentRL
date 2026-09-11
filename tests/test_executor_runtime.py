"""Executor routing and exact-boundary recording without external services."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.application.runtime import AgentExecutionContext, ShoppingContextSnapshot
from app.infrastructure.context_governance.config import GovernanceConfig
from app.infrastructure.evidence_budget import EvidenceUsageBudget
from app.infrastructure.langchain.main_agent import MainAgent
from app.infrastructure.langchain.reliability_middleware import (
    ModelAttemptBudgetMiddleware,
    ModelGatewayMiddleware,
)
from app.infrastructure.llm import create_chat_model, create_executor_model
from app.infrastructure.observability.trajectory_recorder import (
    ModelCallScopeMiddleware,
    TrajectoryRecorderMiddleware,
)
from app.infrastructure.settings import Settings
from training.integration_smoke import ScriptedPlanner


class EchoExecutor(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "test-executor"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content="executor-only-answer"))
            ]
        )


@pytest.mark.parametrize("max_requests", [1, 2])
def test_emergency_attempts_are_recorded_and_individually_budgeted(
    tmp_path: Path, max_requests: int
) -> None:
    from langgraph.runtime import Runtime

    from app.infrastructure.context import reset_context, set_context
    from app.infrastructure.context_governance.middleware import (
        ContextGovernanceMiddleware,
    )
    from app.infrastructure.evidence_budget import EvidenceBudgetExceeded

    async def scenario() -> None:
        budget = EvidenceUsageBudget(max_total_requests=max_requests)
        gateway = ModelGatewayMiddleware(
            max_concurrency=1,
            min_interval_seconds=0,
            max_retries=0,
            evidence_budget=budget,
            external_attempt_accounting=True,
        )
        accountant = ModelAttemptBudgetMiddleware(budget)
        recorder = TrajectoryRecorderMiddleware(
            tmp_path / "capture", agent_role="executor"
        )
        governance = ContextGovernanceMiddleware(
            compressor=None,
            config=GovernanceConfig(
                context_window_tokens=100000,
                hot_message_count=2,
                session_root=tmp_path / "sessions",
            ),
            static_system_prompt="system",
            agent_id="executor",
        )
        context = AgentExecutionContext(thread_id="emergency", run_id="emergency-run")
        messages = [
            HumanMessage(content=f"old {i} " * 20)
            if i % 2 == 0
            else AIMessage(content=f"answer {i} " * 20)
            for i in range(8)
        ]
        messages.append(HumanMessage(content="preserve latest request"))
        request = ModelRequest(
            model=None, messages=messages, state={}, runtime=Runtime(context=context)
        )
        calls = []

        async def endpoint(actual):
            calls.append(actual)
            if len(calls) == 1:
                raise RuntimeError("maximum context length exceeded")
            return ModelResponse(
                result=[
                    AIMessage(
                        content="ok",
                        usage_metadata={
                            "input_tokens": 10,
                            "output_tokens": 2,
                            "total_tokens": 12,
                        },
                    )
                ]
            )

        async def capture(actual):
            return await recorder.awrap_model_call(actual, endpoint)

        async def account(actual):
            return await accountant.awrap_model_call(actual, capture)

        async def govern(actual):
            return await governance.awrap_model_call(actual, account)

        async def route(actual):
            return await gateway.awrap_model_call(actual, govern)

        token = set_context(context)
        try:
            if max_requests == 1:
                with pytest.raises(EvidenceBudgetExceeded):
                    await ModelCallScopeMiddleware().awrap_model_call(request, route)
            else:
                await ModelCallScopeMiddleware().awrap_model_call(request, route)
        finally:
            reset_context(token)
        assert budget.snapshot()["started_requests"] == max_requests
        assert budget.snapshot()["completed_requests"] == max_requests - 1
        assert len(calls) == max_requests
        if max_requests == 2:
            assert len(calls[1].messages) < len(calls[0].messages)
        records = [
            json.loads(p.read_text(encoding="utf-8"))
            for p in (tmp_path / "capture").glob("*.json")
        ]
        assert len(records) == max_requests
        assert len({r["step_id"] for r in records}) == 1
        assert all(
            any("preserve latest" in str(m["content"]) for m in r["messages"])
            for r in records
        )

    asyncio.run(scenario())


def test_real_agent_graph_routes_child_and_records_governed_input(
    tmp_path: Path,
) -> None:
    captures = tmp_path / "captures"
    runtime = MainAgent(
        model=ScriptedPlanner(demand="isolated task"),
        sub_agent_model=EchoExecutor(),
        tools=[],
        governance_config=GovernanceConfig(
            mode="deterministic", session_root=tmp_path / "sessions"
        ),
        main_model_middleware=(ModelCallScopeMiddleware(),),
        sub_model_middleware=(ModelCallScopeMiddleware(),),
        main_boundary_middleware=(
            TrajectoryRecorderMiddleware(captures, agent_role="planner"),
        ),
        sub_boundary_middleware=(
            TrajectoryRecorderMiddleware(captures, agent_role="executor"),
        ),
    )

    async def scenario() -> list[dict]:
        return [
            event
            async for event in runtime.stream(
                "dispatch",
                AgentExecutionContext(
                    thread_id="parent",
                    run_id="parent-run",
                    shopping=ShoppingContextSnapshot("shopping-1", "buyer-1"),
                ),
            )
        ]

    events = asyncio.run(scenario())
    assert any("executor-only-answer" in e.get("content", "") for e in events)
    records = [
        json.loads(p.read_text(encoding="utf-8")) for p in captures.glob("*.json")
    ]
    children = [r for r in records if r["agent_role"] == "executor"]
    assert len(children) == 1
    assert children[0]["model_class"] == "EchoExecutor"
    assert children[0]["parent_run_id"] == "parent-run"
    assert children[0]["thread_id"] != "parent"
    assert any("<active_context>" in str(m["content"]) for m in children[0]["messages"])
    assert children[0]["tools"] == []
    assert len({r["session_group_hash"] for r in records}) == 1
    assert all(
        r["model_class"] == "ScriptedPlanner"
        for r in records
        if r["agent_role"] == "planner"
    )


def test_executor_endpoint_does_not_inherit_planner_secret() -> None:
    settings = replace(
        Settings.from_env(),
        executor_model_name="executor-v1",
        executor_base_url="http://127.0.0.1:8010/v1",
        executor_api_key=None,
        llm_api_key="planner-secret",
    )
    model = create_executor_model(settings)
    assert model.model_name == "executor-v1"
    assert model.openai_api_key.get_secret_value() == "local-no-key"
    assert model.openai_api_base == "http://127.0.0.1:8010/v1"
    assert model.max_retries == 0


def test_unconfigured_planner_keeps_sdk_retry_default() -> None:
    settings = replace(
        Settings.from_env(),
        executor_model_name=None,
        trajectory_root=None,
        llm_api_key="test-no-network",
        llm_base_url="http://127.0.0.1:8010/v1",
        llm_model_name="test-default",
    )
    assert create_chat_model(settings).root_client.max_retries == 2
    assert (
        create_chat_model(
            replace(settings, trajectory_root=Path("data/trajectories"))
        ).max_retries
        == 0
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_composition_wires_opt_in_executor_and_shared_limits(
    tmp_path: Path, monkeypatch: Any, enabled: bool
) -> None:
    from app import composition

    captured = {}
    main_model, executor_model = object(), object()

    class SpyRuntime:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(composition, "MainAgent", SpyRuntime)
    monkeypatch.setattr(composition, "create_chat_model", lambda *a, **k: main_model)
    monkeypatch.setattr(
        composition, "create_executor_model", lambda *a, **k: executor_model
    )
    monkeypatch.setattr(composition, "create_item_embedding_encoder", lambda **k: None)
    monkeypatch.setattr(composition, "create_item_index_registry", lambda *a: None)
    monkeypatch.setattr(composition, "build_item_search_service", lambda *a, **k: None)
    monkeypatch.setattr(
        composition, "create_category_insight_service", lambda *a, **k: None
    )
    settings = replace(
        Settings.from_env(),
        context_llm_enabled=False,
        fallback_llm_model=None,
        lite_llm_model=None,
        executor_model_name="executor-v1" if enabled else None,
        executor_base_url="http://127.0.0.1:8010/v1" if enabled else None,
        executor_fallback_model=None,
        executor_lite_model=None,
        trajectory_root=tmp_path / "captures" if enabled else None,
        executor_context_window=4096 if enabled else None,
        governance=GovernanceConfig(session_root=tmp_path / "sessions"),
        preference_database_path=tmp_path / "preferences.db",
        order_database_path=tmp_path / "orders.db",
        checkpoint_backend="memory",
        redis_enabled=False,
        embedding_cache_enabled=False,
        semantic_cache_enabled=False,
        langfuse_enabled=False,
        tavily_api_key=None,
        category_retriever_backend="local",
    )
    composition.build_container(settings)
    assert captured["model"] is main_model
    assert captured["sub_agent_model"] is (executor_model if enabled else None)
    main_gateway = captured["main_model_middleware"][-1]
    child_gateway = captured["sub_model_middleware"][-1]
    assert main_gateway._semaphore is child_gateway._semaphore
    assert main_gateway._evidence_budget is child_gateway._evidence_budget
    assert main_gateway._external_attempt_accounting is enabled
    if enabled:
        assert main_gateway is not child_gateway
        assert captured["sub_agent_governance_config"].context_window_tokens == 4096
        assert isinstance(
            captured["sub_boundary_middleware"][0], ModelAttemptBudgetMiddleware
        )
        assert isinstance(
            captured["sub_boundary_middleware"][-1], TrajectoryRecorderMiddleware
        )
    else:
        assert main_gateway is child_gateway
        assert captured["sub_boundary_middleware"] == ()


def test_capture_matches_governed_attempt_and_fallback(tmp_path: Path) -> None:
    async def scenario() -> None:
        base = SimpleNamespace(model_name="executor-base")
        fallback = SimpleNamespace(model_name="executor-fallback")
        gateway = ModelGatewayMiddleware(
            max_concurrency=1,
            min_interval_seconds=0,
            max_retries=0,
            fallback_model=fallback,
        )
        recorder = TrajectoryRecorderMiddleware(tmp_path, agent_role="executor")
        scope = ModelCallScopeMiddleware()
        request = ModelRequest(
            model=base,
            messages=[HumanMessage(content="original history")],
            tools=[],
            system_message=SystemMessage(content="system"),
            model_settings={"api_key": "must-not-capture", "temperature": 0},
        )

        async def endpoint(actual: ModelRequest) -> ModelResponse:
            assert actual.messages[0].content == "compressed visible context"
            if actual.model is base:
                raise RuntimeError("service failure with secret-bearing text")
            return ModelResponse(result=[AIMessage(content="recovered")])

        async def governance(actual: ModelRequest) -> ModelResponse:
            governed = actual.override(
                messages=[HumanMessage(content="compressed visible context")]
            )
            return await recorder.awrap_model_call(governed, endpoint)

        async def routing(actual: ModelRequest) -> ModelResponse:
            return await gateway.awrap_model_call(actual, governance)

        await scope.awrap_model_call(request, routing)

    asyncio.run(scenario())
    records = [
        json.loads(p.read_text(encoding="utf-8")) for p in tmp_path.glob("*.json")
    ]
    assert len(records) == 2
    assert len({r["step_id"] for r in records}) == 1
    assert {r["status"] for r in records} == {"error", "success"}
    assert {r["model"] for r in records} == {"executor-base", "executor-fallback"}
    for record in records:
        assert record["messages"][1]["content"] == "compressed visible context"
        assert "must-not-capture" not in json.dumps(record)
        assert "secret-bearing" not in json.dumps(record)
    successful = next(r for r in records if r["status"] == "success")
    assert successful["response"][0]["content"] == "recovered"


def test_executor_gateways_share_concurrency_and_total_budget() -> None:
    async def scenario() -> None:
        budget = EvidenceUsageBudget(max_total_requests=2)
        semaphore = asyncio.Semaphore(1)
        gateways = [
            ModelGatewayMiddleware(
                max_concurrency=1,
                min_interval_seconds=0,
                max_retries=0,
                evidence_budget=budget,
                semaphore=semaphore,
            )
            for _ in range(2)
        ]
        active = 0
        peak = 0

        async def endpoint(_: ModelRequest) -> ModelResponse:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return ModelResponse(result=[AIMessage(content="ok")])

        request = ModelRequest(model=None, messages=[])
        await asyncio.gather(*(g.awrap_model_call(request, endpoint) for g in gateways))
        assert peak == 1
        from app.infrastructure.evidence_budget import EvidenceBudgetExceeded

        with pytest.raises(EvidenceBudgetExceeded):
            await gateways[1].awrap_model_call(request, endpoint)

    asyncio.run(scenario())
