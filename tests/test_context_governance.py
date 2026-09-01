"""验证 cache-aware 会话上下文治理的核心安全边界。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.runtime import Runtime

from app.application.context_governance import (
    CompressionPolicyInput,
    CompressionStrategy,
    DeterministicCompressionPolicy,
)
from app.application.runtime import AgentExecutionContext
from app.infrastructure.context_governance.artifact_store import FileArtifactStore
from app.infrastructure.context_governance.breakpoint import CacheBreakpointManager
from app.infrastructure.context_governance.config import GovernanceConfig
from app.infrastructure.context_governance.event_store import SQLiteEventStore
from app.infrastructure.context_governance.messages import (
    collect_protected_event_ids,
    index_messages,
)
from app.infrastructure.context_governance.middleware import (
    ContextGovernanceMiddleware,
    ToolResultMiddleware,
)
from app.infrastructure.context_governance.provider_cache.qwen import (
    QwenExplicitCacheAdapter,
)
from app.infrastructure.context_governance.runtime import sanitize_thread_id
from app.infrastructure.context_governance.schemas import (
    CompressionDelta,
    EpochBaseline,
    TaskDelta,
    TaskState,
)
from app.infrastructure.langchain.main_agent import MainAgent

TEST_SESSION_ROOT = Path("data/test-output/context-governance-tests").resolve()


class FakeCompressor:
    """记录压缩调用并返回覆盖全部候选事件的合法增量。"""

    def __init__(self) -> None:
        self.summary_calls = 0
        self.baseline_calls = 0

    async def summarize_incrementally(
        self,
        *,
        task_state: TaskState,
        working_memory: dict[str, dict[str, Any]],
        candidates: list[Any],
        target_tokens: int,
    ) -> CompressionDelta:
        del task_state, working_memory, target_tokens
        self.summary_calls += 1
        candidate_ids = [item.event_id for item in candidates]
        return CompressionDelta(
            task_state_patch={"current_step": "继续执行"},
            cold_event_ids=candidate_ids,
            compressed_summary="旧对话已经完成，保留其关键结论。",
        )

    async def consolidate_baseline(
        self,
        *,
        previous: EpochBaseline,
        task_state: TaskState,
        task_delta: TaskDelta,
        working_memory: dict[str, dict[str, Any]],
        artifact_refs: list[str],
    ) -> EpochBaseline:
        del previous, task_delta, working_memory
        self.baseline_calls += 1
        return EpochBaseline(
            goal=task_state.goal,
            task_phase=task_state.task_phase,
            constraints=task_state.constraints,
            stable_artifact_refs=artifact_refs,
        )


class ToolCapableFakeModel(BaseChatModel):
    """支持 `bind_tools` 的本地聊天模型，用于完整 AgentLoop 测试。"""

    @property
    def _llm_type(self) -> str:
        return "tool-capable-fake"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ToolCapableFakeModel:
        """接受 Agent 工具定义，但测试回复不会调用工具。"""

        del tools, kwargs
        return self

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """返回固定 AIMessage，避免测试访问外部模型服务。"""

        del messages, stop, run_manager, kwargs
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="治理循环正常"))]
        )


def _config(tmp_path: Path, **overrides: Any) -> GovernanceConfig:
    """创建使用临时会话根目录的测试配置。"""

    values = {
        "context_window_tokens": 10_000,
        "summary_trigger_ratio": 0.70,
        "forced_summary_ratio": 0.85,
        "emergency_ratio": 0.95,
        "target_ratio": 0.60,
        "hot_message_count": 2,
        "min_summary_candidate_tokens": 1,
        "max_compression_input_tokens": 5_000,
        "tool_offload_tokens": 10,
        "session_root": tmp_path,
    }
    values.update(overrides)
    return GovernanceConfig(**values)


def test_breakpoint_hash_changes_only_when_stable_prefix_changes() -> None:
    manager = CacheBreakpointManager()
    baseline = EpochBaseline(goal="比较商品")
    first = manager.build_breakpoints(
        epoch=1,
        system_message=SystemMessage(content="稳定系统提示"),
        tools=[],
        baseline=baseline,
        model_name="qwen-max",
        model_settings={},
    )
    same = manager.build_breakpoints(
        epoch=1,
        system_message=SystemMessage(content="稳定系统提示"),
        tools=[],
        baseline=baseline,
        model_name="qwen-max",
        model_settings={},
    )
    changed = manager.build_breakpoints(
        epoch=1,
        system_message=SystemMessage(content="已经修改的系统提示"),
        tools=[],
        baseline=baseline,
        model_name="qwen-max",
        model_settings={},
    )

    assert manager.validate_breakpoints(
        [item.model_dump(mode="json") for item in first], same
    )
    assert not manager.validate_breakpoints(
        [item.model_dump(mode="json") for item in first], changed
    )


def test_hot_context_protects_complete_tool_group() -> None:
    messages = [
        HumanMessage(content="旧问题"),
        AIMessage(
            content="",
            tool_calls=[{"name": "lookup", "args": {}, "id": "call-1"}],
        ),
        ToolMessage(content="结果", tool_call_id="call-1"),
        HumanMessage(content="当前问题"),
    ]
    indexed = index_messages("thread-1", messages)
    protected = collect_protected_event_ids(indexed, hot_message_count=2)

    # ToolMessage 位于热区时，对应 AI tool call 也必须一起受保护。
    assert indexed[1].event_id in protected
    assert indexed[2].event_id in protected
    assert indexed[3].event_id in protected


def test_thread_id_is_safe_for_windows_session_directories() -> None:
    assert sanitize_thread_id("task-42:child:a") == "task-42_child_a"


def test_policy_uses_llm_summary_only_after_threshold() -> None:
    config = _config(TEST_SESSION_ROOT)
    policy = DeterministicCompressionPolicy(config)
    low = policy.decide(
        CompressionPolicyInput(
            context_usage_ratio=0.50,
            candidate_token_count=3_000,
            candidate_event_ids=["old"],
        )
    )
    high = policy.decide(
        CompressionPolicyInput(
            context_usage_ratio=0.75,
            candidate_token_count=3_000,
            candidate_event_ids=["old"],
        )
    )

    assert low.strategies == [CompressionStrategy.NONE]
    assert CompressionStrategy.INCREMENTAL_SUMMARY in high.strategies


def test_governance_invokes_structured_compressor_and_archives_events() -> None:
    compressor = FakeCompressor()
    config = _config(
        TEST_SESSION_ROOT,
        context_window_tokens=100,
        summary_trigger_ratio=0.20,
        forced_summary_ratio=0.80,
        emergency_ratio=0.95,
        target_ratio=0.10,
    )
    middleware = ContextGovernanceMiddleware(
        compressor=compressor,
        config=config,
        static_system_prompt="系统",
        agent_id="test-agent",
    )
    state = {
        "messages": [
            HumanMessage(content="很早以前的问题 " * 20),
            AIMessage(content="很早以前的回答 " * 20),
            HumanMessage(content="中间问题 " * 10),
            AIMessage(content="中间回答 " * 10),
            HumanMessage(content="当前请求"),
        ]
    }
    runtime = Runtime(context=AgentExecutionContext(thread_id="thread-summary"))

    update = asyncio.run(middleware.abefore_model(state, runtime))

    assert update is not None
    assert compressor.summary_calls == 1
    assert update["compressed_event_ids"]
    assert update["task_state"]["current_step"] == "继续执行"
    events = asyncio.run(
        SQLiteEventStore(TEST_SESSION_ROOT / "thread-summary").list_events()
    )
    assert any(
        event["event_type"] == "compression.incremental_summary" for event in events
    )
    rebuilt = asyncio.run(
        SQLiteEventStore(TEST_SESSION_ROOT / "thread-summary").rebuild_state()
    )
    assert rebuilt["messages"]
    assert rebuilt["task_state"]["current_step"] == "继续执行"
    assert rebuilt["compressed_event_ids"] == update["compressed_event_ids"]


def test_governance_does_not_call_llm_below_threshold() -> None:
    compressor = FakeCompressor()
    middleware = ContextGovernanceMiddleware(
        compressor=compressor,
        config=_config(TEST_SESSION_ROOT, context_window_tokens=100_000),
        static_system_prompt="系统",
        agent_id="test-agent",
    )
    state = {
        "messages": [
            HumanMessage(content="旧问题"),
            AIMessage(content="旧回答"),
            HumanMessage(content="当前请求"),
        ]
    }
    runtime = Runtime(context=AgentExecutionContext(thread_id="thread-low-context"))

    update = asyncio.run(middleware.abefore_model(state, runtime))

    assert update is not None
    assert compressor.summary_calls == 0
    assert compressor.baseline_calls == 0
    assert update["last_governance"]["strategies"] == ["none"]


def test_semantic_invalidation_rolls_epoch_with_llm_baseline() -> None:
    compressor = FakeCompressor()
    middleware = ContextGovernanceMiddleware(
        compressor=compressor,
        config=_config(TEST_SESSION_ROOT, context_window_tokens=100_000),
        static_system_prompt="系统",
        agent_id="test-agent",
    )
    state = {
        "messages": [
            HumanMessage(content="预算是 100 元"),
            AIMessage(content="已记录"),
            HumanMessage(content="纠正一下，预算改成 200 元"),
        ],
        "task_state": TaskState(goal="选商品").model_dump(mode="json"),
        "epoch_baseline": EpochBaseline(goal="选商品").model_dump(mode="json"),
        "token_ledger": {"model_calls": 1},
        "cache_epoch": 0,
    }
    runtime = Runtime(context=AgentExecutionContext(thread_id="thread-correction"))

    update = asyncio.run(middleware.abefore_model(state, runtime))

    assert update is not None
    assert update["cache_epoch"] == 1
    assert compressor.baseline_calls == 1
    assert update["last_semantic_signal_id"] is not None


def test_tool_middleware_offloads_large_result_without_changing_call_id() -> None:
    config = _config(TEST_SESSION_ROOT, tool_offload_tokens=1)
    middleware = ToolResultMiddleware(config=config, agent_id="test-agent")

    @tool
    async def large_tool(query: str) -> str:
        """返回用于测试 D4 卸载的大型文本；query 是测试查询。"""

        return query

    request = ToolCallRequest(
        tool_call={"name": "large_tool", "args": {"query": "x"}, "id": "call-9"},
        tool=large_tool,
        state={"cache_epoch": 2},
        runtime=SimpleNamespace(
            context=AgentExecutionContext(thread_id="thread-tool"),
        ),
    )

    async def handler(_: ToolCallRequest) -> ToolMessage:
        return ToolMessage(content="大型结果" * 100, tool_call_id="call-9")

    result = asyncio.run(middleware.awrap_tool_call(request, handler))

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-9"
    envelope = json.loads(result.text)
    assert envelope["artifact_ref"].startswith("artifacts/")
    stored = asyncio.run(
        FileArtifactStore(TEST_SESSION_ROOT / "thread-tool").get_text(
            envelope["artifact_ref"]
        )
    )
    assert stored == "大型结果" * 100


def test_qwen_cache_adapter_marks_copies_not_original_messages() -> None:
    adapter = QwenExplicitCacheAdapter()
    system = SystemMessage(content="系统提示")
    baseline = SystemMessage(content="稳定基线", id="baseline-1")

    decorated = adapter.decorate_request(
        system_message=system,
        messages=[baseline, HumanMessage(content="动态请求")],
        model_settings={},
        baseline_message_id="baseline-1",
    )

    assert isinstance(system.content, str)
    assert isinstance(baseline.content, str)
    assert decorated.system_message is not None
    assert decorated.system_message.content[0]["cache_control"] == {"type": "ephemeral"}
    assert decorated.messages[0].content[0]["cache_control"] == {"type": "ephemeral"}


def test_main_agent_runs_create_agent_with_context_governance() -> None:
    agent = MainAgent(
        model=ToolCapableFakeModel(),
        tools=[],
        compressor=FakeCompressor(),
        governance_config=_config(TEST_SESSION_ROOT),
    )

    async def collect_events() -> list[dict[str, str]]:
        return [event async for event in agent.run_agent("你好")]

    events = asyncio.run(collect_events())

    assert events[0]["type"] == "run_started"
    assert {event.get("content") for event in events} >= {"治理循环正常"}
    assert events[-1]["type"] == "run_finished"


def test_main_agent_does_not_give_main_only_tools_to_children() -> None:
    @tool
    def shared_tool(value: str) -> str:
        """供主 Agent 和子 Agent 共同使用的测试工具。"""

        return value

    @tool
    def task_control(value: str) -> str:
        """只允许主 Agent 使用的测试控制面工具。"""

        return value

    agent = MainAgent(
        model=ToolCapableFakeModel(),
        tools=[shared_tool],
        main_only_tools=[task_control],
        compressor=FakeCompressor(),
        governance_config=_config(TEST_SESSION_ROOT),
    )

    assert agent.child_tool_names == ("shared_tool",)
    assert set(agent.main_tool_names) == {
        "shared_tool",
        "task_control",
        "fork_sub_agents",
    }
