# 会话级 Cache-Aware 上下文治理设计方案

> 状态：已完成第一版实现，本文同时作为实现基线与评审文档  
> 适用范围：单次会话内的主 AgentLoop 与 fork 子 AgentLoop  
> 技术栈：LangChain `create_agent`、LangChain Middleware、LangGraph Checkpointer、OpenAI-compatible / Qwen / Anthropic 等模型接口  
> 核心目标：在保证 Agent 状态正确性、可恢复性与上下文质量的前提下，利用 Prompt Cache 降低重复 Prefill 成本，并通过分层治理控制长会话上下文增长。

---

## 1. 设计结论

本方案采用四个核心原则：

1. **原始事件、会话状态、模型输入三者分离**：Event Log 是事实源，Session State 是可恢复状态，Prompt Projection 是每次模型调用时生成的临时视图。
2. **上下文采用三代结构，而不是简单的 prefix/suffix 二分**：
   - `L0 Static Root`：长期稳定的系统提示词、工具定义、Harness 规则；
   - `L1 Epoch Baseline`：一个 cache epoch 内稳定的任务基线快照；
   - `L2 Active Working Set`：每轮变化、持续治理的动态工作集。
3. **Prompt Cache 不能凌驾于上下文质量之上**：同一个 epoch 内冻结 L0/L1，但当状态失效、任务阶段变化、冻结上下文债务过高或缓存收益下降时，必须执行 `ROLL_CACHE_EPOCH`。
4. **Context Governance Boundary 与 Provider Cache Breakpoint 分离**：上下文治理由应用层控制，provider-specific cache breakpoint 仅作为推理层优化，不作为业务语义边界。

---

## 2. 背景与问题

随着 ReAct AgentLoop 持续运行：

```text
Model
  ↓
Tool
  ↓
Model
  ↓
Tool
  ↓
Model
```

会不断累积：

- 用户消息；
- 模型消息；
- AI tool call；
- ToolMessage；
- 大型网页、文件、搜索结果；
- 中间计划；
- 已完成但仍保留在消息历史中的旧信息。

如果每轮模型调用都发送完整历史，会带来：

- 输入 token 持续增长；
- Prefill 成本和延迟上升；
- 大型 Tool Result 挤占上下文窗口；
- 过时历史干扰当前推理；
- Tool Call / ToolMessage 配对被简单裁剪破坏；
- 主 Agent 与 fork 子 Agent 发生上下文污染；
- 修改早期 Prompt 导致 Prompt Cache 命中率下降；
- 长期冻结 Prefix 会形成不可回收的 `Frozen Context Debt`。

因此，上下文治理不能只是“超过阈值后做一次 summary”，而应作为 Agent Runtime 的常驻机制。

---

## 3. 非目标

第一阶段不处理：

- 跨用户、跨会话长期记忆；
- 用户画像；
- 基于向量数据库的长期记忆召回；
- 多设备同步；
- AgentScope 或其他框架接管 AgentLoop；
- 直接让压缩模型修改完整消息历史；
- 复杂语义压缩算法作为第一阶段依赖。

---

## 4. 三视图模型

系统维护三个互相独立但可映射的视图：

```text
完整事件日志 Event Log
        │
        ├── 生成 Session State
        │
        └── 生成 Prompt Projection
```

### 4.1 Event Log

完整、追加式、不可变记录：

- 用户消息；
- AIMessage；
- Tool Call；
- ToolMessage；
- Error；
- Compression Event；
- Epoch Roll Event；
- Artifact 引用；
- Cache Metrics。

用途：

- 审计；
- 回放；
- 重建 Session State；
- 失败恢复；
- 质量评估。

原则：

```text
Event Log 可以增长，但默认不直接进入 Prompt。
```

---

### 4.2 Session State

由 LangGraph Checkpointer 管理，是当前 AgentLoop 的权威可恢复状态。

包含：

- D2 结构化任务状态；
- D3 工作记忆；
- cache epoch；
- breakpoint 元数据；
- 压缩元数据；
- D4 引用；
- token ledger；
- context version。

---

### 4.3 Prompt Projection

每次 Model Call 前临时构造，不持久化为第二套消息状态。

原则：

```text
Event Log != Session State != Prompt Projection
```

Prompt Projection 只包含当前模型真正需要看到的最小必要上下文。

---

# 5. 四层语义上下文

| 层级 | 名称 | 典型内容 | 是否进入 Prompt | 治理方式 |
|---|---|---|---|---|
| D1 | Hot Context | 当前请求、最近必要消息、未闭合工具调用、当前错误 | 必须 | 原样保护 |
| D2 | Structured Task State | goal、constraints、plan、current_step、completed_steps | 必须 | 结构化 patch |
| D3 | Working Memory | 已验证事实、阶段结论、失败路径、有效边界 | 按需 | upsert/delete |
| D4 | Cold Data | 原始 Tool Result、完整 trace、文件、历史过程 | 默认不进入 | artifact/event 引用 |

---

## 5.1 D1：Hot Context

D1 是当前一步推理必须直接看到的信息：

- 当前用户输入；
- 当前任务步骤；
- 最近一次必要 AIMessage；
- 尚未闭合的 tool call；
- 与当前 tool call 对应的 ToolMessage；
- 当前错误；
- 当前重试约束。

硬规则：

- 不交给 LLM 自由摘要；
- 不可被普通 pruning 删除；
- Tool Call / ToolMessage 必须成组保留；
- 当前用户请求永远保留。

---

## 5.2 D2：Structured Task State

推荐 Pydantic 模型：

```python
from pydantic import BaseModel, Field

class TaskState(BaseModel):
    goal: str
    constraints: list[str] = Field(default_factory=list)
    current_plan: list[str] = Field(default_factory=list)
    completed_steps: list[str] = Field(default_factory=list)
    current_step: str | None = None
    failed_tools: list[str] = Field(default_factory=list)
    pending_questions: list[str] = Field(default_factory=list)
```

D2 不允许被自然语言 summary 取代，只允许：

```text
old state
   +
validated patch
   ↓
new state
```

---

## 5.3 D3：Working Memory

保存本会话后续仍可能需要的信息：

- 已验证事实；
- 用户已确认约束；
- 阶段性结论；
- 不应重复尝试的失败路径；
- 对后续步骤持续有效的边界；
- 关键 artifact 引用。

更新方式：

```text
upsert
delete
```

禁止每轮对整个 D3 从头总结。

---

## 5.4 D4：Cold Data

包括：

- 原始网页；
- 大型搜索结果；
- 文件内容；
- 被压缩的历史消息；
- 子 Agent 完整执行轨迹；
- 调试 trace；
- 大型 Tool Result。

Prompt 中只保留：

```text
summary
preview
artifact_ref
重新读取方式
```

---

# 6. 三代 Cache-Aware 上下文结构

最终 Prompt 不采用单一：

```text
prefix | suffix
```

而采用：

```text
┌──────────────────────────────────────┐
│ L0 — Static Root                     │
│ tools / system / harness rules       │
└──────────────────────────────────────┘
                 ↓
         Cache Breakpoint #1
                 ↓
┌──────────────────────────────────────┐
│ L1 — Epoch Baseline                  │
│ stable task/session baseline         │
└──────────────────────────────────────┘
                 ↓
         Cache Breakpoint #2
                 ↓
┌──────────────────────────────────────┐
│ L2 — Active Working Set              │
│ D2 delta / D3 / D1 / tool events     │
└──────────────────────────────────────┘
```

---

## 6.1 L0：Static Root

典型内容：

- System Prompt；
- Tool definitions；
- Tool 顺序；
- Tool JSON Schema；
- Harness policy；
- Skill instructions；
- 稳定 agent identity；
- 不随当前任务变化的规则。

特点：

- 长期稳定；
- 极少修改；
- 适合长期 Prompt Cache；
- 修改意味着高成本 cache invalidation。

禁止放入：

- 当前时间；
- request UUID；
- 当前 token 数；
- 动态 task state；
- Tool Result；
- 当前任务进度。

---

## 6.2 L1：Epoch Baseline

一个 cache epoch 内保持不变的任务基线快照，例如：

```text
goal
confirmed constraints
baseline plan
completed_before_epoch
validated facts
stable artifact refs
```

L1 不是永久冻结，只是在当前 epoch 内稳定。

Epoch Roll 时：

```text
旧 L1
+
L2 已验证增量
+
最新 D2/D3
    ↓
Consolidation
    ↓
New L1
```

---

## 6.3 L2：Active Working Set

持续变化且允许治理：

- D2 当前 delta；
- D3 当前相关 working memory；
- 最近消息；
- 当前 Tool Call；
- Tool Result Envelope；
- 当前错误；
- 当前用户请求。

治理发生在这里：

- clear；
- offload；
- incremental summary；
- prune；
- emergency trim。

---

# 7. Context Boundary 与 Cache Breakpoint 分离

必须区分两个概念：

## 7.1 Context Governance Boundary

业务语义层概念：

```text
哪些内容当前允许压缩？
哪些内容属于稳定 baseline？
什么时候应该重建 baseline？
```

由应用自己决定。

---

## 7.2 Provider Cache Breakpoint

模型 Provider 的推理优化：

```text
在哪些位置创建 prefix cache marker？
哪些 token 命中 provider cache？
```

OpenAI、Anthropic、Qwen 的实现方式可能不同。

因此不要：

```text
CacheBreakpointManager
    直接写死 OpenAI/Qwen 参数
```

而要：

```text
Context Governance
        │
        ▼
Cache Layout
        │
        ▼
Provider Cache Adapter
```

---

# 8. Cache Epoch 生命周期

一个 epoch：

```text
Epoch N
──────────────────────────

L0 Static Root
    ↓
BP1

L1 Baseline Vn
    ↓
BP2

L2 delta
L2 delta
L2 delta
...
```

普通 Model/Tool 调用只治理 L2。

当满足 roll 条件：

```text
ROLL_CACHE_EPOCH
```

执行：

```text
1. 冻结当前业务动作
2. 识别需要保留的 D1/D2/D3
3. 将旧 L2 原始事件归档到 D4/Event Log
4. merge old baseline + validated delta
5. 生成 New Epoch Baseline
6. 清空或最小化 Active Delta
7. epoch += 1
8. 创建新的 provider cache layout
9. 继续 AgentLoop
```

---

# 9. Epoch Roll 触发条件

不能只看 token ratio。

推荐同时考虑五类信号。

## 9.1 Size Pressure

例如：

```text
prefix_tokens / context_window
suffix_tokens / context_window
total_context_usage_ratio
```

---

## 9.2 Semantic Invalidation

强触发：

- 用户纠正了旧事实；
- 旧事实被工具结果证伪；
- 旧约束被撤销；
- 任务目标变化；
- System/Harness 规则更新；
- Prompt Injection 被发现；
- 已缓存信息不再可信。

正确性优先于 Cache Hit。

---

## 9.3 Task Phase Change

例如：

```text
research
   ↓
implementation
   ↓
testing
   ↓
final synthesis
```

明确阶段切换适合新建 epoch。

---

## 9.4 Compaction Churn

如果 L2 最近 N 轮频繁反复压缩：

```text
compress
grow
compress
grow
compress
```

说明旧 baseline 已不再合理，应 roll。

---

## 9.5 Cache Economics

评估：

```text
cache saving
vs
frozen context cost
vs
context quality degradation
```

当旧 prefix 虽然 cache hit，但长期占用大量窗口且语义价值低时，应主动 roll。

---

# 10. Base + Delta 状态语义

必须避免：

```text
L1:
current_step = search

L2 D2:
current_step = analyze
```

同时进入 Prompt 造成冲突。

建议使用：

```text
Epoch Baseline
+
Current Delta
```

而不是把完整 D2 同时重复两遍。

例如：

```python
class EpochBaseline(BaseModel):
    goal: str
    constraints: list[str]
    baseline_plan: list[str]
    completed_before_epoch: list[str]

class TaskDelta(BaseModel):
    current_step: str | None
    newly_completed_steps: list[str]
    new_constraints: list[str]
    failed_tools: list[str]
```

Epoch Roll：

```text
Baseline V3
+
Delta
   ↓
Merge
   ↓
Baseline V4

Delta = empty
```

---

# 11. Cache Breakpoint 数据结构

```python
from datetime import datetime
from pydantic import BaseModel

class CacheBreakpoint(BaseModel):
    epoch: int
    layer: str
    message_id: str | None
    semantic_hash: str
    provider_projection_hash: str
    prefix_token_count: int
    created_at: datetime
```

推荐至少维护两个 hash。

---

## 11.1 Semantic Hash

作用：

```text
Epoch Baseline 的业务语义有没有变化？
```

允许 canonical serialization。

推荐：

```python
orjson + hashlib.sha256
```

---

## 11.2 Provider Projection Hash

作用：

```text
实际发给 provider 的 prefix 是否变化？
```

需要尽量接近真实 request payload：

- model；
- system/developer content；
- tools；
- tool order；
- JSON schema；
- cache config；
- stable message blocks；
- reasoning/model settings。

不要只对“规范化后的业务对象”做 hash，否则可能出现：

```text
Local manager: prefix unchanged
Provider: cache miss
```

---

# 12. 自适应压缩策略

```python
from enum import Enum

class CompressionStrategy(str, Enum):
    NONE = "none"
    CLEAR_TOOL_RESULTS = "clear_tool_results"
    OFFLOAD_ARTIFACTS = "offload_artifacts"
    INCREMENTAL_SUMMARY = "incremental_summary"
    SEMANTIC_PRUNE = "semantic_prune"
    ROLL_CACHE_EPOCH = "roll_cache_epoch"
```

第一阶段只实现：

- `NONE`
- `CLEAR_TOOL_RESULTS`
- `OFFLOAD_ARTIFACTS`

第二阶段：

- `INCREMENTAL_SUMMARY`

后续：

- `SEMANTIC_PRUNE`

`ROLL_CACHE_EPOCH` 从设计上必须存在，但第一版可以由规则触发，不必一开始交给 LLM 自动决定。

---

# 13. 推荐策略顺序

```text
上下文较小
    └── NONE

大型 Tool Result
    └── OFFLOAD_ARTIFACTS

已闭合且原文不重要的 Tool Result
    └── CLEAR_TOOL_RESULTS

历史语义重要，但逐条消息不重要
    └── INCREMENTAL_SUMMARY

普通长文本仍过大
    └── SEMANTIC_PRUNE

baseline 已失效 / phase change / frozen debt 过高
    └── ROLL_CACHE_EPOCH
```

---

# 14. Compression Policy

输入：

```python
class CompressionPolicyInput(BaseModel):
    suffix_event_ids: list[str]
    suffix_token_count: int
    tool_result_token_count: int
    open_tool_call_ids: list[str]
    task_phase: str
    context_usage_ratio: float
    cache_epoch_age: int
    repeated_compactions: int
    semantic_invalidated: bool
    available_strategies: list[CompressionStrategy]
```

输出：

```python
class CompressionDecision(BaseModel):
    strategies: list[CompressionStrategy]
    target_event_ids: list[str]
    preserve_event_ids: list[str]
    target_tokens: int
    should_roll_epoch: bool
    reason: str
```

推荐：

```text
规则优先
+
LLM 辅助
```

流程：

```text
1. deterministic guard
2. 明显场景直接决策
3. 模糊场景调用轻量模型
4. 结构化输出
5. deterministic validation
6. apply
7. post-check
8. rollback on failure
```

---

# 15. Incremental Summary 协议

禁止：

```text
把全部历史重新交给模型总结
```

只输入：

```text
current D2
+
current D3
+
new closed events after last compaction
```

输出：

```python
class CompressionDelta(BaseModel):
    task_state_patch: dict[str, object]
    working_memory_upserts: list["WorkingMemoryItem"]
    working_memory_delete_ids: list[str]
    hot_context_keep_ids: list[str]
    cold_event_ids: list[str]
    compressed_summary: str
```

应用前：

```text
Pydantic validation
+
Tool pairing validation
+
protected event validation
+
token reduction validation
```

---

# 16. Tool Result 治理

Tool Result 必须尽量在进入长期 Prompt 之前治理。

推荐分类：

| Tool Result | 行为 |
|---|---|
| 小型结构化结果 | 原样进入 D1 |
| 中型文本 | 暂时进入 D1，可后续清理 |
| 大型文本/网页 | 立即写入 D4 |
| 文件/二进制 | 只保留元数据和路径 |
| 重复结果 | 差异化保留 |
| Error | 保留错误类型、关键详情、重试建议 |

统一 Envelope：

```python
class ToolResultEnvelope(BaseModel):
    status: str
    summary: str
    preview: str | None = None
    artifact_ref: str | None = None
    original_token_count: int
    visible_token_count: int
```

重要规则：

> 未治理的大型 Tool Result、临时 trace、错误 dump 不允许进入新的 Epoch Baseline。

---

# 17. Prompt Projection

最终模型输入：

```text
[L0 Static Root]
  System Prompt
  Tool Definitions
  Harness Rules

      ↓ provider cache breakpoint #1

[L1 Epoch Baseline]
  Stable Task Snapshot
  Stable Confirmed Facts
  Stable Constraints

      ↓ provider cache breakpoint #2

[L2 Active Working Set]
  D2 Delta
  Relevant D3
  D1 Hot Context
  Current Tool Result Envelope
  Current User Request
```

原则：

- L0 一个 epoch 甚至多个 epoch 内保持稳定；
- L1 一个 epoch 内稳定；
- L2 每轮可治理；
- D1 靠近模型输出端；
- provider cache marker 由 Adapter 添加；
- Prompt Projection 不持久化。

---

# 18. LangChain / LangGraph 接入设计

Agent 仍使用：

```python
from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools=FULL_TOOL_SET,
    state_schema=SessionAgentState,
    middleware=[
        ToolResultMiddleware(),
        ContextGovernanceMiddleware(),
        CacheMetricsMiddleware(),
    ],
    checkpointer=checkpointer,
)
```

---

## 18.1 ContextGovernanceMiddleware

推荐基于：

```python
AgentMiddleware
awrap_model_call()
ModelRequest.override()
```

伪代码：

```python
class ContextGovernanceMiddleware(AgentMiddleware):

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler,
    ) -> ModelResponse:

        state = request.state

        metrics = token_counter.inspect(request)

        governance_view = breakpoint_manager.inspect(
            state=state,
            request=request,
        )

        protected = hot_context_detector.collect(
            request.messages
        )

        decision = await policy_router.decide(
            state=state,
            metrics=metrics,
            protected=protected,
        )

        governed_state = await strategy_executor.apply(
            state=state,
            decision=decision,
        )

        projection = projection_builder.build(
            state=governed_state,
            request=request,
        )

        decorated = cache_provider_adapter.decorate(
            projection
        )

        new_request = request.override(
            messages=decorated.messages,
            system_message=decorated.system_message,
            model_settings=decorated.model_settings,
        )

        return await handler(new_request)
```

---

## 18.2 ToolResultMiddleware

基于：

```python
awrap_tool_call()
```

职责：

```text
Tool
 ↓
result
 ↓
token/type inspection
 ↓
small → return
large → D4
 ↓
ToolResultEnvelope
```

必须保持：

```text
tool_call_id
```

不变。

---

## 18.3 CacheMetricsMiddleware

基于：

```text
aafter_model
```

记录：

- input tokens；
- output tokens；
- cached input tokens；
- cache write tokens（provider 支持时）；
- epoch；
- cache breakpoint；
- latency；
- compression cost。

动态 metrics 禁止写入 L0/L1。

---

# 19. Session State

LangGraph State 推荐使用 `TypedDict`，领域模型使用 Pydantic。

```python
from langchain.agents.middleware import AgentState

class SessionAgentState(AgentState):
    task_state: dict
    task_delta: dict
    working_memory: dict
    cache_epoch: int
    cache_breakpoints: list[dict]
    compressed_event_ids: list[str]
    cold_event_refs: list[str]
    pending_compression_request: dict | None
    token_ledger: dict
    context_version: int
```

推荐分工：

```text
LangGraph State
→ TypedDict

D2 / D3 / CompressionDecision / CompressionDelta
→ Pydantic
```

---

# 20. Checkpointer

第一阶段：

```python
from langgraph.checkpoint.memory import InMemorySaver
```

开发阶段足够。

进入真实实验后建议：

```text
AsyncSqliteSaver
```

生产/多实例：

```text
AsyncPostgresSaver
```

原则：

```text
不要自己实现 Checkpointer。
```

---

# 21. Event Log

Event Log 与 LangGraph Checkpoint 职责分离。

推荐第一版：

```text
SQLite + aiosqlite
```

事件表：

```text
event_id
thread_id
agent_id
sequence
event_type
message_id
tool_call_id
payload
artifact_ref
cache_epoch
created_at
```

以 append-only 为主。

---

# 22. D4 Artifact Store

第一阶段：

```text
pathlib
+
aiofiles
```

目录：

```text
data/sessions/
└── {main_thread_id}/
    ├── events.db
    ├── artifacts/
    │   ├── tool_001.txt
    │   ├── page_002.html
    │   └── result_003.json
    └── sub_agents/
        └── {sub_thread_id}/
            ├── events.db
            └── artifacts/
```

后续如需对象存储，再抽象：

```python
class ArtifactStore(Protocol):
    async def put(...): ...
    async def get(...): ...
```

不建议第一阶段引入 Milvus/FAISS 等向量数据库。

---

# 23. Provider Cache Adapter

定义：

```python
from typing import Protocol

class CacheProviderAdapter(Protocol):

    def decorate_request(
        self,
        request,
        cache_layout,
    ):
        ...

    def extract_metrics(
        self,
        response,
    ):
        ...

    def supports_explicit_cache(self) -> bool:
        ...
```

实现：

```text
provider_cache/
├── base.py
├── openai.py
├── qwen.py
└── anthropic.py
```

职责：

- 添加 provider-specific cache breakpoint；
- 设置 cache options；
- 读取 cached token metrics；
- 处理 TTL；
- 屏蔽不同 provider 参数差异。

---

# 24. Token 统计

热路径推荐：

```python
from langchain_core.messages.utils import (
    count_tokens_approximately,
)
```

用于：

```text
是否治理
是否压缩
是否进入 emergency mode
```

实际指标使用 Provider 返回的 usage。

如果 Qwen 需要精确 token：

```text
transformers.AutoTokenizer
```

OpenAI 模型可按需：

```text
tiktoken
```

---

# 25. Emergency Fallback

`trim_messages()` 不作为主治理器，仅作为最后兜底。

```text
normal governance
    ↓ fail

deterministic cleanup
    ↓ fail

D1 + D2 + minimum D3
    ↓ still too large

trim_messages()
```

并且必须保证：

- Tool Call / ToolMessage 合法配对；
- 当前请求保留；
- 未闭合 Tool Call 保留；
- 系统提示词保留。

---

# 26. 推荐阈值

第一阶段可保留相对阈值：

| Context Ratio | Action |
|---|---|
| `< 50%` | `NONE`，只做零成本清理 |
| `50%–70%` | Tool cleanup / artifact offload |
| `70%–85%` | incremental summary |
| `85%–95%` | 强制压缩 |
| `>=95%` | emergency mode |

但：

```text
ROLL_CACHE_EPOCH
```

不能只依赖 token ratio。

Semantic invalidation / phase change 可以在任意 token ratio 下触发。

---

# 27. fork 子 AgentLoop

每个 fork 子 Agent：

- 独立 `thread_id`；
- 独立 Session State；
- 独立 Event Log；
- 独立 D4；
- 独立 Epoch Baseline；
- 独立 cache epoch；
- 可以共享 L0 Static Root；
- 不共享可变 L1/L2；
- 只向主 Agent 返回结构化最终结果或 artifact 引用。

推荐：

```text
Main
thread_id = task-42:main

├── Child A
│   thread_id = task-42:child:a
│
├── Child B
│   thread_id = task-42:child:b
│
└── Child C
    thread_id = task-42:child:c
```

注意：

```text
独立 thread
≠
无法共享 Prompt Cache
```

因为子 Agent 可以拥有：

```text
相同 L0 Static Root
+
不同 L1/L2
```

因此仍可共享公共 Prompt Prefix 的 provider cache。

---

# 28. 子 Agent 返回协议

不要让主 Agent读取子 Agent checkpoint 作为通信机制。

推荐：

```python
class ChildAgentResult(BaseModel):
    task_id: str
    status: str
    summary: str
    evidence_refs: list[str]
    artifact_refs: list[str]
    errors: list[str]
```

流程：

```text
Child internal checkpoint
        ↓
Child AgentLoop
        ↓
ChildAgentResult
        ↓
Main Agent State
```

Checkpoint 用于恢复，Result Schema 用于 Agent 间通信。

---

# 29. 可观测性

每轮治理至少记录：

```text
thread_id
agent_id
cache_epoch
L0 tokens
L1 tokens
L2 tokens
total prompt tokens
cached input tokens
cache write tokens
compression strategy
tokens before
tokens after
compression ratio
provider cache hit ratio
artifact offload count
epoch roll reason
semantic invalidation
governance latency
business model latency
compression model cost
```

关键指标：

```text
compression_ratio
= tokens_after / tokens_before

cache_hit_ratio
= cached_input_tokens / prompt_tokens

frozen_context_ratio
= (L0 + L1 tokens) / context_window

governance_cost_ratio
= governance_model_cost / business_model_cost

hot_context_retention_rate
= protected_events_kept / protected_events_total
```

建议新增：

```text
epoch_roll_frequency
semantic_invalidation_count
repeated_compaction_count
cache_savings_per_epoch
```

---

# 30. 异常与回退

## 30.1 Provider Prefix Mismatch

如果：

```text
provider_projection_hash mismatch
```

则：

- 禁止认为 cache 仍有效；
- 记录 mismatch；
- 查找 tools/schema/system/model settings 变化；
- 必要时重新创建 cache layout。

---

## 30.2 Compression Model Failure

- 不提交 patch；
- Session State 保持原样；
- 优先做 deterministic cleanup；
- 必要时进入 emergency mode。

---

## 30.3 Compression Inflation

如果：

```text
tokens_after >= tokens_before
```

则：

- 丢弃结果；
- 恢复原状态；
- 记录 `compression_inflation`。

---

## 30.4 Tool Pair Corruption

若 Tool Call / ToolMessage 配对损坏：

- 拒绝 apply；
- 从 Event Log 重建完整组；
- 整组加入 D1 protect set。

---

## 30.5 Context Overflow

执行：

```text
1. emergency cleanup
2. D1 + D2 + minimum D3
3. retry same model request once
4. still fail → raise explicit error
```

---

## 30.6 Semantic Invalidation

发现旧 baseline 错误时：

```text
correctness > cache hit
```

立即：

```text
append explicit correction
+
roll epoch at earliest safe point
```

重大错误可立即强制 roll。

---

# 31. 推荐 Python 依赖

核心：

```text
langchain
langchain-core
langgraph
langchain-openai
langgraph-checkpoint-sqlite

pydantic
orjson
aiosqlite
aiofiles
tenacity
langsmith
```

按需：

```text
transformers   # Qwen 精确 tokenizer
tiktoken       # OpenAI token counting
```

第一阶段不建议：

```text
LLMLingua
FAISS
Milvus
额外 Agent 框架
复杂 Storage Framework
```

---

# 32. 推荐代码结构

```text
app/
├── agent/
│   ├── main_agent.py
│   ├── llm.py
│   └── sub_agents/
│       └── fork.py
│
├── compress/
│   ├── __init__.py
│   ├── schemas.py
│   ├── token_counter.py
│   ├── breakpoint.py
│   ├── epoch.py
│   ├── policy.py
│   ├── strategies.py
│   ├── projection.py
│   ├── middleware.py
│   ├── event_store.py
│   ├── artifact_store.py
│   ├── metrics.py
│   │
│   └── provider_cache/
│       ├── base.py
│       ├── openai.py
│       ├── qwen.py
│       └── anthropic.py
│
└── tools/
    └── context_tools.py
```

---

# 33. 单次 Model Call 流程

```text
1. 接收 Event
2. Event Log append
3. ToolResultMiddleware 治理大结果
4. 读取 LangGraph Session State
5. Token Counter 统计 L0/L1/L2
6. 校验 Epoch / Breakpoint / Hash
7. 识别 D1 Protect Set
8. 检测 Semantic Invalidation
9. PolicyRouter 决策
10. StrategyExecutor 执行治理
11. 必要时 Roll Epoch
12. 更新 D2/D3/D4 refs
13. PromptProjectionBuilder 生成 L0/L1/L2
14. ProviderCacheAdapter 添加 cache markers
15. ModelRequest.override()
16. 调用业务模型
17. 解析 provider usage/cache metrics
18. Event Log append
19. LangGraph Checkpointer 保存 State
```

---

# 34. 分阶段实施

## Phase 1：确定性治理

实现：

- Session State；
- Event Log；
- L0/L1/L2；
- CacheEpoch；
- Provider-independent Breakpoint Manager；
- Prompt Projection；
- Tool Result Offload；
- Token Metrics；
- `NONE`；
- `CLEAR_TOOL_RESULTS`；
- `OFFLOAD_ARTIFACTS`。

目标：

```text
不依赖压缩模型也能稳定运行。
```

---

## Phase 2：结构化 Incremental Summary

实现：

- CompressionDelta；
- Pydantic structured output；
- D2 patch；
- D3 upsert/delete；
- rollback；
- compression evaluation。

---

## Phase 3：Epoch Roll

实现：

- Base + Delta merge；
- Semantic Invalidation；
- Task Phase Change；
- Frozen Context Ratio；
- repeated compaction；
- New Epoch Baseline；
- provider cache re-layout。

---

## Phase 4：自主策略路由

实现：

```text
deterministic rules
+
small LLM decision
```

允许 LLM 推荐策略，但关键安全规则由 deterministic validator 控制。

---

## Phase 5：高级优化

评估：

- semantic prune；
- LLMLingua-2；
- provider cache TTL；
- 多 breakpoint placement；
- 不同 summary model；
- cache hit / context quality tradeoff；
- benchmark & replay evaluation。

---

# 35. 验收标准

## 正确性

- 当前 User Request 永不丢失；
- 未闭合 Tool Call 永不丢失；
- Tool Call / ToolMessage 配对合法；
- D2 只通过 validated patch 更新；
- D4 原始内容可恢复；
- Semantic Invalidation 能触发 baseline 更新；
- 压缩失败不破坏 Event Log。

---

## Cache Correctness

- L0 在预期周期内严格稳定；
- L1 在同一 epoch 内严格稳定；
- provider projection hash 可验证；
- cache marker 不污染业务状态；
- cache miss 可观测；
- cache hit 不作为正确性依赖。

---

## Context Quality

- 不允许无限增长 Frozen Prefix；
- stale baseline 可以被 epoch roll 替换；
- D1 永远保留；
- D3 可淘汰；
- Context Governance 可以在低 token 使用率时因 semantic invalidation 主动触发。

---

## 隔离性

- Main/Child 独立 thread；
- Child 不共享可变 Session State；
- Child 完整轨迹不进入 Main Prompt；
- Agent 间只交换结构化 Result；
- 可共享 Static Root cache。

---

## 性能

- 正常轮次只做轻量治理检查；
- 低使用率不调用 compression LLM；
- 大 Tool Result 不进入长期 Prompt；
- 可以观测 cache savings；
- 压缩收益大于治理成本。

---

# 36. 最终设计原则

整套架构可以浓缩为：

```text
Event Log
    = 完整事实

LangGraph Checkpoint
    = 可恢复状态

D1
    = 当前绝不能丢的信息

D2
    = 结构化任务状态

D3
    = 可更新工作记忆

D4
    = 冷数据与原始材料

L0 Static Root
    = 长期稳定公共前缀

L1 Epoch Baseline
    = 一个 epoch 内稳定的任务基线

L2 Active Working Set
    = 每轮持续治理的动态区域

Prompt Cache
    = 推理层成本优化

Epoch Roll
    = 防止冻结上下文债务并刷新语义基线

thread_id
    = Agent 状态隔离

checkpoint
    = Agent 执行恢复

ChildAgentResult
    = Agent 间通信
```

最重要的工程原则：

> **缓存命中率是优化指标，不是系统正确性的约束。**

> **同一个 cache epoch 内保持稳定前缀，只治理动态 Working Set；当语义失效、阶段变化或冻结上下文成本过高时，主动牺牲一次局部 cache hit，通过 Epoch Roll 重建更短、更准确的 Baseline。**

> **Agent Runtime 负责状态，Provider Cache 负责 Prefill 复用，两者必须解耦。**

---

# 37. 当前实现映射

当前代码已实现：

| 设计能力 | 实现位置 |
|---|---|
| 纯确定性压缩策略和 Epoch Roll 决策 | `app/domain/context/` |
| D1～D4 适配 DTO、Session State、CompressionDelta | `app/infrastructure/context_governance/schemas.py` |
| L0/L1 双 breakpoint 与双哈希 | `app/infrastructure/context_governance/breakpoint.py` |
| 结构化增量摘要和 Baseline 合并 | `app/infrastructure/context_governance/compressor.py` |
| Tool Result 清理和压缩结果应用 | `app/infrastructure/context_governance/strategies.py` |
| Prompt Projection | `app/infrastructure/context_governance/projection.py` |
| SQLite Event Log 与状态重建 | `app/infrastructure/context_governance/event_store.py` |
| D4 Artifact Store | `app/infrastructure/context_governance/artifact_store.py` |
| Qwen 显式缓存和通用隐式缓存 | `app/infrastructure/context_governance/provider_cache/` |
| LangChain Model/Tool Middleware | `app/infrastructure/context_governance/middleware.py` |
| 主 AgentLoop Driver | `app/infrastructure/langchain/main_agent.py` |
| fork 领域规则 | `app/domain/orchestration/policy.py` |
| fork 子 Agent LangGraph Driver | `app/infrastructure/langchain/sub_agents/fork.py` |

第一版有意不实现：

- 策略选择 LLM；
- `SEMANTIC_PRUNE`；
- LLMLingua；
- 跨会话长期记忆；
- 自动向量召回；
- 让业务 Agent 主动决定 Epoch Roll。

这些内容不属于当前确认的 LLM 职责边界。`SEMANTIC_PRUNE` 枚举仍保留为后续扩展点，但当前确定性策略不会选择它。
