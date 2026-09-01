# Agent Factory 设计

## 1. 文档目标

本文规划 Globex Agent 平台从当前“主 Agent 内部装配同质子 Agent”演进为“按能力类型集中装配专家 Agent”的方式。

设计吸收 `SearchAgentFactory` 的核心思想：

- 共享稳定的应用服务和基础设施；
- 集中维护每类 Agent 的提示词、工具、模型和 Middleware；
- 主 Agent 可以直接调用安全工具，也可以把复杂任务派发给专家子 Agent；
- 子任务之间必须隔离消息历史、运行状态和上下文治理数据。

本设计不会照搬“每个子任务都重新创建完整 Agent 对象”。当前项目使用 LangGraph，优先保留：

```text
每个 Agent 类型创建一个可复用 Agent 图
每个子任务创建独立 thread_id、run_id、AgentState 和执行上下文
```

## 2. 当前实现

当前装配关系如下：

```text
composition.py
  ├─ 创建 ItemSearchService
  ├─ 创建 CategoryInsightService
  ├─ create_item_search_tool()
  ├─ create_category_insight_tool()
  └─ MainAgent(model, tools, context governance)
          │
          ├─ 创建一个通用 forked_sub_agent 图
          ├─ 给主、子 Agent 注册相同业务工具
          ├─ 给主 Agent 注册 fork_sub_agents
          └─ 创建 main_agent 图
```

### 2.1 当前已经具备的能力

- 工具使用 `@tool` Factory 闭包绑定应用服务；
- 主、子 Agent 共用同一个商品搜索和品类洞察业务实现；
- 子任务使用独立 `thread_id` 和 `run_id`；
- 子任务继承 `ShoppingContextSnapshot`，但不继承主 Agent 消息历史；
- 子 Agent 只向主 Agent 返回最终结论，不回传完整工具轨迹；
- `asyncio.Semaphore` 控制 fork 子任务并发数；
- 上下文治理分别为主 Agent 和子 Agent 创建 Middleware。

### 2.2 当前主要限制

1. 只有一种同质 `forked_sub_agent`，还没有 SearchAgent、TradeAgent 等专家类型。
2. 主、子 Agent 使用同一个系统提示词和同一套工具。
3. `MainAgent.__init__()` 同时负责主 Agent 和子 Agent 的装配，职责会随平台扩展而膨胀。
4. 所有子 Agent 都获得完整工具集，无法实施最小工具权限。
5. `fork_sub_agents` 不能通过 `agent_type` 选择专家 Agent。
6. `should_fork()` 尚未接入真实派发链，实际是否 fork 主要由模型根据 Prompt 判断。
7. 当前并发限制只覆盖 fork 子任务，不是覆盖所有模型调用的全局限流器。
8. 召回服务有内部降级，但没有跨 Agent 共享的工具超时、熔断和半开恢复机制。

## 3. 设计原则

### 3.1 状态隔离，基础设施共享

容器级共享：

- Settings；
- LLM 客户端或模型配置；
- ItemSearchService；
- CategoryInsightService；
- Repository 和外部客户端；
- EventBus；
- 全局模型限流器；
- 熔断状态注册表。

任务级隔离：

- thread_id；
- run_id；
- AgentState；
- 消息历史；
- ReAct 中间过程；
- 上下文治理快照；
- 子 Agent 会话目录。

Factory 中不得保存：

- 当前 buyer_id；
- 当前 shopping_session_id；
- 当前用户查询；
- 当前工具结果；
- 当前子任务状态；
- 任意会话级可变字段。

### 3.2 业务能力与框架装配分离

- Domain 保存电商业务模型和规则；
- Application 保存运行用例、派发计划、Agent 类型和端口；
- Infrastructure 使用 LangChain/LangGraph 装配 Agent、Tool 和 Middleware；
- Composition Root 创建具体实现并完成依赖连接。

LangChain 的 `BaseTool`、`create_agent`、Middleware 等类型不能进入 Application 或 Domain。

### 3.3 工具单一来源

同一种业务工具必须由一个 Factory 方法统一创建，避免主 Agent 与子 Agent 的实现分叉：

```text
MainAgent 直接检索 ──────┐
                         ├──> SearchAgentFactory.build_shared_tools()
SearchAgent 子任务检索 ──┘               │
                                         v
                              同一个 Application UseCase
```

### 3.4 最小工具权限

- MainAgent 只直接持有安全、常用、低成本工具；
- SearchAgent 只持有检索和知识工具；
- TradeAgent 只持有其交易阶段允许的工具；
- 有副作用的工具不能因为方便而加入所有 Agent 的公共工具集。

### 3.5 复用 Agent 图，按线程隔离状态

LangGraph 编译图主要描述执行结构，消息状态由 Checkpointer 按 `thread_id` 管理。因此默认采用：

```text
一个 SearchAgent 图
  ├─ thread search-a：任务 A 的状态
  ├─ thread search-b：任务 B 的状态
  └─ thread search-c：任务 C 的状态
```

仅在以下情况下改为每次派发重新 `create_agent()`：

- Middleware 保存无法消除的请求级可变状态；
- Agent 图会在运行时修改自身节点或工具清单；
- 第三方 Runtime 明确声明 Agent 实例不可并发复用；
- 安全审计要求每个任务拥有物理独立实例。

## 4. 目标结构

```text
app/
├─ application/
│  └─ agents/
│     ├─ orchestration.py          # fork/dispatch 业务策略
│     ├─ dispatch_models.py        # AgentType、DispatchTask、DispatchPlan
│     ├─ registry_ports.py         # 专家 Runtime 注册表端口
│     └─ run_agent.py              # 主 Agent 运行用例
│
├─ infrastructure/
│  └─ langchain/
│     ├─ agents/
│     │  ├─ search_factory.py      # SearchAgent 工具和图装配
│     │  ├─ trade_factory.py       # TradeAgent 工具和图装配
│     │  ├─ main_factory.py        # MainAgent 图装配
│     │  └─ registry.py            # AgentType -> 专家 Runtime
│     ├─ sub_agents/
│     │  ├─ runtime.py             # 独立线程执行、并发和重试
│     │  └─ dispatch_tool.py       # @tool task_dispatch
│     ├─ tools/
│     │  ├─ product_search.py
│     │  ├─ category_insight.py
│     │  └─ ...
│     └─ prompts/
│        ├─ main.py
│        ├─ search.py
│        └─ trade.py
│
└─ composition.py                 # 创建共享依赖、Factory 和容器
```

第一阶段只实现 `SearchAgentFactory`，`TradeAgentFactory` 保留扩展位置，不创建空实现。

## 5. 核心对象职责

### 5.1 SearchAgentFactory

建议职责：

```python
class SearchAgentFactory:
    def build_shared_tools(self) -> list[BaseTool]: ...
    def build_subagent_tools(self) -> list[BaseTool]: ...
    def build_runtime(self) -> ForkedAgentLoop: ...
```

保存的依赖：

- 商品搜索应用服务；
- 品类洞察应用服务；
- 模型或模型 Provider；
- 上下文治理配置和 Compressor；
- Checkpointer Provider；
- 后续加入的全局限流器与工具熔断注册表。

方法边界：

- `build_shared_tools()` 创建 MainAgent 和 SearchAgent 都可以使用的安全检索工具；
- `build_subagent_tools()` 创建只允许 SearchAgent 使用的工具；
- `build_runtime()` 创建使用搜索 Prompt、搜索工具和搜索 Middleware 的可复用 Agent 图。

### 5.2 MainAgentFactory

建议职责：

- 接收已经装配好的专家 Factory 或专家注册表；
- 组合 MainAgent 可以直接调用的工具；
- 注册 `task_dispatch`；
- 创建主 Agent 专属 Middleware；
- 创建并返回 `MainAgent` Runtime。

MainAgentFactory 不负责创建：

- ItemSearchService；
- OpenSearch 客户端；
- Repository；
- EventBus；
- 全局限流器；
- 熔断注册表。

这些共享依赖继续由 `composition.py` 创建。

### 5.3 SubAgentRegistry

注册表维护：

```text
AgentType.SEARCH -> SearchAgent Runtime
AgentType.TRADE  -> TradeAgent Runtime
```

它只负责根据类型找到 Runtime，不负责让模型自行拼装 Agent。

未知 Agent 类型必须返回稳定的参数错误，不能静默退回全工具 Agent。

### 5.4 task_dispatch 工具

目标签名示意：

```python
@tool
async def task_dispatch(
    agent_type: Literal["search", "trade"],
    tasks: list[str],
    reason: Literal["parallel", "context_isolation", "deep_chain"],
) -> str:
    ...
```

职责：

1. 校验并规范化任务；
2. 根据 `agent_type` 查找专家 Runtime；
3. 为每个子任务创建独立执行上下文；
4. 按全局和 Agent 类型并发限制执行；
5. 只返回子任务最终结论；
6. 发布子任务开始、完成和失败事件。

工具仍必须遵守仓库规范：使用 `@tool`，通过 Factory 闭包注入 Registry 和策略依赖。

## 6. 目标运行链

### 6.1 简单检索

```text
用户
  -> MainAgent
  -> item_search/category_insight
  -> Application UseCase
  -> MainAgent 汇总
```

不创建额外子 Agent，减少延迟和 Token 消耗。

### 6.2 复杂检索

```text
用户
  -> MainAgent
  -> task_dispatch(agent_type="search")
  -> SearchAgent Runtime
       ├─ 独立 thread_id
       ├─ 独立 AgentState
       ├─ 搜索专属 Prompt
       └─ 搜索专属工具
  -> 仅返回最终结论
  -> MainAgent 汇总
```

### 6.3 并行检索

```text
MainAgent
  -> task_dispatch(search, [任务 A, 任务 B, 任务 C])
       ├─ SearchAgent / thread A
       ├─ SearchAgent / thread B
       └─ SearchAgent / thread C
  -> asyncio.gather 按输入顺序汇总
```

## 7. Prompt 策略

### 7.1 MainAgent Prompt

负责：

- 理解用户总目标；
- 判断简单任务直接执行还是派发；
- 拆分可独立子任务；
- 选择 Agent 类型；
- 汇总专家结果；
- 不重复执行已经由专家完成的工作。

### 7.2 SearchAgent Prompt

负责：

- 标准化检索词；
- 判断 `category_insight` 在商品搜索前还是后调用；
- 执行商品召回、比较和补充检索；
- 遵守预算、配送地、币种等硬约束；
- 返回自包含的检索结论和必要证据。

SearchAgent 不负责：

- 创建或取消订单；
- 修改用户长期偏好；
- 决定其他 Agent 的任务；
- 向用户输出最终跨领域回复。

## 8. fork 决策治理

当前 `should_fork()` 尚未进入实际运行链。调整后采用两层判断：

### 8.1 模型提议

MainAgent 根据 Prompt 判断：

- `parallel`：两个及以上任务彼此独立，可同时运行；
- `context_isolation`：中间结果庞大、敏感或不应污染主上下文；
- `deep_chain`：预计需要不少于三层调用或多轮检索改写。

### 8.2 确定性策略批准

`OrchestrationPolicy` 根据结构化参数进行校验：

- tasks 非空；
- 任务去重；
- 并行理由原则上要求不少于两个任务；
- 单次任务数不能超过配置上限；
- 每条任务长度不能超过限制；
- Agent 类型必须已注册；
- 有副作用的任务不能错误派发给只读 SearchAgent。

模型负责语义判断，策略负责可确定验证。策略不依赖 LangChain。

## 9. 并发、限流和熔断

### 9.1 分层并发

需要区分：

1. `task_dispatch` 的子任务并发上限；
2. 单个 Agent 类型的并发上限；
3. 全平台 LLM 请求并发上限；
4. 单一供应商的请求间隔与 429 退避。

当前 `ForkedAgentLoop` 的 Semaphore 可以继续承担第 1 或第 2 层，但不能代替全局 LLM 闸门。

### 9.2 全局模型闸门

后续增加共享 `GatewayThrottle`：

```text
MainAgent 模型调用 ──────┐
SearchAgent 模型调用 ────┼──> GatewayThrottle
TradeAgent 模型调用 ─────┤
上下文压缩模型调用 ──────┘
```

闸门必须由 `composition.py` 创建并注入，不能由每个 Factory 单独创建。

### 9.3 工具熔断

后续增加共享 `CircuitBreakerRegistry`，统一处理：

- 工具超时；
- 连续失败；
- 熔断窗口；
- 半开探测；
- 恢复；
- 失败事件记录。

召回内部的 embedding/reranker/关键词三级降级继续保留。业务降级与平台工具熔断解决的是不同层次的问题。

## 10. 上下文治理

每类 Agent 都通过同一个 Middleware Factory 创建上下文治理组件，但使用不同 `agent_id` 和 Prompt：

```text
main_agent   -> main Prompt   -> main context namespace
search_agent -> search Prompt -> search context namespace
trade_agent  -> trade Prompt  -> trade context namespace
```

子 Agent：

- 继承 `ShoppingContextSnapshot`；
- 不继承主 Agent 的全量聊天历史；
- 只接收自包含任务描述和明确选择的偏好；
- 使用独立 session directory；
- 只返回最终答案和允许回传的结构化元数据。

## 11. 迁移计划

### 阶段一：抽离 SearchAgentFactory

1. 新建 `infrastructure/langchain/agents/search_factory.py`。
2. 将检索工具清单从 `composition.py` 收敛到 `build_shared_tools()`。
3. 将当前 `ForkedAgentLoop.create()` 的搜索子 Agent 装配移入 `build_runtime()`。
4. 新增搜索专属 Prompt。
5. 保持商品搜索和品类洞察应用服务不变。

完成标准：主 Agent 直接检索和 fork 检索仍调用相同用例，现有功能无回归。

### 阶段二：引入 Registry 与 task_dispatch

1. 在 Application 定义 `AgentType`、`DispatchTask` 和 `DispatchPlan`。
2. 新建 Infrastructure Registry。
3. 将 `fork_sub_agents` 演进为支持 `agent_type` 的 `task_dispatch`。
4. 将 `should_fork()` 或其升级策略真正接入派发批准链。
5. 保留旧工具名一段兼容期，确认无调用后再删除。

完成标准：MainAgent 可以明确派发给 SearchAgent，未知类型和非法计划会被拒绝。

### 阶段三：抽离 MainAgentFactory

1. 将主工具组合、主 Middleware 和 `create_agent()` 移出 `MainAgent.__init__()`。
2. `MainAgent` 只保留 Runtime 适配和流式输出职责。
3. Composition Root 创建 SearchAgentFactory、Registry 和 MainAgentFactory。

完成标准：增加新专家 Agent 时不需要修改 `MainAgent` Runtime 实现。

### 阶段四：加入平台级韧性组件

1. 增加共享 GatewayThrottle；
2. 增加 CircuitBreakerRegistry；
3. 增加工具超时和熔断 Middleware；
4. 将主 Agent、专家 Agent 和压缩模型纳入统一模型请求预算；
5. 补充 EventBus 子 Agent 与工具级事件。

完成标准：并发上限和熔断状态不会因 Factory 或 Agent 实例数量而倍增。

### 阶段五：增加第二类专家 Agent

只有在交易应用用例和权限模型稳定后才增加 TradeAgentFactory：

1. 定义交易专属 Prompt；
2. 明确只读和有副作用工具边界；
3. 增加确认、幂等和恢复机制；
4. 注册到 SubAgentRegistry；
5. 增加跨 SearchAgent/TradeAgent 编排测试。

## 12. 测试计划

### 12.1 Factory 测试

- SearchAgentFactory 始终包含商品搜索与品类洞察工具；
- 连续构建的工具包装对象引用同一应用服务；
- SearchAgent 不包含交易工具；
- SearchAgent 使用搜索专属 Prompt 和 agent_id；
- Factory 不保存请求级状态。

### 12.2 派发测试

- `agent_type=search` 能找到 SearchAgent Runtime；
- 未注册类型会返回参数错误；
- parallel 任务少于两个时被策略拒绝；
- 空任务、重复任务和过多任务被正确处理；
- 并行返回顺序与输入任务顺序一致；
- 子任务异常不会泄漏其他任务上下文。

### 12.3 隔离测试

- 每个子任务生成不同 thread_id 和 run_id；
- 子任务继承相同 shopping_session_id 和 buyer_id；
- 子任务不读取主 Agent 消息历史；
- 两个并发子任务不会互相读取工具结果；
- 子 Agent 只返回最终结论；
- ContextVar 在异常后正确 reset。

### 12.4 共享资源测试

- 主 Agent 和 SearchAgent 调用同一个 Application UseCase；
- 所有 Agent 共享同一个 GatewayThrottle；
- 同一工具的熔断状态跨 Agent 可见；
- 每个 Agent 类型复用自己的图，但不复用线程状态；
- 多会话并发不会把 Factory 变成请求状态容器。

### 12.5 架构测试

- Domain 不导入 LangChain/LangGraph；
- Application 不导入 LangChain/LangGraph；
- Agent Factory 只位于 Infrastructure；
- Composition Root 是具体依赖的唯一装配入口。

## 13. 不在本次设计中的内容

- 不立即实现 TradeAgent；
- 不引入 Agent 对象池；
- 不让子 Agent 共享主 Agent 全量历史；
- 不把 LangChain Agent Factory 放入 Application；
- 不为了使用 Factory 名称而创建没有替换价值的抽象接口；
- 不改变 ItemSearch 和 CategoryInsight 的领域模型及召回算法；
- 不用 Factory 代替 ContextVar、Checkpointer 或 EventBus。

## 14. 风险与应对

| 风险 | 应对方式 |
| --- | --- |
| Factory 变成新的大杂烩 | 每类 Agent 一个 Factory，外部资源仍由 Composition Root 创建 |
| MainAgent 与 SearchAgent 工具再次分叉 | 公共工具只从 `build_shared_tools()` 获取 |
| 复用 Agent 图导致状态串扰 | 强制唯一 thread_id，并保留并发隔离测试 |
| Middleware 引入可变实例字段 | Code Review 和并发测试；必要时改为每任务创建 Middleware |
| 模型错误选择 Agent 类型 | Registry 白名单和确定性 DispatchPolicy 校验 |
| 全局限流被多个 Factory 倍增 | GatewayThrottle 只在 Composition Root 创建一次 |
| 交易工具权限泄漏 | 按 Agent 类型建立工具白名单，默认拒绝未知工具 |
| 重构期间旧接口中断 | 分阶段迁移，旧 fork 工具保留短期兼容入口 |

## 15. 验收标准

完成本设计的核心重构后，应满足：

1. MainAgent 不再直接创建搜索子 Agent 图。
2. SearchAgent 的工具、Prompt、Middleware 和图由一个 Factory 集中装配。
3. 简单搜索仍可由 MainAgent 直接调用工具。
4. 复杂搜索可通过 `task_dispatch(agent_type="search")` 派发。
5. 主、子 Agent 使用同一个商品搜索和品类洞察应用服务。
6. 每个子任务使用独立 thread_id、run_id、AgentState 和会话目录。
7. SearchAgent 不拥有未授权工具。
8. 新增专家 Agent 不需要修改 MainAgent Runtime 核心逻辑。
9. Domain 和 Application 继续保持框架无关。
10. 现有 Agent、fork、上下文治理、ItemSearch 和 CategoryInsight 测试全部通过。

## 16. 最终决策

本项目采用 Factory 的核心思想，但按 LangGraph 和六边形架构进行调整：

```text
采用：
  专家 Agent 集中装配
  工具单一来源
  按 Agent 类型派发
  状态隔离、基础设施共享
  全局限流和熔断依赖注入

保留：
  可复用 LangGraph Agent 图
  独立 thread_id/checkpoint/context
  @tool Factory 闭包
  Composition Root 统一装配

不照搬：
  每个任务强制重新创建 Agent 图
  在 Application 层直接依赖 Agent 框架
  为模式名称创建无实际用途的抽象层
```

一句话总结：

> 用 Factory 统一“每种专家 Agent 如何构建”，用 Registry 决定“任务交给哪种专家 Agent”，用独立 thread/checkpoint/context 保证“每个子任务互不污染”，同时让 Application 和 Domain 保持框架无关。
