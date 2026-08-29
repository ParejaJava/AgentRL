# Globex Agent 架构调整说明

## 1. 调整目标

本项目的核心定位是“服务于电商业务的 Agent 平台”。因此采用 DDD-lite 与
Hexagonal/Onion Architecture，但不把所有重要代码都称为 Domain：

- `domain` 只表达电商业务事实、实体、值对象和业务规则；
- `application` 负责 Agent 运行用例、编排策略、上下文治理决策和端口；
- `infrastructure` 负责 LangChain/LangGraph、模型、ContextVar、事件总线、检索和持久化实现；
- `presentation` 负责 FastAPI、SSE、WebSocket 与 DTO；
- `composition.py` 是唯一装配具体实现的组合根。

这次调整的重点不是改目录名，而是保证依赖始终指向内层：

```text
Presentation ─┐
              ├──> Application ───> Domain
Infrastructure┘          ▲
                         │ ports
              Infrastructure adapters
```

`application` 不导入 LangChain、LangGraph、FastAPI、Redis、Pydantic 等具体框架；
`domain` 既不导入框架，也不导入 Application、Infrastructure 或 Presentation。

## 2. 调整后的目录职责

```text
app/
├── domain/
│   └── catalog/                    # 商品实体及电商业务不变量
├── application/
│   ├── agents/
│   │   ├── run_agent.py            # Run 生命周期与事件发布用例
│   │   └── orchestration.py        # fork 条件和 ForkPlan
│   ├── catalog/                    # 商品搜索用例及其端口
│   ├── context_governance/         # 确定性压缩决策，不直接调用 LLM
│   ├── events/                     # TradeEvent 稳定契约
│   ├── runtime/                    # Run、执行上下文、Checkpoint 模型
│   ├── tools/                      # 框架无关的工具契约
│   └── ports/                      # EventPublisher、CheckpointStore、TaskQueue
├── infrastructure/
│   ├── context.py                  # ContextVar 载体
│   ├── context_governance/         # LangChain Middleware、SQLite、压缩器适配
│   ├── eventbus/                   # 进程内事件总线实现
│   ├── langchain/                  # create_agent、@tool、主/子 AgentLoop
│   ├── retrieval/                  # BGE、FAISS 等检索实现
│   ├── llm.py                      # 模型创建与供应商配置
│   └── settings.py                 # 环境配置
├── presentation/                   # FastAPI、SSE、WebSocket、DTO
└── composition.py                  # 唯一装配入口
```

以后新增 `order`、`shipping`、`pricing`、`buyer` 等电商能力时，应先判断它是否包含
独立于 Agent 框架仍然成立的业务规则；只有满足这一点才进入 `domain`。

## 3. 原模块迁移关系

| 调整前 | 调整后 | 原因 |
|---|---|---|
| `domain/runtime` | `application/runtime` | Run 是平台用例的执行状态，目前不是电商领域聚合 |
| `domain/orchestration` | `application/agents/orchestration.py` | fork 是 Agent 用例编排策略 |
| `domain/context` | `application/context_governance` | 压缩决策属于应用策略，LLM/Middleware 实现仍在基础设施层 |
| `domain/tools` | `application/tools` | 工具契约是平台能力契约，不是电商实体 |
| `domain/session` | `application/runtime` 与 `application/ports` | Checkpoint 是运行恢复模型及存储端口 |
| `domain/catalog` | 保持不变 | Product 和商品约束是真实电商领域概念 |

旧 `app.domain.runtime`、`app.domain.orchestration`、`app.domain.context`、
`app.domain.tools` 和 `app.domain.session` 已移除，避免新代码继续从错误边界导入。

## 4. 三种标识必须分离

| 标识 | 生命周期 | 用途 | 是否被子 Agent 继承 |
|---|---|---|---|
| `shopping_session_id` | 一段购物对话 | 业务关联、事件分区、WebSocket 订阅 | 是 |
| `thread_id` | 一条 Agent 消息历史 | LangGraph Checkpointer、上下文治理文件目录 | 否，fork 后重新生成 |
| `run_id` | 一次执行 | 追踪、错误定位、运行生命周期 | 否，fork 后重新生成 |

主 Agent 收到 `RunAgentCommand` 后创建 `run_id` 和 `AgentExecutionContext`。
fork 子 Agent 复制其中的 `ShoppingContextSnapshot`，但创建全新的 `thread_id` 与
`run_id`，因此既能共享买家身份、地区和币种，又不会读取主 Agent 或其他子 Agent 的
消息历史。

## 5. ShoppingContext 与 ContextVar

`ShoppingContextSnapshot` 位于 Application，因为它是入口、Worker、Agent 和工具共同理解的
数据契约。`ContextVar` 位于 Infrastructure，因为它只是 Python 进程内的上下文传递机制。

运行链路如下：

```text
AgentRequest
  -> RunAgentCommand
  -> AgentExecutionContext
  -> MainAgent.stream 绑定 ContextVar
  -> LangGraph / @tool / fork 子 Agent 读取当前上下文
  -> finally reset ContextVar
```

`ShoppingContext.require_current()` 在工具确实依赖购物身份时使用。它不会返回
`anonymous`，因为静默共享默认会话会产生跨用户串台风险。

ContextVar 只在同一 Python 进程的调用链内传播。将任务放入 Redis Stream 或其他 Worker
队列时，必须把 `ShoppingContextSnapshot` 序列化到任务载荷，Worker 取出后重建并绑定；
不能假设 ContextVar 自动跨进程传递。

## 6. Agent Runtime 和事件流

`RunAgent` 应用用例现在是运行生命周期的负责人：

1. 校验消息并创建 `AgentRun`；
2. 发布 `run.started`；
3. 调用 `AgentRuntimePort.stream`；
4. 把框架输出转换成 `token.delta`；
5. 发布 `final.result` 与 `run.finished`；
6. 捕获运行异常并发布 `error`。

`MainAgent` 只负责把上述端口适配到 LangChain `create_agent` 和 LangGraph，不再拥有平台
事件模型。这样未来替换 Agent 框架时，接口层和应用事件不需要跟着改变。

`MainAgent.run_agent(message)` 暂时保留为兼容入口，但新代码应调用
`RunAgent.execute(RunAgentCommand(...))`。它不是 LangChain 弃用 API，而是本项目自己的旧入口；
待所有外部调用迁移后可以删除。项目继续使用 `langchain.agents.create_agent`，不使用已经被
替代的 `create_react_agent`。

## 7. EventBus 设计

`TradeEvent` 和 `EventPublisher` 位于 Application；`InMemoryTradeEventBus` 是
Infrastructure 的单进程实现。每个订阅者拥有独立的有界 `asyncio.Queue`，按
`shopping_session_id` 隔离。慢 WebSocket 队列满时丢弃最旧的过程事件，避免反向阻塞
AgentLoop。

目前提供三种输出：

- `POST /api/agent`：SSE 返回本次运行事件；
- `WS /ws/events/{shopping_session_id}`：独立订阅某个购物会话的过程事件；
- `WS /ws/agent`：保留在同一连接上提交请求并读取事件的便捷入口。

进程内总线不能跨 API/Worker 进程。接入 Redis Pub/Sub 时，应新增 EventBus
基础设施适配器，并通过 `EventPublisher` 端口替换，不能让 Application 直接导入 Redis。
跨进程事件还应携带 `event_id` 和来源进程标识，以便去重并避免背板回环广播。

## 8. 上下文治理边界

上下文治理被拆成两个职责：

- Application 的 `DeterministicCompressionPolicy` 只根据 token 水位、候选事件、语义纠正、
  阶段变化和 cache epoch 指标作出决策，本身不调用 LLM；
- Infrastructure 的 LangChain Middleware 收集框架状态、执行压缩器、写 SQLite Event Log、
  构造 cache-aware Prompt，并把结果交回 LangGraph。

因此“何时压缩”可以独立测试，“怎样从 LangChain 消息压缩并持久化”仍由适配器处理。
Cache Breakpoint 仍由 `infrastructure/context_governance/breakpoint.py` 构建，因为当前实现依赖
具体 Prompt 投影和模型消息格式；如果未来形成框架无关的 Breakpoint 值对象，再逐步上移。

## 9. 已实现与尚未实现

本次已经实现：

- Domain 收缩到真实电商领域；
- Agent Runtime、fork 和压缩决策迁入 Application；
- 显式 ShoppingContext 与三类 ID 分离；
- 主/子 Agent 的上下文继承和隔离；
- 会话级 TradeEvent 契约与进程内 EventBus；
- SSE、直接 WebSocket 和独立事件订阅入口；
- 架构依赖测试、EventBus 测试、RunAgent 测试和 fork 隔离测试。

仍未实现：

- Redis Stream 任务队列与独立 Worker 消费循环；
- Redis Pub/Sub EventBus 背板；
- CheckpointStore 的 SQL/Redis 适配器及进程重启恢复；
- API 身份认证，以及 buyer/session/order 的授权校验；
- Order、Shipping、Pricing、Buyer 等完整电商领域模型；
- Agent 事件与上下文治理内部事件的统一追踪/可观测性投影。

这些空缺保留为端口或明确的后续能力，不在 Domain 中用框架对象提前占位。

品类知识工具已经按照相同边界实现：`domain/catalog` 保存知识模型，Application 通过
`CategoryKnowledgeRetriever` 端口聚合，Infrastructure 负责 Markdown 结构化、JSONL
缓存和 LangChain `@tool`。OpenSearch 尚未配置，当前使用可替换的本地检索适配器；详见
[`category_insight.md`](category_insight.md)。

## 10. 约束与验证

`tests/test_architecture.py` 通过 AST 检查依赖方向：

- Domain 不能导入框架和任何外层模块；
- Application 不能导入 Infrastructure、Presentation 或具体框架；
- LangChain 工具继续统一使用 `langchain_core.tools.tool` 的 `@tool` 声明；
- 工具需要运行时依赖时，使用工厂闭包创建被装饰函数。

新增模块时应先写端口与应用用例，再在 Infrastructure 中实现适配器，最后只在
`composition.py` 装配。这样框架变化只影响外层，电商规则和 Agent 用例保持稳定。
