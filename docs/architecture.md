# Globex Agent 架构

## 定位

Globex Agent 是一个以电商为业务背景的 Agent 平台。平台核心领域是 Agent
Runtime、Orchestration、Context Governance、Tool 和 Session；Catalog 等模块是
平台当前承载的电商垂直领域。

架构采用 DDD-lite + Hexagonal/Onion Architecture，核心目标不是目录命名，而是
保护 Runtime Kernel 不被 LangGraph、模型供应商、Redis 或具体存储绑定。

## 依赖方向

```text
Presentation ──→ Application ──→ Domain
       │                ↑           ↑
       └──→ Composition │           │
                        └── Infrastructure
```

- `domain` 只能依赖标准库和自身模块。
- `application` 只能依赖 Domain 和 Application Ports。
- `infrastructure` 实现端口，可以依赖第三方框架。
- `presentation` 只负责 HTTP、SSE、WebSocket 和 DTO。
- `composition.py` 是唯一装配具体模型、工具、存储和 Runtime Driver 的入口。

## 当前代码结构

```text
app/
├── domain/
│   ├── runtime/             # Agent、Run、Thread 生命周期
│   ├── orchestration/       # fork 和路由策略
│   ├── context/             # 压缩、Breakpoint、Epoch 决策
│   ├── tools/               # 框架无关的工具契约
│   ├── session/             # Checkpoint 语义和仓储端口
│   └── catalog/             # 当前已落地的电商商品实体
├── application/
│   ├── agents/              # RunAgent 用例
│   ├── catalog/             # 商品搜索用例和端口
│   └── ports/               # TaskQueue 等出站端口
├── infrastructure/
│   ├── context_governance/  # LangChain Middleware、SQLite、D4、缓存适配
│   ├── langchain/           # Agent Driver 和 @tool 适配器
│   ├── retrieval/           # BGE、FAISS、用户向量适配器
│   ├── context.py           # ContextVar
│   ├── llm.py               # ChatModel 工厂
│   └── settings.py          # 环境配置
├── presentation/            # FastAPI、SSE、WebSocket、DTO
├── composition.py           # Composition Root
└── worker.py                # 后台任务消费者入口
```

## Runtime 与 LangGraph 的边界

Domain 拥有 `AgentRun`、`RunStatus`、fork 条件、工具契约和治理决策。
Infrastructure 中的 LangGraph Driver 负责把这些语义适配到 `AgentState`、
`BaseMessage`、`BaseTool`、Checkpointer 和节点调度。

任何新增规则如果回答的是“平台应该怎么决定”，优先进入 Domain；如果回答的是
“某个框架或供应商怎样执行”，则进入 Infrastructure。

## 商品检索边界

- `domain.catalog.Product` 是纯商品实体。
- `application.catalog.ItemSearchService` 编排召回和重排端口。
- BGE、FAISS 和用户向量 Provider 位于 Infrastructure。
- `item_search` 的 `@tool` 函数只是 LangChain 入站适配器。

## 尚未实现

以下模块是目标架构的一部分，但目前没有伪造空实现：

- Redis Stream `TaskQueue` 和独立 Worker 消费循环；
- Redis/PostgreSQL Checkpoint Repository；
- Qdrant 向量索引适配器；
- 平台级 EventBus、Tracing、Throttle 和 Circuit Breaker；
- Order、Shipping、Pricing、Buyer Preference 等电商领域；
- RAG KnowledgeBase 和 embedding/semantic cache。

这些能力应先定义 Domain/Application 语义与端口，再添加 Infrastructure 实现。

## 架构回归

`tests/test_architecture.py` 使用 AST 检查依赖方向。新增代码不得通过延迟导入、
类型字符串或兼容模块绕过约束；如果确实需要新的边界，应先修改本文件并说明原因。
