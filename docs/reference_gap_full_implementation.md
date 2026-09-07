# 参考项目能力补齐：完整实现说明

## 1. 本轮目标与结论

本轮以 `reference/globex-agent` 为能力参考，不复制它的技术选型，也不替换当前项目已经更适合自身目标的实现。项目继续采用：

- DDD-lite + Hexagonal/Onion 分层；
- LangChain `create_agent` + LangGraph Checkpointer；
- 同一套图定义 fork 同构子 Agent，以独立 `thread_id` 隔离上下文；
- 自研 Cache Breakpoint/Epoch 会话上下文治理；
- 商品侧 BGE-M3 + FAISS + BGE Reranker；
- 品类知识侧 Markdown 卡片 + OpenSearch Hybrid Pipeline + BGE Reranker；
- FastAPI、SSE、WebSocket、SQLite、Redis 与 Docker。

在此基础上补齐了参考项目存在、当前项目原先完全缺失的能力：跨会话偏好、订单意向、模型/工具运行防护、LangFuse、Redis 跨进程事件、Redis Stream 任务队列、Embedding/语义缓存、持久化 Checkpointer、可选 Web Search、健康检查和容器化交付。

平台仍定位为“跨境购物决策与订单意向 Agent”，不是支付系统：不会扣款、占用真实库存或声称完成真实平台下单。

## 2. 保留而未替换的差异

| 能力 | 当前项目方案 | 为什么保留 |
|---|---|---|
| Agent 编排 | LangGraph 同构主/子 Agent + TaskBoard DAG | 子 Agent 复用已编排图，同时通过独立线程隔离上下文 |
| 上下文治理 | D1-D4、L0-L2、Cache Breakpoint/Epoch、增量摘要、Artifact Offload | 比简单截断或全量摘要更适合前缀缓存和长链任务 |
| 商品召回 | FAISS HNSW + BGE-M3 + 交叉编码重排 | 已有独立领域搜索规格、用户塔和三级降级链 |
| 品类 RAG | 结构化知识卡 + OpenSearch BM25/KNN Pipeline | 只把压缩后的可信卡片交给 Agent，避免原始文档污染上下文 |
| 任务工具 | TaskCreate/Get/List/Update | 已复现依赖 DAG、原子认领和并行就绪判断，无需引入 AgentScope |
| 到手价 | Money、汇率、运费、关税领域规则 | 可测试且与检索框架解耦 |

## 3. 新增能力与落点

### 3.1 跨会话偏好与个性化决策

核心文件：

- `app/domain/buyer/preference.py`：偏好值对象与 `like/dislike` 类型；
- `app/application/memory/ports.py`：`PreferenceStore` 端口；
- `app/application/memory/selector.py`：进入工作记忆的偏好选择策略；
- `app/infrastructure/memory/sqlite_store.py`：SQLite 持久化适配器；
- `app/infrastructure/langchain/tools/preferences.py`：`remember_preference`、`forget_preference` 工具；
- `app/infrastructure/langchain/preference_middleware.py`：模型调用前动态注入偏好；
- `app/infrastructure/retrieval/item_search/user_tower.py`：将持久偏好投影为检索用户塔文本。

读写路径：

```text
用户明确表达偏好
  -> remember_preference / forget_preference
  -> 使用 ContextVar 中可信 buyer_id
  -> SQLite PreferenceStore

后续任意会话
  -> PreferenceMemoryMiddleware 读取 buyer_id
  -> dislike 全保留，like 按上限选择
  -> 注入动态工作记忆
  -> StoredPreferenceProfileSource 同时影响商品检索用户塔
  -> 搜索结果既满足硬约束，也按用户偏好个性化排序
```

偏好没有写入 L0/L1 稳定前缀，避免某个用户的动态偏好破坏 Cache Epoch；语义缓存也把买家和偏好版本纳入命名空间，防止串用旧回答。

### 3.2 订单意向与购物闭环边界

核心文件：

- `app/domain/order/models.py`：`Address`、`OrderLine`、`Order`、状态机；
- `app/application/orders/service.py`：创建、查询、取消用例；
- `app/infrastructure/orders/product_reader.py`：从已构建商品索引读取可信商品快照；
- `app/infrastructure/orders/sqlite_repository.py`：事务、幂等键和持久化；
- `app/infrastructure/langchain/tools/orders.py`：三个订单工具；
- `app/presentation/server.py`：订单查询、取消 REST API。

创建订单意向前必须满足：

1. 当前购物会话已经调用 `item_search`；
2. 用户显式确认；
3. 商品与 SKU 能从索引读到；
4. 目的地、数量、库存快照与金额字段校验通过；
5. 同买家、同幂等键重复提交返回同一订单。

返回结果明确包含 `order_type=purchase_intent`、`inventory_reserved=false` 和 `payment_status=not_supported`。取消只改变意向单状态，不触发退款。

### 3.3 模型网关、工具韧性与 Harness

`app/infrastructure/langchain/reliability_middleware.py` 提供三层保护：

- `ModelGatewayMiddleware`：全局并发信号量、最小请求间隔、可重试异常指数退避、备用模型回退；
- `ToolResilienceMiddleware`：按工具超时、连续失败熔断、恢复窗口、同线程相同参数循环调用拦截；开启 `BREAKER_SHARED` 后由 Redis 在 API/worker 副本间共享熔断状态，并在 Redis 故障时回退本地保护；
- `ToolHarnessMiddleware`：写操作前置证据链、工具 JSON 结果检查、关键字段检查、工具返回中的提示注入过滤。

当前确定性顺序断言：

```text
item_search -> create_order_intent
query_order -> cancel_order
```

超时、熔断、循环或 Schema 错误都会返回结构化错误，模型会看到失败事实，而不是由中间件伪造成业务成功。

`app/application/safety.py` 在应用入口识别明显的系统提示词/密钥窃取指令，并在最终输出前清理疑似密钥。安全拒绝通过 `safety.blocked` 事件显式暴露。

`app/application/agents/drift.py` 按 `shopping_session_id` 维护轻量行为轨迹，每隔固定工具步数检测目标关键词遗忘、连续空召回和 Token 成本突增。该能力默认关闭；开启后只发布 `agent.drift_detected`，不在已经结束的当前轮强行改写回答。

`app/application/runtime/budget.py` 提供一次意图级的共享 Token 账本。主 Agent 和 fork 子任务通过 ContextVar 共享同一对象，模型网关按剩余比例路由：`main > 50%`、`lite 20%-50%`、`minimal 5%-20%`、`fallback <= 5%`。`TOKEN_BUDGET_TOTAL=0` 表示关闭；配置轻量模型后，后两档会切换模型并收紧回答，预算耗尽时停止新增模型推理并发出 `budget.exhausted`。

### 3.4 LangFuse 全链路观测

`app/infrastructure/observability/langfuse.py` 懒加载官方 LangChain `CallbackHandler`。开启后，主 Agent 和 fork 子 Agent 都在 LangGraph 调用配置中携带回调与以下元数据：

- `run_id`；
- `thread_id`；
- `shopping_session_id`；
- `buyer_id`；
- `agent_id`；
- 主/子 Agent 标签。

未设置 `LANGFUSE_ENABLED=true` 时返回空 callback 和空 metadata，不导入 SDK、不发网络请求。密钥只由 LangFuse SDK 从环境读取，不放入 Trace 元数据。

### 3.5 Redis 事件、异步任务和缓存

#### 跨进程事件

`RedisTradeEventBus` 保留进程内 fan-out，同时通过 Redis Pub/Sub 在 API 与 worker 之间广播。每个实例带 origin 标识，避免自己发布的事件经 Redis 回流后重复发送。

#### Redis Stream 任务队列

`RedisStreamTaskQueue` 实现：

- 幂等 enqueue；
- consumer group 消费；
- 完成后 ack 并保存结果；
- 失败计数、重试和 dead-letter stream；
- 任务状态查询。

队列维护普通流和大请求流。同一 `thread_id` 达到 `QUEUE_LARGE_REQUEST_TURNS` 后进入大请求流，worker 优先读取普通流，避免长会话阻塞短请求。

`app/worker.py` 是真实消费入口；API 提供：

- `POST /api/agent/jobs`：提交异步 Agent 任务；
- `GET /api/agent/jobs/{task_id}`：查询状态和结果。

#### Embedding 缓存

`CachedEmbeddingEncoder` 装饰已有 BGE 编码器，以“模型修订 + query/document 类型 + 文本哈希”为 key。Redis 不可用时直接回退原编码器，不影响召回主链。

#### 语义响应缓存

`RedisSemanticResponseCache` 只缓存首轮、只读、无上下文指代的成功回答：

- 订单、支付、取消、偏好写入等请求一律绕过；
- 非首轮线程绕过；
- key 按模型、系统提示词指纹、买家和偏好快照隔离；
- 相似度达到阈值才命中；
- Redis 故障 fail-open，回到 Agent 主链。

### 3.6 持久化 Checkpointer

`LazyAsyncSqliteSaver` 是 LangGraph `AsyncSqliteSaver` 的惰性适配器。默认仍可使用内存 Checkpointer；设置 `CHECKPOINT_BACKEND=sqlite` 后，主图与 fork 子图共享同一数据库连接，但依靠不同 `thread_id` 隔离状态，可在进程重启后恢复会话图状态。

### 3.7 可选 Web Search

`app/application/web_search.py` 定义与供应商无关的端口，`TavilyWebSearch` 是基础设施适配器，`web_search` 是工厂闭包中声明的 `@tool`。只有设置 `TAVILY_API_KEY` 时才注册。

该工具仅用于时效性跨境政策、平台规则等外部信息，不替代商品索引中的价格、库存与 SKU 事实。

### 3.8 事件协议与 API 推送

统一事件类型包括：

- `run.started`、`run.finished`；
- `agent.dispatch`；
- `tool.invoke`、`tool.result`；
- `plan.update`；
- `context.compressed`；
- `cache.hit`；
- `budget.exhausted`、`agent.drift_detected`；
- `safety.blocked`；
- `token.delta`、`final.result`、`error`。

同步请求通过 `POST /api/agent` 返回 SSE；`WS /ws/agent` 可在一个连接内连续执行；`WS /ws/events/{shopping_session_id}` 只订阅该购物会话的事件，不发起 Agent。

## 4. 端到端执行流

```text
HTTP / WebSocket / Redis Stream
  -> AgentRequest 参数校验
  -> ShoppingContextSnapshot
  -> SafetyPolicy 输入校验
  -> 首轮只读语义缓存查询（可选）
  -> ContextVar 绑定可信执行上下文
  -> MainAgent / LangGraph
       -> Cache-aware 上下文投影与压缩
       -> 跨会话偏好动态注入
       -> 模型网关
       -> 主 Agent 自用工具，或 fork 同构子 Agent
       -> Harness + 超时/熔断/循环保护
       -> Task DAG / 检索 / 记忆 / 订单意向
  -> 输出安全清理
  -> SSE / WebSocket / Redis PubSub 逐事件发布
  -> 合格回答写入语义缓存（可选）
```

`shopping_session_id` 用于购物业务事件分区，`thread_id` 用于 LangGraph 对话状态。它们可以一对多，不能互换。ContextVar 只负责当前异步调用链内的可信上下文传播；跨进程时必须把这些字段显式放进任务消息。

## 5. 配置

### 基础必需

```dotenv
LLM_MODEL_NAME=kimi-k2.6
LLM_API_KEY=...
LLM_BASE_URL=...
```

Kimi K2.5/K2.6 不接受任意 temperature，模型工厂会自动省略该参数；离线结构化与上下文压缩会关闭 thinking，避免推理内容占满输出预算。

### 运行防护

```dotenv
MODEL_MAX_CONCURRENCY=50
MODEL_MAX_RETRIES=2
FALLBACK_LLM_MODEL=
TOOL_TIMEOUT_SECONDS=30
TOOL_FAILURE_THRESHOLD=3
TOOL_RECOVERY_SECONDS=30
TOOL_REPEATED_CALL_LIMIT=3
BREAKER_SHARED=false
DRIFT_DETECT_ENABLED=false
DRIFT_CHECK_INTERVAL=3
LITE_LLM_MODEL=
TOKEN_BUDGET_TOTAL=0
```

### Redis、持久化与观测

```dotenv
REDIS_ENABLED=true
REDIS_URL=redis://localhost:6379/0
QUEUE_LARGE_REQUEST_TURNS=30
CHECKPOINT_BACKEND=sqlite
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

可按需启用：

```dotenv
EMBEDDING_CACHE_ENABLED=true
SEMANTIC_CACHE_ENABLED=true
TAVILY_API_KEY=...
```

所有选项、默认值和检索参数以 `.env.example` 为准。

## 6. 安装与运行

### 本地开发

```bash
uv sync --extra search --extra rag --extra platform --extra dev
uv run python scripts/build_product_eval_index.py
uv run python scripts/ingest_category_knowledge.py
uv run uvicorn app.presentation.server:app --reload
```

不启用 Redis、LangFuse、Tavily 时，基础 Agent 使用内存事件总线和本地持久化能力正常运行。

### Docker 全栈

确保 Docker Desktop 已启动，然后执行：

```bash
docker compose -f docker/docker-compose.yml up --build
```

Compose 包含 Redis、OpenSearch、API 和 worker。API/worker 共用 `data` 卷，并通过 Redis 传递任务与事件。

### 健康检查

```text
GET /health        进程存活
GET /health/ready  已启用 Redis、LangFuse、OpenSearch 配置状态
```

### Smoke 与评测

```bash
uv run python scripts/smoke_agent_api.py
uv run python scripts/evaluate_product_recall.py --k 1 3 5 10
uv run python scripts/evaluate_category_recall.py --backend local --k 1 3 5 10
uv run python scripts/evaluate_category_recall.py --backend opensearch --k 1 3 5 10
```

## 7. 验收证据

本轮实际完成：

- `ruff check app scripts tests`：通过；
- `pytest`：86 项通过（包含 Harness 前置链、四档预算和漂移隔离测试）；
- Composition Root：可构建 `MainAgent`、默认内存事件总线；
- 品类知识摄取：5 份 Markdown 并发结构化，生成 37 张卡片；第二次执行 5 份全部按哈希跳过，证明增量复用生效；
- 真实 LLM + FastAPI SSE：收到 `run.started -> token.delta -> final.result -> run.finished`；
- 商品召回 GPU 评测：Recall@5=0.9552、Recall@10=0.9627、MRR@10=0.9322；
- 品类本地词法降级基线：100 条用例，Recall@5=0.4611、MRR@10=0.3481、负例拒答率=1.0；
- Docker Compose 配置解析通过。

品类本地结果只是词法降级基线，不等于 OpenSearch Hybrid 最终结果。当前机器检查时 Docker Desktop daemon 未启动，因此 Redis/OpenSearch 容器级集成与 OpenSearch 最终评测需要在 Docker 启动后执行。

## 8. 已知边界

以下能力被有意留在平台边界外，文档和返回值均不声称已实现：

- 真实 Amazon/Shopee 等平台商品 API；
- 真实库存锁定；
- 支付授权、扣款、退款；
- 真实承运商下单和物流轨迹；
- 生产级身份认证与 `buyer_id` 签发；
- OpenSearch/Redis 的生产 TLS、认证、高可用和备份策略；
- 前端完整购物交互。

这些能力后续应继续通过 application port 接入，不把供应商 SDK 或协议下沉到 domain。

## 9. 主要官方接口依据

- LangGraph 持久化与异步 SQLite Checkpointer：<https://docs.langchain.com/oss/python/langgraph/persistence>
- LangFuse LangChain/LangGraph Callback：<https://langfuse.com/integrations/frameworks/langgraph>
- Tavily Python 异步客户端：<https://docs.tavily.com/sdk/python/quick-start>
- Kimi K2.5/K2.6 thinking 开关：<https://platform.kimi.ai/docs/guide/kimi-k2-6-quickstart>
