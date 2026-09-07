# 参考项目缺失能力审计与实现计划

> 实施状态（2026-09-05）：Phase 1-6 已完成。完整代码映射、运行方式、验收结果与剩余边界见 [`reference_gap_full_implementation.md`](reference_gap_full_implementation.md)。

反向审计补项也已完成：Silent-Drift 事件检测、单次意图四档 Token 预算、Redis 普通/大请求双流优先级，以及跨 API/worker 实例共享熔断状态。

## 1. 目标与约束

本文记录 `reference/globex-agent` 与当前项目的能力差异，并作为后续实现与验收清单。

实现遵守以下约束：

- 保留当前 Python、LangChain/LangGraph、FastAPI、FAISS、OpenSearch、SQLite、Redis、WebSocket 和 Docker 技术路线；
- 保留当前 DDD-lite + Hexagonal 分层、同构 fork 子 Agent、Task DAG、Cache-aware 上下文治理和两套检索实现；
- 不引入 AgentScope，不用 Qdrant 替换 FAISS/OpenSearch，不用参考项目的固定 SearchAgent/TradeAgent 替换当前同构 Worker；
- 只新增当前完全缺失的能力，或为已有端口补充尚未接通的适配器；
- 所有 LangChain 工具使用 `langchain_core.tools.tool` 的 `@tool` 声明；
- `.env` 中的密钥只由配置对象读取，不写入日志、文档、测试快照或 Git。

## 2. 已有能力：不替换

| 能力 | 当前实现 | 决策 |
|---|---|---|
| 主/子 Agent 编排 | LangGraph `create_agent` + 同构 fork Worker + 独立 thread | 保留 |
| 任务管理 | SQLite TaskBoard、状态机、依赖 DAG、原子认领 | 保留 |
| 会话上下文治理 | D1-D4、L0-L2、Cache Breakpoint/Epoch、增量摘要、Artifact Offload | 保留 |
| 商品检索 | BGE-M3 + FAISS HNSW + BGE Reranker + 词法降级 | 保留 |
| 品类知识 RAG | Markdown 结构化知识卡 + OpenSearch Hybrid Pipeline + BGE Reranker | 保留 |
| 到手价 | 当前 Money、汇率、运费和关税规则 | 保留 |
| API 流式接口 | FastAPI + SSE/WebSocket | 保留并扩展事件种类 |

## 3. 确认缺失的能力

### 3.1 业务能力

1. **跨会话买家偏好**
   - 缺少 Preference 领域模型和持久化端口；
   - 缺少 remember/forget 工具；
   - 缺少按 `buyer_id` 读取、筛选并注入 Agent 的读路径；
   - 商品检索虽已有 User Tower 端口，但 Composition Root 当前使用空信号提供器。

2. **订单意向单**
   - 缺少 Order/OrderLine/Address 和状态机；
   - 缺少创建、查询、取消用例及工具；
   - 缺少订单仓储和 REST 查询/取消接口；
   - 参考项目同样没有真实支付和物流，本项目只实现可审计的订单意向单，不宣称完成支付。

3. **可选联网检索**
   - 缺少用于时效性跨境政策/平台规则查询的 Web Search 端口和工具；
   - 仅在显式配置供应商密钥后装配，未配置时不注册工具。

### 3.2 Agent 运行保障

4. **模型网关治理**
   - 缺少统一并发闸门、请求起点间隔、瞬时错误重试和备用模型回退；
   - 必须作为模型适配器能力实现，不改变 Agent 图。

5. **工具韧性与 Harness**
   - 缺少按工具分级超时、熔断、循环调用检测、工具顺序断言和返回结构检查；
   - 不与已有工具大结果卸载中间件重复。

6. **输入/输出安全护栏**
   - 缺少提示词注入特征检测和最终输出敏感内部信息清理；
   - 安全拒绝必须显式可观测，不能静默改写业务事实。

### 3.3 可观测性、异步化与缓存

7. **LangFuse 全链路观测**
   - 缺少 LangFuse 客户端、LangChain Callback 接线、会话/买家/run 标签和无配置降级；
   - 必须覆盖主 Agent 和 fork Worker，并避免记录密钥等敏感配置。

8. **Redis 跨进程事件背板**
   - 当前仅有进程内 EventBus；缺少 worker → API 的 Redis Pub/Sub 传播和 origin 去重。

9. **Redis Stream 意图队列**
   - 当前只有 TaskQueue 端口和空 worker 入口；缺少 enqueue/consume/ack、状态查询、幂等键和死信路径。

10. **安全语义缓存与 Embedding 缓存**
    - 缺少按模型、Prompt、买家和偏好版本隔离的只读问句语义缓存；
    - 写操作、指代型问题和已有会话历史必须绕过；
    - Embedding 缓存作为现有编码器装饰器实现，不改变检索用例。

### 3.4 持久化与交付

11. **会话/对话持久化闭环**
    - 当前上下文 Event Log 可审计，但 LangGraph 默认 `InMemorySaver`；
    - 需要持久化 Checkpointer 或明确的状态恢复适配器，并补齐对话查询能力。

12. **依赖健康检查与完整事件流**
    - 当前健康检查和事件类型有限；
    - 需要报告数据库、Redis、OpenSearch、模型观测开关等状态，并推送 tool/task/fork/context/cache 事件。

13. **可复现运行环境**
    - Docker Compose 需要补齐 API、worker、Redis、OpenSearch 的依赖关系与健康检查；
    - 增加 smoke、并行、故障降级和端到端验证入口；
    - 前端不作为本轮业务核心，先保证 HTTP/SSE/WebSocket 可完整消费事件。

## 4. 分步实施顺序

### Phase 1：长期偏好与个性化检索

1. 新增 BuyerPreference、PreferenceKind 和 PreferenceStore 端口；
2. 实现 SQLite PreferenceStore，保证同买家同语句幂等；
3. 实现 dislike 全保留、like 按相关性/时间选取的 PreferenceSelector；
4. 新增 remember/forget 两个 `@tool` 工厂；
5. 在模型调用前以动态消息注入偏好，不污染 L0/L1 稳定缓存前缀；
6. 将 PreferenceStore 接入现有 User Tower，让偏好实际影响商品召回和重排；
7. 验收跨会话、删除、黑名单优先、无偏好降级及缓存前缀稳定性。

### Phase 2：订单意向单

1. 新增订单聚合、状态机和仓储端口；
2. 实现 SQLite 仓储与并发安全订单 ID；
3. 实现创建、查询、取消用例，库存能力通过端口注入；
4. 新增三个 `@tool` 和 REST 查询/取消接口；
5. 添加用户确认边界、写操作幂等键和审计事件；
6. 明确不实现支付授权、退款或真实物流。

### Phase 3：运行防护和安全

1. 模型并发/间隔/重试/回退适配器；
2. 工具分级超时与熔断中间件；
3. 调用循环、顺序、Schema 和 Token 预算检查；
4. 输入注入检测与最终输出护栏；
5. 故障注入测试证明降级路径不编造成功。

### Phase 4：LangFuse 可观测性

1. 使用官方 LangChain 集成接入 Callback Handler；
2. 统一 `run_id/thread_id/shopping_session_id/buyer_id/agent_id` 元数据；
3. 主 Agent 和 fork Worker 共用观测工厂；
4. 未配置 LangFuse 时零副作用运行；
5. 验证模型、工具、子 Agent、Token 和延迟链路。

### Phase 5：Redis 队列、事件背板与缓存

1. Redis 客户端生命周期和健康检查；
2. EventBus Pub/Sub 背板和 origin 去重；
3. Stream 队列、任务状态、幂等提交、重试与死信；
4. worker 消费与 API 同步/异步提交模式；
5. 安全语义缓存和 Embedding 缓存；
6. Redis 故障时直跑/未命中降级。

### Phase 6：持久化、交付和总体验收

1. 持久化 LangGraph Checkpointer/恢复路径；
2. 扩展事件协议和健康检查；
3. 补齐 Compose、README、运行脚本和 `.env.example`；
4. 运行单元测试、架构测试、两套召回评测和真实 LLM smoke；
5. 输出最终实现说明、边界、配置、启动方式与验证证据。

## 5. 完成判定

只有同时满足以下条件才视为完成：

- 上述新增模块均从 Composition Root 可达，不是孤立代码；
- 不配置 Redis、LangFuse、Web Search 时，基础 Agent 主链仍能运行；
- 配置相应依赖后，偏好、订单、事件、队列、缓存和观测均有对应集成测试或 smoke 证据；
- 全部自动化测试通过；
- 使用 `.env` 中模型配置完成至少一次真实 Agent 调用；
- 文档明确区分已实现能力、可选能力和仍未实现的支付/真实平台商品接入。
