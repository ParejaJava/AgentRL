# Globex Agent

> 面试证据、真实指标边界和一键复现入口见 [`docs/interview_evidence/README.md`](docs/interview_evidence/README.md)。简历数字以绑定固定 Commit 的发布报告为准。

Globex Agent 是一个面向多 Agent 场景的全栈项目骨架，后端使用 FastAPI，前端使用 React + Vite。

后端采用 DDD-lite + Hexagonal/Onion Architecture：`domain` 只保存电商业务模型与规则；Agent Runtime、编排和上下文治理策略属于 `application`；LangChain、LangGraph、ContextVar、SQLite、FAISS 和 FastAPI 都位于外层适配器。详细调整说明见 [`docs/architecture_reorganization.md`](docs/architecture_reorganization.md)。

## 快速开始

```bash
uv sync --extra search --extra rag --extra platform --extra dev
docker compose -f docker/docker-compose.yml up -d redis opensearch
uv run python scripts/ingest_category_knowledge.py
uv run uvicorn app.presentation.server:app --reload
```

复制 `.env.example` 为 `.env`，至少配置 `LLM_API_KEY`、`LLM_BASE_URL`
和 `LLM_MODEL_NAME`。`app.infrastructure.settings` 会在容器启动时读取该文件。

如需一次启动 API、worker、Redis 和 OpenSearch，请先启动 Docker Desktop，再执行：

```bash
docker compose -f docker/docker-compose.yml up --build
```

## 平台运行能力

- 主 Agent 与 fork 子 Agent 复用同一套 LangGraph 编排定义，使用独立 `thread_id` 隔离状态；
- SQLite 持久化买家偏好，并同时注入模型工作记忆与商品检索用户塔；
- 订单意向支持创建、查询与取消，但不处理真实支付、库存占用和退款；
- 模型调用具备并发门控、重试和备用模型，工具具备超时、熔断、循环及前置证据链保护；
- 可选启用单次意图四档 Token 预算、Silent-Drift 事件和 Redis 跨实例共享熔断；
- 可选 LangFuse Trace、Redis Pub/Sub 事件、Redis Stream 异步任务、Embedding 与语义响应缓存；
- `POST /api/agent` 返回 SSE，`WS /ws/events/{shopping_session_id}` 可订阅工具、fork、压缩和最终结果事件。

完整实现与边界见
[`docs/reference_gap_full_implementation.md`](docs/reference_gap_full_implementation.md)。

## 会话级上下文治理

主 Agent 和每个 fork 子 Agent 都通过相同的 LangChain Middleware 执行：

- L0 静态系统提示词与工具定义保持稳定；
- L1 Epoch Baseline 在同一 cache epoch 内保持稳定；
- L2 动态工作集执行工具结果清理、D4 卸载和增量摘要；
- Event Log 写入 `data/sessions/{thread_id}/events.db`；
- 大型工具结果写入 `data/sessions/{thread_id}/artifacts/`；
- LLM 只在结构化增量摘要和 Epoch Baseline 合并时调用；
- 压缩触发、D1 保护、工具配对和 Epoch Roll 均由确定性规则控制。

完整设计和配置说明见
[`docs/cache_aware_session_context_management_design.md`](docs/cache_aware_session_context_management_design.md)。

默认使用供应商的隐式前缀缓存。只有切换到支持显式缓存标记的 Qwen 模型并确认接口兼容后，才设置：

```dotenv
LLM_CACHE_PROVIDER=qwen
LLM_EXPLICIT_CACHE=true
```

显式缓存标记只添加到临时 Prompt Projection，不会写入 LangGraph State。

## 品类洞察知识库

`category_insight` 会在商品搜索前后提供稳定的品类常识。首次使用或修改
`knowledge/*.md` 后，运行一次增量摄取：

```bash
uv run python scripts/ingest_category_knowledge.py
```

摄取脚本只对新增或内容哈希发生变化的 Markdown 调用结构化 LLM，结果保存在
`data/category_insight/cards.jsonl`，未变化文档会直接复用并增量同步 OpenSearch。在线查询采用
BM25 + `BAAI/bge-m3` KNN 混合召回，由 OpenSearch Search Pipeline 做加权 RRF，再使用
`BAAI/bge-reranker-v2-m3` 精排。它不会调用结构化生成式 LLM，也不会把知识库原文返回给
Agent；详细设计见 [`docs/category_insight.md`](docs/category_insight.md)。

摄取完成后可运行 100 条离线检索评测：

```bash
uv run python scripts/evaluate_category_recall.py --backend opensearch
```

报告包含 Recall@K、Precision@K、MRR、NDCG@K、负例拒答准确率和标签切片结果。

商品搜索工具使用独立的 60 商品、67 查询评测集：

```bash
uv run python scripts/build_product_eval_index.py
uv run python scripts/evaluate_product_recall.py --k 1 3 5 10
```

商品搜索的 `category`、`ship_to`、预算、币种和 Top-K 均由结构化参数校验；硬约束
由应用代码过滤，不交给模型猜测。传入 `ship_to` 后，商品卡会内联“商品小计 + 运费
+ 关税”的估算到手价，并携带静态规则及汇率快照版本。当前规则边界、计算公式和索引
兼容性见 [`app/infrastructure/retrieval/item_search/README.md`](app/infrastructure/retrieval/item_search/README.md)。

前端开发：

```bash
cd frontend
npm install
npm run dev
```

存活检查：`GET http://127.0.0.1:8000/health`；依赖就绪检查：`GET http://127.0.0.1:8000/health/ready`。

统一证据入口：

```powershell
uv run python -m scripts.evidence preflight
uv run python -m scripts.evidence run --suite offline
uv run python -m scripts.evidence run --suite live
uv run python -m scripts.evidence publish --run-id <run_id>
```

会话事件订阅：`WS /ws/events/{shopping_session_id}`。`shopping_session_id`
是购物会话和事件分区键，`thread_id` 是 LangGraph 短期消息历史键，两者不能混用。
