# Globex Agent 项目简历描述（目标完成态）

> 本文按“最终可投递状态”设计。简历正文聚焦设计思想、工程行动与结果，不展开函数、
> 脚本和接口名称；当前尚未完成或尚未压测的结果统一使用 `X`，待对应能力和报告完成后
> 替换。文末检查清单仅供投递前自查，不需要放进正式简历。

---

## 版本 A：一页简历结果版

### Globex Agent｜跨境电商多 Agent 搜索与决策平台（个人项目）

**技术栈：** Python、LangGraph/LangChain、FastAPI、BGE-M3、BGE Reranker、FAISS、
OpenSearch、SQLite、Redis、Docker Compose、Pytest、uv

设计并实现服务于跨境电商场景的 Agent 平台，覆盖品类知识洞察、商品召回与精排、
跨境到手价估算、多任务并行编排、长会话上下文治理和运行可观测性；以 DDD-lite +
Hexagonal Architecture 隔离电商领域规则、Agent 应用用例与模型/检索基础设施。

- **多 Agent 编排：** 针对复杂购物请求中单 Agent 上下文串扰、并行任务重复认领和
  计划在长对话中丢失的问题，设计 Main/Worker 同质运行图与外部化任务状态，使用依赖
  DAG、状态机、原子认领和隔离执行统一管理串并行任务；支持最多 **50** 个子任务并发
  配置，验证 **20 路并发写入无 ID 冲突**，最终将复杂任务成功率提升至 **X%**、独立任务
  并行执行耗时降低 **X%**。

- **Cache-aware 上下文治理：** 针对长 AgentLoop 的 token 膨胀与摘要破坏 Prompt Cache
  稳定前缀的矛盾，构建“稳定基线、结构化任务状态、热消息、冷事件”分层上下文，结合
  工具结果卸载、增量摘要、Cache Epoch、双 Breakpoint 哈希校验和溢出重试；将平均输入
  token 降低 **X%**，Prompt Cache 命中率稳定至 **X%**，长会话任务信息保留率达到 **X%**。

- **商品搜索与跨境计价：** 针对自然语言需求、商品语义相关性和可购买硬约束难以统一的
  问题，实现 BGE-M3 + FAISS 召回、BGE Reranker 精排、品类/配送地/SKU/预算多层校验，
  并在请求包含收货地时内联汇率、运费、关税和免税额度计算；通过三级检索降级保证基础
  模型故障时服务可用，在 **67 条**离线集上取得 **Recall@10 96.27%、MRR@10 93.22%、
  NDCG@10 92.58%**，Top-1 Precision 达 **91.04%**。

- **品类知识 RAG：** 针对模型重复阅读品类文档、知识输出不稳定和证据难复用的问题，
  将 Markdown 知识一次性结构化为可版本化知识卡，线上采用 OpenSearch BM25+KNN
  Pipeline 融合与 BGE 重排，并提供向量、重排和检索服务故障时的三级降级；建设
  **100 条**中英混合、口语改写、多相关项和负例评测集，最终达到 Recall@10 **X%**、
  MRR@10 **X%**、负例拒答准确率 **X%**。

- **Agent Harness 可靠性：** 针对工具失控、单个子任务失败拖垮整批结果和供应商限流，
  构建参数/领域/服务三层校验、并发信号量、429 指数退避、逐任务失败结算、工具大结果
  截断与错误显式返回机制；将异常场景任务完成率提升至 **X%**，并确保单 Worker 失败不
  丢失同批其他任务结果。

- **架构与可观测性：** 通过端口/适配器保护 Agent Runtime 与电商领域核心，使用统一事件
  协议串联模型、任务、工具、检索和上下文治理链路，并以 SSE/WebSocket + Redis 背板
  推送进度、对接全链路 Trace；完成 **72 项自动化测试**，将 badcase 定位时间由 **X 分钟**
  缩短至 **X 分钟**，并通过容器化编排实现一键复现评测与服务环境。

---

## 版本 B：Agent / LLM 工程技术版

### Globex Agent｜Cache-aware Multi-Agent Commerce Platform（个人项目）

**技术栈：** Python 3.11+、LangGraph/LangChain 1.x、FastAPI、BGE-M3、BGE Reranker、
FAISS HNSW、OpenSearch Hybrid Search、SQLite WAL、Redis、Docker Compose、Pytest、uv

从 0 到 1 构建跨境电商 Agent Platform，使主 Agent 既能直接完成品类洞察、商品搜索与
到手价计算，也能把复杂目标拆成带依赖的任务图并并行调度隔离 Worker；同时建设
Cache-aware 会话治理、结构化 RAG、检索评测和可观测运行时。

- **DDD-lite + Hexagonal Architecture：** 针对 Agent 框架、检索实现和电商规则耦合导致
  业务难测试、基础设施难替换的问题，将商品/SKU/金额/配送/关税建模为纯领域对象，
  将运行、编排、检索和事件抽象为应用端口，模型、LangGraph、OpenSearch、Redis、SQLite
  作为可替换适配器；以唯一 Composition Root 管理装配并用架构测试约束依赖方向，保持
  **72 项测试通过**，使核心业务规则不依赖具体 Agent/RAG 框架。

- **LangGraph 多 Agent Runtime：** 针对单 Agent 长链路上下文污染与权限扩散，设计主/
  子 Agent 同质图，复用模型节点、工具节点、条件边、Checkpoint 和 Context Middleware，
  同时通过独立运行标识和会话目录隔离消息历史；主 Agent 负责计划与调度，Worker 仅获得
  完成任务所需的最小工具权限，结合并发信号量和限流退避将最高并发配置为 **50**，复杂
  任务吞吐提升 **X 倍**。

- **Harness-level Task Planning：** 针对自然语言计划不可恢复、模型无法可靠判断可并行性
  的问题，将 Work State 从对话历史中分离，构建持久化任务看板、状态机、Owner、结果/
  错误、依赖 DAG 和动态可运行投影；派发前在事务内完成依赖校验、循环检测和原子认领，
  派发后逐项回写成功/失败并自动解锁后继任务，验证 **20 路并发创建无冲突**，任务重复
  派发率降至 **0%**，中断恢复成功率达到 **X%**。

- **四层 Cache-aware Context Governance：** 针对大工具结果、热对话、稳定任务状态与冷
  Trace 混入同一 Prompt 的问题，建立稳定 Baseline、结构化 Work State、可更新 Working
  Memory、热消息与冷引用分层；使用 append-only Event Log、Artifact Offloading、确定性
  压缩策略、LLM 增量摘要、Cache Epoch 和 Provider Cache Breakpoint 维持稳定前缀，并在
  上下文溢出时切换 Emergency Projection，将 P95 输入 token 控制在 **X**、缓存命中率
  提升至 **X%**。

- **商品检索与弹性降级：** 针对语义相关商品不一定可购买、检索模型故障会中断 AgentLoop
  的问题，将查询编码、用户信号、向量索引、重排器和计价服务抽象为独立端口，使用
  BGE-M3 + FAISS 召回与 BGE Reranker 精排，并在 Embedding/Reranker 异常时自动降级至
  向量原序或本地词法召回；在 **67 条**最终排序评测中达到 **Recall@3 94.65%、
  Recall@10 96.27%、MRR@10 93.22%、NDCG@10 92.58%**。

- **跨境 Pricing Domain：** 针对跨币种预算过滤与到手价逻辑散落、浮点金额不可审计的
  问题，以金额值对象和计价端口统一处理数量、汇率、运费、关税、免税额度与规则版本，
  使用整数最小货币单位和 Decimal 保证精度；计价不可用时保留有效商品候选并返回可解释
  原因，将估算结果与真实结账价的 P90 误差控制在 **X%**。

- **CategoryInsight Hybrid RAG：** 针对品类知识重复抽取、在线 LLM 输出漂移和纯关键词
  召回不准，设计“文档变更检测 → 一次结构化 → 知识卡持久复用 → 混合召回 → 交叉编码
  精排”链路；使用 OpenSearch BM25+KNN Pipeline 进行候选融合，BGE Reranker 负责最终
  排序，并保留本地降级通路，在 **100 条**离线集上达到 Recall@10 **X%**、MRR@10 **X%**、
  NDCG@10 **X%**。

- **事件驱动可观测性：** 针对长链路执行过程黑盒、工具与子任务 badcase 难定位的问题，
  定义覆盖 Run、Model、Tool、Task、Fork、Retrieval、Context 的统一事件协议，通过
  SSE/WebSocket 实时推送并使用 Redis 支撑 API/Worker 跨进程分发，接入 Trace、Token、
  Cache 和延迟指标；将问题定位时间从 **X 分钟**缩短至 **X 分钟**，事件投递成功率达到
  **X%**。

- **评测与交付工程：** 针对模型、索引、显存配置和运行环境变化导致结果不可复现，建立
  商品/品类两套离线评测，按场景与标签统计 Recall、Precision、MRR、NDCG 和负例拒答，
  统一 GPU Device/FP16/Batch Size 配置，并以多服务容器编排固化 API、Worker、OpenSearch、
  Redis 与推理依赖；将新环境搭建时间从 **X 小时**缩短至 **X 分钟**。

---

## 投递前私人核验清单（不要复制到简历）

### 已有真实结果，可保留

| 结果 | 当前可信值 |
|---|---:|
| 商品离线评测规模 | 67 条 |
| 商品 Recall@3 / Recall@10 | 94.65% / 96.27% |
| 商品 Precision@1 | 91.04% |
| 商品 MRR@10 / NDCG@10 | 93.22% / 92.58% |
| 品类离线评测规模 | 100 条（90 正例、10 负例） |
| 并发任务写入验证 | 20 路，无重复顺序 ID |
| 自动化测试 | 72 passed |
| 子 Agent 并发配置 | 默认 50 |

### `X` 指标取得后再替换

| 能力 | 建议最终指标 |
|---|---|
| Task DAG | 复杂任务成功率、并行加速比、重复派发率、中断恢复率 |
| 上下文治理 | 平均/P95 token、缓存命中率、关键信息保留率、压缩调用次数 |
| 品类 RAG | Recall@K、MRR、NDCG、负例拒答、正常/降级路径分桶指标 |
| Agent Harness | 模型/工具故障注入下的完成率、重试成功率、P95 延迟 |
| 到手价 | 对真实订单的 P50/P90 误差及不可估价比例 |
| 可观测性 | 事件投递率、首事件延迟、badcase 平均定位时间 |
| 容器化 | 冷启动时间、环境搭建时间、评测可复现率 |

### 写入最终简历前必须补齐的实现

1. 完成品类知识卡摄取与 OpenSearch 索引初始化，重新跑完 100 条评测；
2. 建立固定长会话集，对比治理开启/关闭时的 token、缓存和信息保留率；
3. 为任务图构建串行/并行对照实验与故障注入评测；
4. 完成模型、工具、任务、子 Agent、检索和压缩的逐事件推送；
5. 使用 Redis EventBus/Queue 替换单进程实现，并验证跨进程去重和恢复；
6. 接入统一 Trace 后记录 badcase 定位耗时；
7. 完成 API、Worker、OpenSearch、Redis 和推理服务的容器化编排；
8. 接入真实或回放订单，校准到手价误差。

不建议为了贴近参考简历而加入与当前项目路线无关的 SFT、Agentic RL、跨会话长期记忆或
专用文化合规 Agent。只有在确实完成训练、评测或产品需求验证后，再将其升级为简历成果。
