# Globex Agent 项目简历描述（证据约束版）

> 数字均引用 `docs/interview_evidence/README.md` 中的正式 Claim；未测量的生产收益不写入正文。

## 版本 A：一页简历结果版

### Globex Agent｜跨境购物多 Agent 搜索与决策平台（个人项目）

**技术栈：** Python、LangGraph/LangChain、FastAPI、BGE-M3、BGE Reranker、FAISS、OpenSearch、Redis、SQLite、WebSocket/SSE、Docker Compose、Pytest、uv

设计并实现面向跨境购物决策的 Agent 平台，覆盖品类知识洞察、个性化商品搜索、跨境到手价估算、多任务并行、长会话治理和订单意向；以 DDD-lite + Hexagonal Architecture 隔离领域规则、应用编排与模型/检索基础设施。

- **多 Agent 与 Task DAG（ORCH-001）：** 针对复杂请求的上下文串扰和独立任务串行等待，将计划外置为持久化依赖 DAG，通过原子认领、Worker 租约、独立 `thread_id`、最小工具权限和 settled 幂等回写实现主/子 Agent 协作；30 轮 4 路 I/O 实验的 P50 加速比为 **3.43×**，重复派发率与上下文泄漏率均为 **0**，部分失败结果保留率、中断恢复率和幂等回写率均为 **100%**。

- **Cache-aware 上下文治理（CTX-001）：** 将会话上下文拆为结构化任务状态、可更新工作记忆、热消息和冷引用，结合工具结果卸载、确定性裁剪、LLM 增量摘要、Cache Epoch 与 Breakpoint 稳定前缀；3 组固定 8 轮 Kimi A/B/C 实验中，`full` 相比 `off` 的 AgentLoop 输入 Token 降低 **43.30%**，关键信息保留率和 Epoch 内前缀 Hash 稳定率均为 **100%**。供应商未返回缓存 Token，因此不声称真实 Prompt Cache 命中率。

- **个性化商品搜索与跨境计价（SEARCH-001、SEARCH-PERSONALIZATION-001、COMMERCE-001）：** 使用 BGE-M3 + FAISS 召回、交叉编码重排、用户偏好加权和黑名单过滤，并仅在存在 `ship_to` 时内联汇率、运费、关税与免税额；在 67 条冻结集上达到 **Recall@10 96.27%、MRR@10 93.22%、NDCG@10 92.58%**，定向画像集 personalization on 相比 off 的 MRR 提升 **6.67 个百分点**。

- **品类知识 RAG（RAG-001、RAG-THRESHOLD-001）：** 将 Markdown 知识离线一次结构化为可复用知识卡，在线使用 OpenSearch BM25 + KNN Hybrid RRF 和 BGE Reranker，并提供三级降级；独立开发集选择拒答阈值后，100 条冻结集达到 **Recall@10 94.44%、MRR@10 80.37%、NDCG@10 81.79%**，负例拒识率达到 **90%**。

- **可靠性与可观测性（REL-001、OBS-001）：** 建立参数/领域/服务三层校验、共享并发与 Token 硬预算、429/超时指数退避、熔断、三级检索降级和 Emergency Projection；17 类故障注入全部通过。以统一事件协议向 WebSocket/SSE 推送 Run、Model、Tool、Task、Fork、Retrieval 和 Context 事件，并接入 Langfuse；两套 live 实验均回读 **3/3** 条抽样 Trace、run/trace 映射率 **100%**，最终通过 **170 项**回归测试。

## 版本 B：Agent 平台技术深挖版

### Globex Agent｜Cache-aware Multi-Agent Commerce Platform（个人项目）

**技术栈：** Python 3.12、LangGraph/LangChain 1.x、FastAPI、BGE-M3、BGE-reranker-v2-m3、FAISS、OpenSearch Hybrid Search、Redis、SQLite WAL、Langfuse、Docker Compose、Pytest、uv

- **DDD-lite + Hexagonal（ARCH-001）：** 将商品、SKU、Money、关税和订单意向建模为纯领域对象，将运行、编排、检索、存储和事件定义为端口，把 LangGraph、模型供应商、OpenSearch、Redis、SQLite 和 Langfuse 置于适配器层；使用 AST 架构测试持续阻止 Domain 反向依赖框架，关键领域规则行/分支覆盖率均为 **100%**。

- **同构 Agent Runtime 与权限隔离（ORCH-001）：** 主/子 Agent 复用 LangGraph 的模型节点、工具节点、条件边、中间件和停止逻辑，不共享消息 State；路由策略仅在“可并行、需上下文隔离、预计调用链 ≥3 层”之一成立时 fork，Worker 按任务获得最小工具集。真实证据运行固定模型并发 **10**、子 Agent 并发 **8**、请求启动间隔 **0.75 秒**。

- **可恢复 Task Planning（ORCH-001）：** 以 SQLite WAL 持久化 Task、依赖、Owner、租约和终态，使用乐观并发控制完成循环检测、原子认领与后继解锁；过期租约把中断任务恢复为 `pending`，重复提交相同终态为 no-op，冲突终态拒绝覆盖。30 轮实验中 DAG 正确率、中断恢复率和幂等回写率均为 **100%**。

- **四层上下文与全局模型预算（CTX-001、REL-001）：** 每轮在 Breakpoint 后执行确定性卸载/裁剪，并按阈值调用 LLM 生成增量工作摘要；Cache Epoch 变化时才重建稳定前缀。主 Agent、子 Agent、重试和摘要模型共享线程安全账本，每套 live 证据硬限制 **250 次请求、1,000,000 Token**，超限立即停止且禁止发布残缺报告。

- **双检索系统与可解释评测（RAG-001、SEARCH-001）：** 将知识卡 RAG 与商品候选召回分开评测；前者覆盖 keyword/BM25/KNN/Hybrid/Hybrid+Reranker 消融及拒答阈值校准，后者覆盖 lexical/embedding/embedding+reranker 和 personalization off/on，统一输出 Recall、Precision、MRR、NDCG、P50/P95 延迟、降级模式及全部失败样本。

- **购物领域不变量（COMMERCE-001）：** 使用 Money/Decimal 处理跨币种预算与到手价，在工具执行前拒绝非法币种、数量、SKU 和目的地；订单意向要求已有商品搜索证据和用户确认，并以幂等键阻止重复创建。项目明确不连接真实库存、支付、退款或第三方平台下单。

- **可复现证据工程（ENV-001、TEST-001、E2E-001）：** 建立 preflight、offline、live、publish 四阶段入口，报告绑定 Git SHA、依赖锁摘要、数据集 SHA256、模型版本、Docker/OpenSearch Pipeline、执行命令、耗时和失败样本；发布器拒绝脏工作区、失败套件、Commit 不一致或脱敏失败的结果。12 个真实 Kimi 场景各重复 3 次，**36/36 通过**。

## 正式证据核验表

| 简历结论 | Claim ID | 正式值 | 报告套件 |
|---|---|---:|---|
| 编排加速与安全性 | ORCH-001 | 3.43×；泄漏/重复派发 0；恢复/幂等 100% | Offline |
| 上下文 Token 降幅 | CTX-001 | 43.30%；信息保留 100% | Live Context |
| 商品搜索 | SEARCH-001 | R@10 96.27%；MRR 93.22%；NDCG 92.58% | Offline |
| 个性化排序 | SEARCH-PERSONALIZATION-001 | MRR +6.67 个百分点 | Offline |
| 品类 RAG | RAG-001 | R@10 94.44%；MRR 80.37%；NDCG 81.79%；拒识 90% | Offline |
| 自动化回归 | TEST-001 | 170 passed；总覆盖率 73.42% | Offline |
| 真实 Agent 场景 | E2E-001 | 36/36 通过 | Live E2E |
| Langfuse Trace | OBS-001 | 两套均回读 3/3 抽样；映射率 100% | Live E2E + Live Context |

## 不得写成成果的边界

- 不写“完成购买”：系统只创建订单意向，不处理真实库存、支付、退款或平台下单。
- 不写“Prompt Cache 命中率 80%”：当前只能证明同 Epoch 稳定前缀 Hash 一致率。
- 不写“上下文治理降低总 Token 成本”：当前只验证 AgentLoop 输入下降 43.30%，计入摘要模型后总 Token 尚未下降。
- 不写生产 QPS、线上转化率、真实计价误差或 badcase 定位耗时：尚无对应正式实验。
- 不写“全部遥测 100% 上传”：当前只证明抽样 Trace 可回读及其结构完整度。
