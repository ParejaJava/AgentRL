# REL-001 / OBS-001：可靠性与可观测性

模型网关统一处理并发、请求启动间隔、指数退避、备用模型与四档 Token 预算。证据运行额外启用 250 次请求和 1,000,000 Token 硬上限；Agent 推理和上下文压缩 LLM 共享同一并发安全账本，重试请求也会计数。达到上限会停止，套件标记失败，不能发布残缺快照。

工具中间件负责分级超时、重复同参调用保护、熔断和结构化错误。商品与品类检索分别验证 embedding、reranker、检索后端故障的降级链；Task DAG 使用 settled 结果保证一个 Worker 失败时保留同批成功项；上下文接近溢出时使用 Emergency Projection。

正式 Offline 快照逐项运行 17 类故障注入，**17/17 通过**，包括 Kimi 429、连接超时与暂时错误重试，工具超时无副作用、工具异常熔断、Redis/OpenSearch 不可用、两套检索 embedding/reranker 故障、压缩器失败、上下文溢出、单 Worker 失败和大型工具结果卸载。故障替身证明控制流与不变量，不代表生产故障率。证据见 [rel-001.json](results/20260909T073736Z-offline/rel-001.json)。

OpenSearch 就绪检查实际执行 ping、索引存在、Search Pipeline 查询和索引集群绿色检查，不再根据环境变量开关伪造健康。单节点索引使用零副本避免长期 yellow。

统一事件总线覆盖 Run、Model、Tool、Task、Fork、Context 和 `final.result`，通过 SSE/WebSocket 推送。真实 Kimi 套件启用 Langfuse，并在公开快照中只保存 Trace/run 对应、父子拓扑、调用顺序、Token 和延迟摘要；密钥、私有地址、原始用户输入、模型输出和完整 Prompt 均不导出。

E2E 快照抽样请求并回读 3/3 条 Trace，其中 2 条满足完整主/子 Agent 拓扑，Fork 与 final 事件存在；Context 快照同样回读 3/3 条，其中 1 条满足完整上下文治理拓扑，compression 事件存在。两套 run/trace 映射率均为 100%。这是链路可用性抽样，不能扩大为全量遥测上传率。证据见 [E2E observability](results/20260909T094839Z-live-e2e/observability.json) 和 [Context observability](results/20260909T141654Z-live-context/observability.json)。

## STAR 话术

- S：长链 Agent 的失败可能来自模型、工具、检索、任务或上下文，只有最终错误无法定位。
- T：限制故障半径、保留部分结果，并让每次结论可追溯。
- A：统一网关和工具防线、三级降级、原子任务状态、真实健康探针、逐事件协议和 Langfuse 层级 Trace。
- R：只引用故障注入矩阵和 live Trace 实际报告；在没有人工计时实验前不写“排障从 30 分钟降到 5 分钟”。
