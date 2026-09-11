# 面试证据链实现与取证状态

## 正式状态

| Claim ID | 能力 | 状态 | 正式证据 |
|---|---|---|---|
| ENV-001 | Python/uv/锁文件、GPU、Docker、Redis、OpenSearch、Langfuse 与数据集预检 | verified | 三套 `preflight.json` |
| ARCH-001 | DDD-lite/Hexagonal 依赖方向与权限矩阵 | verified | Offline |
| ORCH-001 | 并行、DAG、部分失败、租约恢复、幂等与隔离 | verified | Offline |
| TEST-001 | 全量测试、分层覆盖率和关键领域分支门槛 | verified | Offline |
| REL-001 | 模型/工具/检索/存储/上下文故障矩阵 | verified | Offline |
| RAG-THRESHOLD-001 | 独立开发集拒答阈值选择 | verified | Offline |
| RAG-001 | 品类五种检索策略消融 | verified | Offline |
| SEARCH-001 | 商品三种检索策略消融 | verified | Offline |
| SEARCH-PERSONALIZATION-001 | 个性化 off/on 定向实验 | verified | Offline |
| COMMERCE-001 | 偏好、黑名单、计价、参数和订单意向规则 | verified | Offline |
| E2E-001 | 真实 Kimi 工具路由、Fork/DAG 与边界 | verified | Live E2E |
| CTX-001 | 三模式真实 Kimi 长会话 A/B/C | verified | Live Context |
| OBS-001 | Langfuse Trace 回读与事件覆盖 | verified（抽样） | Live E2E + Live Context |
| BOUNDARY-PAYMENT | 真实库存、支付、退款和第三方平台下单 | boundary | 不实现 |

## 三套正式快照

### Offline：`20260909T073736Z-offline`

- Commit：`fa1ca544bbfcd720918bbbed835d4f95bce57472`。
- 170 项测试通过；总行覆盖率 73.42%。
- Domain 行覆盖率 93.19%，Application 行覆盖率 85.79%，关键领域规则行/分支覆盖率 100%。
- 4 路 I/O P50 串行 0.240 秒、并行 0.070 秒，加速 3.43×。
- 重复派发率和上下文泄漏率为 0；部分失败保留、DAG、中断恢复和幂等回写均为 100%。
- 17 类故障注入 17/17 通过。
- 品类最终策略：Recall@10 94.44%、MRR@10 80.37%、NDCG@10 81.79%、负例拒识率 90%。
- 商品最终策略：Recall@10 96.27%、MRR@10 93.22%、NDCG@10 92.58%。
- 发布脱敏扫描通过。

### Live E2E：`20260909T094839Z-live-e2e`

- Commit：`bd67b9a9a03e004760e835dc6e1dd5b5cf08c05c`。
- 12 个场景 × 3 次，36/36 通过。
- 启动 123 次模型请求，完成并记录 116 次响应；观测 515,793 Token。
- 模型并发 10、子 Agent 并发 8、启动间隔 0.75 秒，未触发请求/Token 上限。
- Langfuse 回读 3/3 条抽样 Trace，其中 2 条满足完整主/子拓扑；Fork 与 final 事件存在，run/trace 映射率 100%。
- 发布脱敏扫描通过。

### Live Context：`20260909T141654Z-live-context`

- Commit：`38e4fdf8b9436bb083d3a75ce2aa675e46cf7add`。
- `off`、`deterministic`、`full` × 3 会话 × 8 轮全部通过。
- `full` 相比 `off` 的 AgentLoop 输入 Token 降低 43.30%，信息保留率和同 Epoch 前缀稳定率为 100%。
- 7 次摘要尝试中 5 次产生净压缩，2 次被安全拒绝；卸载 7 个大工具结果，滚动 7 次 Epoch。
- 共 107 次模型请求、118,743 个观测 Token。
- Langfuse 回读 3/3 条抽样 Trace，其中 1 条满足完整治理拓扑；compression 事件存在，run/trace 映射率 100%。
- 发布脱敏扫描通过。

## 后续维护规则

1. 修改对应能力后，只重跑受影响套件，并在干净 Commit 上重新发布。
2. 简历数字引用正式快照，不用本地临时候选覆盖。
3. 新快照若回归失败，保留旧 verified 结果，同时将新问题记录为待修复，不能选择性隐藏失败样本。
4. 每次投递前检查 [results/latest.json](results/latest.json)、[诚实边界](boundaries.md) 和 [面试速讲](interview_brief.md) 是否一致。
