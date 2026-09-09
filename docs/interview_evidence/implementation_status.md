# 面试证据链实现与取证状态

## 已实现

| Claim ID | 实现内容 | 本地候选状态 |
|---|---|---|
| ENV-001 | Python/uv/锁文件、OS、CPU/GPU/CUDA、模型缓存、Docker、Redis、OpenSearch、Langfuse 与数据集预检 | verified |
| ARCH-001 | AST 依赖边界、依赖方向图、Composition Root 图、主/子工具权限矩阵 | verified |
| ORCH-001 | 4 路串并行、菱形 DAG、部分失败、原子认领、Worker 租约恢复、幂等回写、会话隔离，各 30 次 | verified |
| TEST-001 | 全量测试、总/分层覆盖率、关键领域分支门槛 | verified |
| REL-001 | 模型限额/重试、Redis 熔断 fail-open、两套三级降级、压缩失败、应急投影与工具卸载故障矩阵 | verified |
| RAG-THRESHOLD-001 | 独立开发集拒答阈值选择，冻结测试集不参与调参 | verified |
| RAG-001 | keyword/BM25/KNN/Hybrid RRF/Hybrid + Reranker 消融 | verified |
| SEARCH-001 | lexical/embedding/embedding + reranker 消融 | verified |
| SEARCH-PERSONALIZATION-001 | 固定定向集 personalization off/on | verified |
| COMMERCE-001 | 偏好、黑名单、ship_to 条件计价、Money/Decimal、参数拒绝、订单证据/确认/幂等 | verified |
| E2E-001 | 12 场景 × 3 次真实 Kimi 工具路由、参数、Fork/DAG、安全与边界 | verified |
| CTX-001 | 3 模式 × 3 会话 × 8 轮真实 Kimi A/B/C | verified |
| OBS-001 | 事件协议、WebSocket/SSE、Langfuse 回读、run/trace 关联和脱敏拓扑摘要 | verified |
| BOUNDARY-PAYMENT | 真实库存、支付、退款和第三方平台下单 | boundary |

这里的 `verified` 表示本地候选报告有真实运行结果；由于当前 Git 工作区尚未提交，它们还不是正式发布快照。

## 最新原始候选

最终代码已完成新的完整 offline 候选回归。由于工作区未提交，它仍不能发布为正式快照。

### Offline：`20260908T160311Z-offline`

- 套件总状态：通过。
- 170 项测试通过；总行覆盖率 73.42%。
- Domain 行覆盖率 93.19%，Application 行覆盖率 85.79%。
- 关键领域规则行/分支覆盖率均为 100%。
- 4 路 I/O P50 串行 0.248 秒、并行 0.070 秒，加速 3.55×。
- 重复派发率与上下文泄漏率均为 0；部分失败保留率、DAG 正确率、中断恢复率与幂等回写率均为 100%。
- 17 类模型、工具、检索、存储、上下文和子 Agent 故障注入全部通过。
- 品类阈值开发集选择 0.01，Recall@10 93.33%、负例拒识率 86.67%。
- 品类冻结集最终策略 Recall@10 94.44%、MRR@10 80.37%、NDCG@10 81.79%、负例拒识率 90%。
- 商品冻结集最终策略 Recall@10 96.27%、MRR@10 93.22%、NDCG@10 92.58%。

### Live：`20260908T143344Z-live`

- 套件总状态：通过。
- 36 次端到端运行完成 36 次，通过率 97.22%；唯一失败是一次品类知识请求未调用工具。
- E2E 使用 130 次请求、550,589 Token；上下文实验使用 120 次请求、128,995 Token。
- 整个 live 套件合计恰好 250 次请求、679,584 Token，没有超过硬上限。
- 上下文 `full` 相比 `off` 输入 Token 降低 32.21%，信息保留率 100%，稳定前缀率 100%。
- Langfuse 成功回读 3/3 条完整主/子 Agent Trace，run/trace 映射率 100%，Fork、压缩和 final 事件摘要均存在。

## 发布前还需执行

1. 审查当前改动并提交，确保 `git status --short` 为空。
2. 在该 Commit 上重新执行 offline 和 live；旧原始报告因 Commit 不同不能发布。
3. 分别执行 `publish --run-id`，让发布器完成脱敏扫描并生成 `docs/interview_evidence/results/` 快照。
4. 发布 live 快照中的 `observability.json`；它已通过 API 回读验证，但只能证明 3 条抽样 Trace，不能宣称全量遥测 100% 上传。
5. 将简历每个数字链接到最新正式快照；没有报告支持的耗时收益继续留为 planned。
