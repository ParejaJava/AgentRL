# 证据复现手册

## 环境准备

```powershell
uv sync --extra search --extra rag --extra platform --extra dev
docker compose -f docker/docker-compose.yml up -d redis opensearch
uv run python scripts/ingest_category_knowledge.py --sync-only
```

模型必须已进入本地 Hugging Face 缓存；正式评测期间应设置离线模式，防止误访问公网：

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
```

## 统一入口

```powershell
# 模型、GPU、Docker、Redis、OpenSearch、Langfuse（--live）和数据集
uv run python -m scripts.evidence preflight
uv run python -m scripts.evidence preflight --live

# 确定性编排、架构/覆盖率、业务/可靠性测试、阈值校准和两套消融
uv run python -m scripts.evidence run --suite offline

# 真实 Kimi、12 条规划场景、上下文 A/B/C，并从 Langfuse 回读脱敏 Trace 摘要
uv run python -m scripts.evidence run --suite live

# 仅在工作区干净、固定 Commit 且套件全部通过时发布
uv run python -m scripts.evidence publish --run-id <run_id>
```

live 默认固定：模型并发 10、子 Agent 并发 8、启动间隔 0.75 秒、最多 250 次模型请求和 1,000,000 Token。可用 `--repetitions`、`--max-requests`、`--max-total-tokens` 向下收紧，不应超过默认门槛。

为避免临时网络重试随机挤占 E2E 与上下文实验共享的 250 次硬预算，正式 E2E 取证关闭外层模型重试，并把该策略写入 `live.json`；上下文实验保留环境中配置的重试策略。429、超时、暂时性错误的重试与退避能力由不调用外部模型的 `REL-001` 故障注入矩阵独立证明。

live 只有在 `observability.json` 至少回读一条包含主 Agent、子 Agent、模型和工具节点的完整 Trace，且 Fork、压缩、`final.result` 事件摘要齐全时才通过。公开报告不会保存原始输入、完整 Prompt 或模型输出。

## 单项消融

```powershell
uv run python scripts/evaluate_category_recall.py --backend opensearch --strategy keyword
uv run python scripts/evaluate_category_recall.py --backend opensearch --strategy bm25
uv run python scripts/evaluate_category_recall.py --backend opensearch --strategy knn
uv run python scripts/evaluate_category_recall.py --backend opensearch --strategy hybrid_rrf
uv run python scripts/evaluate_category_recall.py --backend opensearch --strategy hybrid_rerank

uv run python scripts/evaluate_product_recall.py --strategy lexical
uv run python scripts/evaluate_product_recall.py --strategy embedding
uv run python scripts/evaluate_product_recall.py --strategy embedding_rerank

# 个性化定向集：off/on（正式套件会自动执行并计算差值）
uv run python scripts/evaluate_product_recall.py --dataset eval/evidence/product_personalization.jsonl --strategy embedding_rerank
uv run python scripts/evaluate_product_recall.py --dataset eval/evidence/product_personalization.jsonl --strategy embedding_rerank --buyer-id evidence-buyer --profile "喜欢环保、天然、无塑料、再生尼龙和小众设计；不喜欢塑料材质"
```

若 `publish` 因脏工作区拒绝，这是预期行为：先审查并提交代码，再在该 Commit 上重新运行正式套件。不要为绕过门槛手工复制原始报告。

首次成功发布 offline 快照时，发布器会生成 `docs/interview_evidence/results/coverage_baseline.json`，把该次总覆盖率及 Domain/Application/可靠性分层覆盖率锁定为后续回归下限；关键领域规则的分支门槛无论首次测量如何都不得低于 90%。

发布成功后，每个运行目录会包含 `claim_index.json`，把 Claim ID、代码/测试锚点和本次报告文件连接起来；`docs/interview_evidence/results/latest.json` 会分别保存最新 offline 与 live 指针，不会让后发布的套件覆盖另一套证据入口。
