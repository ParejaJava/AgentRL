# Evaluation

这里存放 Agent Runtime、编排、检索和上下文治理的评测用例与回归报告。

当前自动化正确性测试位于 `tests/`；后续离线评测数据应放在本目录，而不是
导入 `app` 运行包。

## CategoryInsight 检索评测

`category_recall.jsonl` 固定包含 100 条样本：90 条正例和 10 条无答案负例。正例覆盖
品类、爆款、属性、价格、避坑、多来源、同义改写、错别字与中英文混输。

相关性单元使用 `(source_document, card_type)`，而不是 `card_id`：`card_id` 可能随结构化
模型重新摄取而变化，而来源文件与卡片类型是稳定标注。`topic` 仅用于人工审阅，`grade`
用于 NDCG 分级：3 是核心答案、2 是必要补充、1 是弱相关。

脚本计算：

- `Recall@K`：相关单元中有多少被找回；
- `Precision@K`：Top-K 中相关单元占 K 的比例；
- `MRR@max(K)`：第一个相关单元排名的倒数；
- `NDCG@K`：考虑 1-3 级相关度的排序质量；
- `Negative rejection accuracy`：无答案查询是否返回空结果。

同一来源、同一卡片类型的多个检索结果会在计算前折叠，防止重复卡片虚增指标。正例排序
指标和负例拒答指标分开汇总。完整逐条结果默认写入
`eval/reports/category_recall.json`，该运行产物不提交 Git。

## ItemSearch 商品检索评测

`seed_products.py` 提供 60 个评测商品，已经适配当前通用 `Product`：旧版 SKU、价格、库存、
配送地和商品亮点被投影到 `attributes`，因此都会进入 BGE-M3 商品检索文本。

`product_recall.jsonl` 包含 67 条查询，使用稳定 `item_id` 标注相关商品。每条样本包含：

- `case_id`：稳定用例 ID；
- `index_id`：固定为 `evaluation-products`；
- `kind`：`lexical` 或 `semantic`；
- `relevance`：商品 ID 与 1-3 级相关度；
- `tags`：支持 lexical、semantic、constraint-in-query 等切片。

当前 `item_search` 没有 `price_max_major` 和 `ship_to` 独立参数，所以价格与配送约束保留在
query 文本中，由 BGE-M3 与 reranker 进行语义匹配。它们不属于确定性元数据过滤测试。

评测完整商品工具的最终排名，即：

```text
query -> BGE-M3 -> FAISS HNSW 召回 -> BGE reranker -> Top-K
```

指标与品类评测保持一致：Recall@K、Precision@K、MRR@max(K)、NDCG@K。报告还会按
`tags` 输出同样四项指标的切片结果。

命令行评测默认逐条打印进度，逐条明细和最终汇总仍以 JSON 报告为准。进度行中的指标
使用命令指定的最大 K，例如 `--k 1 3 5 10` 会显示 `R@10`、`P@10` 和 `NDCG@10`。
