# Scripts

这里存放索引构建、并发验证、评测回归和运维脚本。脚本必须通过
`app.composition` 获取应用依赖，不能自行装配模型或存储客户端。

## 品类知识摄取

```bash
uv run python scripts/ingest_category_knowledge.py
```

该脚本通过 `build_category_ingestor()` 获取结构化模型和卡片存储。第一次运行会处理
全部知识 Markdown；以后只处理新增或 SHA-256 发生变化的文件。

## 品类检索离线评测

评测线上 OpenSearch + Pipeline + reranker：

```bash
uv run python scripts/evaluate_category_recall.py --backend opensearch
```

快速验证本地降级检索器：

```bash
uv run python scripts/evaluate_category_recall.py --backend local --k 1 3 5 10
```

执行前必须先运行品类知识摄取。脚本通过 `build_category_retriever()` 获取与线上完全相同的
检索器，默认读取 `eval/category_recall.jsonl`，并将完整报告写入
`eval/reports/category_recall.json`。

## 商品检索离线评测

先用 60 个种子商品构建独立评测索引：

```bash
uv run python scripts/build_product_eval_index.py
```

然后运行完整 ItemSearch 两阶段检索评测：

```bash
uv run python scripts/evaluate_product_recall.py --k 1 3 5 10
```

默认索引域是 `data/indexes/evaluation-products`，评测集是
`eval/product_recall.jsonl`，报告写入 `eval/reports/product_recall.json`。索引构建器为保护
已有产物不会覆盖非空目录；种子变化后请先明确移走旧索引，或通过 `--index-id` 构建新版本，
并同步修改评测数据中的 `index_id`。

该评测计算 ItemSearch 最终重排结果的 Recall@K、Precision@K、MRR 和 NDCG@K。价格与
配送条件只作为 query 文本语义，不会绕过当前工具边界引入隐藏过滤器。
