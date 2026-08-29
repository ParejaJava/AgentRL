# Knowledge

这里存放经过版本管理的电商品类知识文档。知识摄取程序应通过 Application
端口写入向量库，Domain 不直接读取 Markdown、Qdrant 或其他具体存储。

当前摄取入口为：

```bash
uv run python scripts/ingest_category_knowledge.py
```

规则：

- `README.md` 不参与摄取，其他 `*.md` 会递归发现；
- Markdown 应明确区分品类定位、热卖款型、关键属性、价格区间和避坑点；
- 摄取 manifest 使用 SHA-256 判断内容是否变化；
- 未变化文档不会再次调用结构化 LLM；
- 生成文件位于 `data/category_insight/`，不提交到 Git；
- 修改知识事实必须修改源 Markdown，不能直接编辑生成的 JSONL。
