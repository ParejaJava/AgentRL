# CategoryInsight 工具设计与用法

## 在线三级降级

每次返回都会通过 `recall_strategy` 标出实际检索路径：

1. `embedding_rerank`：OpenSearch BM25+KNN Pipeline 融合后使用 BGE 重排；
2. `embedding_only`：reranker 故障时保留 Hybrid 原始顺序；
3. `keyword_2gram`：embedding、OpenSearch 或 Hybrid 召回故障/空召回时，读取本地
   JSONL 卡片快照，以英文词元和中文二元组做确定性检索。

非法请求和损坏的知识卡数据不会被降级链吞掉。

## 1. 功能边界

`category_insight` 查询稳定的品类常识，包括：

- 典型组件和热卖款型；
- 常见使用场景与卖点；
- 关键材质、参数和属性选择口径；
- 入门、主流、高端价格档；
- 明确的避坑点。

它不查询具体商品、库存、促销或实时价格。这些事实仍由 `item_search` 等商品工具负责。

主 Agent 与 fork 子 Agent 都装配该工具。系统提示词允许模型在搜索前调用它改善检索词，
也允许在拿到候选商品后调用它补充精挑依据。

## 2. 分层位置

```text
domain/catalog/category_insight.py
  纯品类知识模型与不变量

application/catalog/
  category_insight_models.py       请求与召回结果
  category_insight_ports.py        CategoryKnowledgeRetriever 端口
  get_category_insight.py          quick/deep 聚合用例

infrastructure/retrieval/category_insight/
  ingestion.py                     Markdown 增量摄取与结构化 LLM
  schemas.py                       LLM/JSONL Pydantic Schema
  local_store.py                   JSONL 审计存储与本地降级检索
  opensearch.py                    索引、Search Pipeline、同步与在线检索
  factory.py                       local/opensearch 服务装配

infrastructure/langchain/tools/category_insight.py
  @tool 工厂
```

Application 只依赖检索端口，不知道 JSONL、LangChain 或未来的 OpenSearch。

## 3. 输出模型

工具成功时返回以下结构：

```json
{
  "status": "ok",
  "depth": "deep",
  "category": "旅行装备",
  "components": ["收纳袋", "颈枕", "眼罩"],
  "bestsellers": [
    {
      "name": "旅行三件套",
      "components": ["收纳袋", "颈枕", "眼罩"],
      "use_cases": ["长途飞行"],
      "selling_points": ["轻量", "好收纳"],
      "typical_attributes": {},
      "confidence": 0.92
    }
  ],
  "attributes": [],
  "price_tiers": [],
  "pitfalls": [],
  "source_card_ids": ["category-..."],
  "last_updated": "2026-08-26T00:00:00+00:00",
  "confidence": 0.9
}
```

`source_card_ids` 用于审计，但 `raw_evidence` 永远不会进入工具返回值。

## 4. quick 与 deep

| depth | 返回内容 | 默认召回量 |
|---|---|---:|
| `quick` | 组件、爆款、价格档、避坑点 | 8 |
| `deep` | quick 全部内容，加属性分布和详细选择口径 | 20 |

工具默认使用 `quick`。只有材质、参数、属性差异会影响决策时才应使用 `deep`。

## 5. 一次结构化、长期复用

执行：

```bash
uv run python scripts/ingest_category_knowledge.py
```

摄取步骤：

1. 递归读取 `knowledge/*.md`，排除 README；
2. 计算每份原文的 SHA-256；
3. 对新增或哈希变化的文档调用一次零温度结构化 LLM；
4. 验证每段 `raw_evidence` 确实存在于输入 Markdown；
5. 生成稳定 `card_id`；
6. 原子写入 `data/category_insight/cards.jsonl`；
7. 更新 `data/category_insight/manifest.json` 中的哈希和 card_id 映射；
8. 使用 `BAAI/bge-m3` 为新增或变化卡片生成 1024 维向量，并同步 OpenSearch；
9. 删除 OpenSearch 中已不在本地完整快照内的旧卡片。

如果文档哈希和卡片均存在，后续运行会跳过该文档，不再产生 LLM 调用。删除 Markdown
后重新摄取，会从卡片库中移除该来源对应的卡片。

## 6. 在线查询

在线 `category_insight` 不调用生成式 LLM，只执行：

```text
category + depth
  -> BAAI/bge-m3 查询向量
  -> OpenSearch Hybrid Query：BM25 + KNN
  -> Search Pipeline：加权 RRF 融合两路排名
  -> BAAI/bge-reranker-v2-m3 对候选卡做交叉编码精排
  -> 最低置信度过滤
  -> 类型分组和稳定去重
  -> 相关度加权整体置信度
  -> CategoryInsightOutput
```

这里的两个 BGE 模型职责不同：`bge-m3` 是双塔向量模型，适合高吞吐召回；
`bge-reranker-v2-m3` 同时读取查询和候选文本，速度较慢但排序更准，因此只处理默认 50 条
候选。它不是生成式 LLM，不会改写知识卡内容。

`LocalCategoryCardRetriever` 保留为无 OpenSearch 时的开发/降级适配器。没有可靠命中时返回
`status=not_found`，不会编造常识。

## 7. Docker 部署 OpenSearch

安装 Docker Desktop 后，在仓库根目录执行：

```bash
docker compose -f docker/docker-compose.yml up -d opensearch
curl http://localhost:9200
```

Compose 使用 OpenSearch `3.8.0` 单节点模式，持久数据卷为 `opensearch-data`，端口为
`9200`（REST）和 `9600`（Performance Analyzer）。当前配置关闭安全插件，仅适合本机开发，
不能直接暴露到公网。

索引和名为 `globex-category-rrf` 的 Search Pipeline 不需要手工创建：首次摄取或查询时，
`OpenSearchCategoryCardRepository.ensure_resources()` 会幂等创建/更新。Pipeline 使用
`score-ranker-processor`，按 `0.4 / 0.6` 对 BM25 与 KNN 执行加权 RRF；这解决两路原始分数
量纲不同的问题。随后 Python 侧 reranker 做最终精排，因此 Pipeline 与 reranker 不是重复步骤。

## 8. 安装依赖与首次摄取

```bash
uv sync --extra search --extra rag
docker compose -f docker/docker-compose.yml up -d opensearch
uv run python scripts/ingest_category_knowledge.py
```

`search` extra 安装 FlagEmbedding，`rag` extra 安装官方 `opensearch-py` 客户端。Docker 容器是
独立运行的搜索服务，Python 依赖只是让本项目能连接该服务，两者缺一不可。

首次运行会下载 BGE 模型并构建向量，耗时取决于网络和硬件。以后若文档哈希、卡片 ID 和
embedding 模型都未变化，既不会再次调用结构化 LLM，也不会重复生成向量。

## 9. 配置

```dotenv
CATEGORY_KNOWLEDGE_ROOT=knowledge
CATEGORY_CARD_STORE=data/category_insight/cards.jsonl
CATEGORY_INGESTION_MANIFEST=data/category_insight/manifest.json
CATEGORY_STRUCTURING_MODEL=qwen-max
CATEGORY_STRUCTURING_MAX_TOKENS=4096
CATEGORY_QUICK_RECALL_K=8
CATEGORY_DEEP_RECALL_K=20
CATEGORY_MIN_CONFIDENCE=0.6
CATEGORY_RETRIEVER_BACKEND=opensearch
CATEGORY_EMBEDDING_MODEL=BAAI/bge-m3
CATEGORY_EMBEDDING_DIMENSION=1024
CATEGORY_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
CATEGORY_HYBRID_RECALL_K=50

OPENSEARCH_URL=http://localhost:9200
OPENSEARCH_VERIFY_CERTS=false
OPENSEARCH_CATEGORY_INDEX=globex_category_kb
OPENSEARCH_CATEGORY_PIPELINE=globex-category-rrf
OPENSEARCH_BM25_WEIGHT=0.4
OPENSEARCH_KNN_WEIGHT=0.6
```

摄取模型复用 OpenAI-compatible 的 `LLM_API_KEY` 和 `LLM_BASE_URL`，但模型名及输出预算可以
独立配置。

如需暂时绕过 OpenSearch，设置 `CATEGORY_RETRIEVER_BACKEND=local`。修改向量维度后必须使用
新索引名或重建旧索引，因为 `knn_vector.dimension` 不能在线修改。

相关官方文档：

- [OpenSearch Docker 安装](https://docs.opensearch.org/latest/install-and-configure/install-opensearch/docker/)
- [Hybrid search](https://docs.opensearch.org/latest/vector-search/ai-search/hybrid-search/index/)
- [Score ranker processor](https://docs.opensearch.org/latest/search-plugins/search-pipelines/score-ranker-processor/)
