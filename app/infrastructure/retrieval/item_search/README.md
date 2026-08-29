# ItemSearch 商品搜索

ItemSearch 在应用装配时指定的索引域中检索商品。面向 LLM 的工具只接收标准化查询、
品类、收货地、预算和结果数量等业务参数；内部索引 ID 和买家 ID 由运行环境注入，
不会暴露给模型填写。

## 在线检索链路

1. Agent 传入标准化后的业务搜索参数。
2. BGE-M3 对查询编码；用户塔提供处于相同向量空间的用户偏好向量。
3. 对查询向量和用户向量归一化，再按照可配置权重融合。
4. FAISS HNSW 使用内积从指定索引域召回最多 100 个商品。
5. BGE Reranker v2 M3 对所有“查询—商品”候选对进行精排。
6. `ProductSearchSpec` 确定性校验品类、收货地、购买数量、目标币种和预算约束。
7. 过滤没有可售主 SKU 的商品，再从合格候选中截取 Top-K。
8. 传入 `ship_to` 时，在商品卡中内联估算到手价，包括商品小计、运费、关税、
   到手总价及规则版本。

## 三级降级链

每次响应都会通过 `recall_strategy` 标明实际使用的检索路径：

1. `embedding_rerank`：BGE-M3 向量召回成功，然后使用 BGE Reranker 精排。
2. `embedding_only`：Reranker 失败，保留 FAISS 向量相似度顺序。
3. `keyword_2gram`：Embedding 或向量召回失败，或者没有返回向量候选；此时使用
   本地商品元数据，以英文词元和中文二元组进行词法匹配。

非法请求和损坏的领域数据不会触发降级。降级只处理 Embedding、向量召回和
Reranker 等检索基础设施故障，避免把数据错误误判成可以绕过的问题。

## 参数核验和到手价

参数共经过三层核验：

1. LangChain 工具的 Pydantic 输入模型拒绝未知字段和基础格式错误。
2. 与框架无关的 `ProductSearchSpec` 再次校验全部业务不变量，确保 HTTP、测试、
   worker 或未来新增的工具适配器都不能绕过规则。
3. `ItemSearchService` 校验索引限制、支持的收货地、商品配送范围、主 SKU 库存、
   品类以及折算后的预算上限。

跨境计价通过应用层的 `PricingProvider` 端口提供。当前基础设施适配器使用带版本的
静态汇率、关税、免税额度和运费快照，计算规则如下：

```text
商品小计 = 主 SKU 单价 × 数量
运费 = 首件基础运费 + 首件基础运费 × 60% × 续件数量
关税 = max(折合 CNY 的商品小计 - 免税额度, 0) × 品类税率
到手总价 = 商品小计 + 运费 + 关税
```

金额以整数最小货币单位保存，并使用 `Decimal` 计算，避免浮点误差。返回结果是估算值，
不是最终结账报价；商品卡会同时返回关税规则版本和汇率版本，便于审计和问题追踪。

`price_max_major` 当前约束的是折算成目标币种后的主 SKU 商品价格，不包含运费和关税。

## 离线索引

`ItemIndexBuilder` 把稳定的商品文本编码成归一化的 BGE-M3 向量，并为每个检索域写入
一个独立目录：

```text
data/indexes/<index_id>/
├── index.faiss
├── manifest.json
└── products.json
```

`manifest.json` 固定记录 Embedding 模型和向量维度，防止在线编码器误查不兼容的旧索引。

新索引使用 manifest schema v2，以最小货币单位保存精确价格，并记录
`primary_sku_id`。读取端仍兼容 schema v1 及旧的 `{major, currency}` 价格结构；但是，
如果旧商品语料完全没有 SKU，就必须重新构建索引，因为这类商品不能被视为可购买商品。

## 模型加载

BGE 和 FAISS 适配器采用延迟导入。仅导入包、使用测试替身运行单元测试或创建服务时，
都不会下载或加载模型。只有第一次执行真实编码或重排时，才会加载相应模型。

## 离线评测

项目提供包含 60 个商品和 67 条查询的版本化评测集。执行以下命令构建索引并评测：

```bash
uv run python scripts/build_product_eval_index.py
uv run python scripts/evaluate_product_recall.py --k 1 3 5 10
```

评测脚本针对最终商品排序计算以下指标：

- Recall@K：相关商品能否被召回。
- Precision@K：返回的前 K 个商品中相关商品的比例。
- MRR：第一个相关商品的排名质量。
- NDCG@K：考虑相关性等级的整体排序质量。
