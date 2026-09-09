# RAG-001 / SEARCH-001：两套检索

## 为什么是两套

品类检索返回的是离线结构化知识卡，属于 RAG；商品检索返回可购买候选商品，属于业务召回。两者数据、相关性标签和失败边界不同，指标不能合并。

## 品类知识 RAG

Markdown 只在离线发生变化时调用一次结构化 LLM，产出版本化知识卡。在线依次支持 keyword、OpenSearch BM25、BGE-M3 KNN、Hybrid RRF、Hybrid RRF + BGE cross-encoder reranker 五种消融。生产链发生 embedding/hybrid 或 reranker 故障时降级，但消融入口禁止隐式降级，防止“测 A 实际跑 B”。

拒答阈值只在 `eval/evidence/category_threshold_dev.jsonl` 上选择：先约束 Recall@10 ≥93%，再最大化负例拒识率、NDCG 和阈值。100 条冻结测试集不参与选择。

## 商品候选召回

商品链支持 lexical、embedding、embedding + reranker 三种消融。BGE-M3 双塔把查询和商品离线/在线编码到同一向量空间；FAISS 做高召回候选搜索；cross-encoder 将“查询 + 单个候选”一起编码，计算更精确的相关性，因此只用于候选重排。用户偏好向量与查询向量融合，硬约束过滤和到手价发生在排序之后的应用/领域规则中。

冻结集回归门槛：Recall@10 ≥95%、MRR@10 ≥92%、NDCG@10 ≥92%。每个策略同时记录 Precision 和 P50/P95 延迟，全部失败用例进入报告。

最新本地候选中，品类 Hybrid RRF + Reranker 在冻结集达到 Recall@10 94.44%、MRR@10 80.37%、NDCG@10 81.79%、负例拒识率 90%；商品 embedding + reranker 达到 Recall@10 96.27%、MRR@10 93.22%、NDCG@10 92.58%。个性化定向集 on 相比 off 的 MRR@10 提升 6.67 个百分点、NDCG@10 提升 3.83 个百分点。正式简历引用前仍需在干净 Commit 上重跑和发布。

## STAR 话术

- S：单纯词法检索无法处理语义改写；只看语义又会把不可购买商品排到前面。
- T：分别提高知识命中和商品候选质量，并保证模型故障时仍可服务。
- A：两套独立标注集、Hybrid/向量召回、交叉编码重排、确定性约束和三级降级；通过消融拆分每层收益。
- R：只引用对应策略报告，不能把品类的提升写成商品提升，或把旧负例 0% 隐藏起来。
