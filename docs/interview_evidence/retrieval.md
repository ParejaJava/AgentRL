# RAG-001 / SEARCH-001：两套检索

## 为什么是两套

品类检索返回的是离线结构化知识卡，属于 RAG；商品检索返回可购买候选商品，属于业务召回。两者数据、相关性标签和失败边界不同，指标不能合并。

## 品类知识 RAG

Markdown 只在离线发生变化时调用一次结构化 LLM，产出版本化知识卡。在线依次支持 keyword、OpenSearch BM25、BGE-M3 KNN、Hybrid RRF、Hybrid RRF + BGE cross-encoder reranker 五种消融。生产链发生 embedding/hybrid 或 reranker 故障时降级，但消融入口禁止隐式降级，防止“测 A 实际跑 B”。

拒答阈值只在 `eval/evidence/category_threshold_dev.jsonl` 上选择：先约束 Recall@10 ≥93%，再最大化负例拒识率、NDCG 和阈值。100 条冻结测试集不参与选择。

## 商品候选召回

商品链支持 lexical、embedding、embedding + reranker 三种消融。BGE-M3 双塔把查询和商品离线/在线编码到同一向量空间；FAISS 做高召回候选搜索；cross-encoder 将“查询 + 单个候选”一起编码，计算更精确的相关性，因此只用于候选重排。用户偏好向量与查询向量融合，硬约束过滤和到手价发生在排序之后的应用/领域规则中。

冻结集回归门槛：Recall@10 ≥95%、MRR@10 ≥92%、NDCG@10 ≥92%。每个策略同时记录 Precision 和 P50/P95 延迟，全部失败用例进入报告。

正式 Offline 快照结果如下：

| 系统/策略 | Recall@10 | MRR@10 | NDCG@10 | P50 延迟 |
|---|---:|---:|---:|---:|
| 品类 keyword | 46.11% | 34.81% | 36.50% | 2.16 ms |
| 品类 Hybrid RRF | 95.56% | 71.44% | 75.57% | 125.66 ms |
| 品类 Hybrid + Reranker | 94.44% | 80.37% | 81.79% | 666.30 ms |
| 商品 lexical | 98.26% | 88.62% | 87.49% | 2.75 ms |
| 商品 embedding | 95.90% | 94.70% | 91.91% | 35.98 ms |
| 商品 embedding + reranker | 96.27% | 93.22% | 92.58% | 509.49 ms |

品类重排相较 Hybrid RRF 的 Recall 下降 1.11 个百分点，但 MRR、NDCG 和负例拒识率分别提高约 8.93、6.21 和 90 个百分点，说明它主要改善首位排序和域外拒答。商品最终策略相较 lexical 的 Recall 下降约 1.99 个百分点，但 MRR 和 NDCG 提高约 4.60 和 5.09 个百分点，且仍通过 Recall≥95% 的回归门槛；因此不能表述为“商品 Recall 相较词法提升”。

独立 30 条开发集选择阈值 0.01，当时 Recall@10 为 93.33%、负例拒识率为 86.67%；冻结测试集最终负例拒识率为 90%。个性化定向集开启画像后，MRR@10 提升 6.67 个百分点、NDCG@10 提升 3.83 个百分点，Recall 保持 100%。证据见 [品类报告](results/20260909T073736Z-offline/category_hybrid_rerank.json)、[商品报告](results/20260909T073736Z-offline/product_embedding_rerank.json) 和 [个性化报告](results/20260909T073736Z-offline/product_personalization.json)。

## STAR 话术

- S：单纯词法检索无法处理语义改写；只看语义又会把不可购买商品排到前面。
- T：分别提高知识命中和商品候选质量，并保证模型故障时仍可服务。
- A：两套独立标注集、Hybrid/向量召回、交叉编码重排、确定性约束和三级降级；通过消融拆分每层收益。
- R：品类最终策略达到 Recall 94.44%、MRR 80.37%、NDCG 81.79%、负例拒识 90%；商品最终策略达到 Recall 96.27%、MRR 93.22%、NDCG 92.58%，并明确说明 Recall 与排序质量的取舍。
