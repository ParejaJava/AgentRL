# 决策样本设计

每条样本监督子 Agent 的**下一步动作**：调用工具，或直接回答/追问。
输入包含当时可见的消息与生产工具定义，目标只包含最后一个 assistant 消息。
历史 assistant、system、user、工具返回和内部隐藏思维链都不是本次 loss 目标。

当前样本由我在本次任务中根据项目规则编写场景并生成，`source=authored_synthetic`。
没有把成功返回的模型响应自动当成正确答案，也没有使用真实用户对话冒充合成数据。
生成器位于 `training/build_dataset.py`，冻结文件在 `eval/executor/`。

| 场景 family | 学习的决策 |
|---|---|
| search / simple | 搜索商品，提取明确约束；不为未提供的条件编造值 |
| modify | 使用用户本轮修改后的条件 |
| category-quick / category-deep | 按问题深度选择品类知识查询模式 |
| clarify | 必要信息不明且要求先确认时追问 |
| empty | 如实说明工具没有返回候选 |
| error | 工具熔断后停止重复调用并报告限制 |
| grounded | 根据已返回商品编号和金额回答，不编造商品事实 |
| compressed | 从压缩后的任务状态恢复条件 |
| preference-conflict | 应用明确的偏好与排除约束 |
| category-then-search | 利用品类知识后继续查具体商品 |
| retry-once | 对允许重试的临时失败采取下一步动作 |
| web | 为可能变化的规则选择网页检索 |

三个真实训练文件示例（以下省略共同 system 和工具 schema）：

- `train-00-search`：用户“帮我找咖啡杯，寄到德国，预算30欧元，给我2个候选。”
  目标为 `item_search(normalized_query="咖啡杯", category="咖啡杯", ship_to="DE",
  price_max_major=30, target_currency="EUR", top_k=2)`。国家和预算必须进入结构化参数。
- `train-00-clarify`：用户“找咖啡杯，预算50，不知道用哪种币种，先问清楚再搜。”
  目标为“请确认预算使用哪种币种？”，此步不调用搜索。
- `train-00-error`：商品工具返回 `circuit_open` 并说明勿重复调用。
  目标说明暂时无法确认可购买商品和价格，建议稍后重试。

当前训练、开发、测试分别为 16、4、6 个类别实体，每个实体 14 类场景，共
224/56/84 条。类别实体和主要用户表达模板跨 split 隔离，场景家族及部分结构化
上下文模板共享，因此它是小规模受控实验，不能据此宣称真实用户泛化已经解决。
测试集在完整训练前冻结；模型选择只看开发集 loss。看过测试结果后应使用新版本
实验及新的独立测试数据，不能改当前评分规则来追求通过率。

运行轨迹使用 `source=reviewed_runtime`，经质量与隐私审核后才能导出。错误响应
不自动成为目标；可提供 `corrected_target` 并明确审核来源。同一购物会话的各 fork
属于同一 split，导出校验使用捕获的会话哈希额外检查泄漏。导出后的消息还需经过
实际 tokenizer 的前缀、标签掩码和长度检查，才能进入训练。
