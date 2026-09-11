# 子 Agent LoRA SFT 实验报告

状态：工程、真实训练、完整冻结测试对照、真实模型 Agent 接入与轨迹导出已执行。
当前 adapter 的端到端回答质量检查未通过，不应作为生产默认执行模型。

## 已验证的实现

- 主/子 Agent 可分别配置模型与 endpoint，默认不切换执行模型。
- 角色专用 lite/fallback；主子共享并发信号量和实际请求/token 预算。
- 实际调用边界记录治理后输入、工具定义、输出、使用量、父子关系和失败状态。
  普通重试、fallback、应急投影通过逻辑 step_id 关联；开启 executor 覆盖或采集时
  关闭 SDK 隐藏重试。未配置新功能时保留原有重试和缓存命名空间。
- 训练数据导出需要显式质量与隐私审核；检查工具参数、消息配对和会话 split 泄漏。
- 独立训练环境、目标 assistant loss、LoRA 保存/加载、冻结评测与比较器。
- 本地开发推理服务、实际 MainAgent/fork 集成检查入口及配置回滚说明。

运行命令与边界见 [training/README.md](../training/README.md)。

## 训练条件与结果

| 项目 | 实际值 |
|---|---|
| 基座 | Qwen/Qwen3-0.6B |
| Revision | c1899de289a04d12100db370d81485cdf75e47ca |
| GPU | RTX 5060 Laptop，8 GB |
| 方法 | BF16 LoRA SFT，非思考模式 |
| LoRA | rank 16，alpha 32，dropout 0.05，q/k/v/o projection |
| 可训练参数 | 4,587,520 |
| 学习率 / 累积 | 1e-4 / 每 8 条样本更新一次 |
| 数据来源 | 自行设计的合成决策，未使用真实用户数据或付费教师 API |
| 划分 | 224 train / 56 dev / 84 test |
| 训练样本最大长度 | 1,376 tokens |
| Epoch | 2，按开发集 loss 选择第 1 轮 |
| 开发 loss | 初始 0.833539 → 第 1 轮 0.128454 → 第 2 轮 0.149594 |
| 脚本内训练计时 | 1,447.916 秒 |
| 峰值分配显存 | 1,933,725,696 bytes（约 1.80 GiB） |
| Adapter 权重大小 | 18,380,008 bytes |

训练产物位于 `training/outputs/executor-v2/`。首次完整训练因 CUDA OOM 失败，
保留为 `executor-v1` 失败记录；随后只计算目标预测位置的词表 logits，降低显存开销。
该计算与完整 masked causal loss 的微型 Qwen 对照：loss 相同，梯度最大误差 0。
小样本 smoke adapter 已实际加载并生成工具调用；它不作为最终质量结论。

## 评测与适用范围

基座和按开发集选择的第 1 轮 adapter 均完成全部 84 条测试，比较器已验证输入数据、
基座文件、评分代码、模板、解码、上下文与生成上限一致，且关闭 fallback 和响应缓存。

| 指标 | 基座 | LoRA adapter |
|---|---:|---:|
| 单步规则通过数 | 40/84 | 72/84 |
| 单步规则通过率 | 47.62% | 85.71% |
| 描述性 Wilson 95% 区间 | 37.28%–58.17% | 76.67%–91.64% |
| 工具选择/格式通过率（54 条工具目标） | 85.19% | 100% |
| 工具参数通过率（54 条工具目标） | 57.41% | 98.15% |
| 生成延迟 p50 | 15.24 秒 | 25.92 秒 |
| 生成延迟 p95 | 28.75 秒 | 44.11 秒 |
| 总生成耗时 | 1,513.84 秒 | 2,345.00 秒 |

配对比较有 33 条从失败变通过、1 条从通过变失败，净提升 38.10 个百分点。
p95 延迟增加 53.39%；当前直接加载未合并的 PEFT adapter，未做生产推理优化。
本次实验不能支持“推理更快”的结论，也没有测量生产货币成本。

弱项集中在币种追问（0/6，基座为 1/6）、偏好冲突（1/6，基座为 0/6）以及品类查询后
继续搜索（5/6）。其余 11 类各为 6/6。完整切片、逐例原始输出及耗时位于
`eval/reports/executor/base-test-v1/` 和 `adapter-test-v1/`，比较报告为
`eval/reports/executor/comparison.json`。

数据按类别实体与主要表达模板划分，任务家族和部分上下文结构共享。它能支持小规模
工具决策可行性实验，不代表真实用户泛化、现有主模型等效能力或生产成本节省。
关键词评分需要结合原始输出复核；训练 loss 下降不能替代工具生成评测。
逐条 Wilson 区间仅作描述性参考，同品类样本有相关性，不能当作线上泛化的置信保证。

### 已检查的失败原始输出

以下检查仅解释现有冻结测试结果，没有用于修改权重、样本或评分程序：

- `test-00-clarify`：输入明确“货币单位尚未确定，确认好才能检索”，adapter 却调用
  `item_search` 并自行指定 `target_currency="CNY"`。这是应追问时直接行动的错误，
  不只是措辞没有命中关键词。
- `test-00-preference-conflict`：输入同时要求排除品牌甲并仅选品牌甲，adapter 回答
  “你同时满足历史不喜欢的品牌和候选必须出自品牌甲的条件。”，没有说明冲突或请求确认。
- `test-02-category-then-search`：输入要求配送美国，品类工具返回后 adapter 搜索参数
  使用 `ship_to=null`，其余类别、预算、币种、数量正确。参数 JSON 合法仍不等于任务约束正确。

逐例证据见 `eval/reports/executor/adapter-test-v1/cases.jsonl`，对应输入在冻结的
`eval/executor/test.jsonl`。这些反例说明当前 adapter 仍需要独立场景验证与约束检查，
不能仅凭总体评分切换为生产默认执行模型。

## 实际 Agent 接入与轨迹闭环

两次运行均使用实际本地 adapter、生产 MainAgent/fork、确定性上下文治理、生产
`item_search` 工具和商品过滤逻辑。planner 为脚本化分派器，目录为明确的合成词法夹具，
因此不验证付费主模型的规划能力或向量检索质量。

| 运行 | 工具与结果证据 | 回答质量 |
|---|---|---|
| integration-v1 | 模型将品类简化为“水杯”；精确品类过滤返回空；实际父子链和结果消费正常 | 未通过，未纠正 category_mismatch，笼统归因于配送/预算不满足 |
| integration-v2 | 补充明确目录品类名称；DE/EUR/50/top_k=2 正确；返回唯一 CUP-OK，过滤超预算与不配送商品 | 未通过，将同一 CUP-OK 重复列举两次 |

第二次运行的原始接入检查曾显示 `passed=true`，因为当时只检查调用路径和有效商品出现。
复核发现重复商品后已补充质量检查与回归测试，追加审计文件
`eval/reports/executor/integration_quality_audit.json` 将两次整体质量均判为失败，保留原始记录。
这说明接口和真实工具执行链已接通，但不能宣称端到端购物任务已经可靠。

第二次捕获的两个 executor attempt 经逐条审核后导出：正确工具调用保留原响应，重复回答
通过 `corrected_target` 修正为唯一商品、20 欧元及非到手总价说明。审核者为本次 Codex
代理，不是外部人工标注团队；内容全部来自合成夹具。导出明确记录目标来源及原响应/审核哈希。
两个样本属于同一会话、同一 train split，未加入本次已完成训练或测试。

实际导出位于 `eval/reports/executor/integration-v2/reviewed-decisions.jsonl`，
`reviews.json` 保留审核理由；`export-validation.json` 证明 JSON Schema、配对、会话隔离、
真实 tokenizer 前缀和目标标签检查通过。序列长度分别为 1,213 和 1,642 tokens。

本次自有训练、评测和本地推理进程均已退出，8010 无监听，GPU 显存回落至 30 MiB。

## 工程验证

完整项目测试：194 passed。报告位于 `eval/reports/executor/regression-completion.xml`。
覆盖独立模型图执行、父子上下文、治理后请求记录、fallback、应急重试额度、
数据隐私审核、split 泄漏、工具格式、金额精度及开发服务请求契约。
新增接入质量回归覆盖重复商品、错误金额，以及不能从任务原文偷取答案所缺的价格说明。
当前产物交叉核验见 `eval/reports/executor/completion_evidence.json`，逐项对应关系见
[executor_sft_audit.md](executor_sft_audit.md)。

没有修改用户 `.env`，没有提交或推送代码，没有购买云资源。用户原有文档改动保留。
