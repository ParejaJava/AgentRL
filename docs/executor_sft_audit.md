# 子 Agent SFT 完成核对

此表按用户的最小闭环目标核对实际证据。工程、真实训练、评测、接入与导出均已执行；
模型回答质量未达到生产替换要求。原始运行产物在本地保留，权重、环境和轨迹不进入 Git。

| 要求 | 证据入口 | 当前结论 |
|---|---|---|
| 检查硬件、缓存与项目约束后选择基座 | `executor_sft_worklog.md` 初始检查；`eval/reports/executor/base_model_manifest.json` | RTX 5060 Laptop 8 GB；固定 revision 的 Qwen3-0.6B，权重哈希已核验 |
| 子 Agent 独立模型、endpoint、上下文与备用模型 | `app/infrastructure/settings.py`、`app/infrastructure/llm.py`、`app/composition.py`；`tests/test_executor_runtime.py` | 已实现并通过配置、图路由及密钥隔离测试 |
| 保留主 Agent 默认行为与共享预算 | runtime 测试中的默认 SDK 重试、共享并发/预算、应急请求测试 | 未启用新配置时保留原路径；启用后逐实际请求计数 |
| 实际调用轨迹采集 | `trajectory_recorder.py`；`integration-v1/captures/`、`integration-v2/captures/`；runtime 测试 | 两次真实图执行各捕获 2 planner + 2 executor attempts，父子关系、治理输入、工具消费及预算已核验 |
| 决策样本设计、数据导出与校验 | `eval/executor/manifest.json`；`training/DATASET.md`；`integration-v2/reviews.json`、`reviewed-decisions.jsonl`、`export-validation.json` | 224/56/84 合成划分；实际导出 2 条同会话样本，1 原始工具调用 + 1 显式修正答案；Schema、配对、隔离和 tokenizer 检查通过 |
| 独立训练环境和训练脚本 | `training/.venv/pyvenv.cfg`、`training/environment-lock.txt`、`training/train_sft.py`、`training/check_loss.py` | 已真实训练；目标 assistant token loss 与完整 masked loss 对照一致 |
| 一次真实 LoRA 训练、保存与加载 | `training/outputs/executor-v2/`；`eval/reports/executor/adapter_validation.json` | 两轮训练完成，按 dev loss 选择第 1 轮；保存权重通过检查并已加载生成 |
| 基线与微调模型评测 | `eval/reports/executor/base-test-v1/`、`adapter-test-v1/`、`comparison.json` | 两侧均 84 条完成，40→72 条通过，配对 33 改善/1 退步；模型/数据/代码/解码一致；p95 变慢已报告 |
| 实际 Agent 接入 | `training/serve_executor.py`、`integration_smoke.py`；`integration-v1/`、`integration-v2/`、`integration_quality_audit.json` | 真实调用链、正确工具参数与生产过滤结果消费已验证；品类简化和重复回答导致质量未通过，两次原始结果及追加审计均保留 |
| 接入、复现与回滚说明 | `training/README.md`、`training/executor.env.example` | 已提供独立环境、训练、评测、版本接入及回滚命令 |
| 回归验证与限制说明 | `eval/reports/executor/regression-completion.xml`、`executor_sft_report.md` | 194 项通过；Ruff、diff whitespace、离线 lock 检查通过；明确披露追问/冲突/品类/重复回答及延迟问题 |
| 保留用户修改、不提交推送、不购买云资源或调用未授权付费 API | 工作树状态、脚本化 planner 与仅限 loopback 的集成入口 | 用户既有文档修改保留；未改 `.env`，未提交或推送 |

本地路径缩写均以 `eval/reports/executor/` 为运行证据根目录。
`completion_evidence.json` 将当前数据内容哈希、模型文件哈希、比较报告、实际捕获、
审核导出、标签校验与测试 XML 逐项交叉核验。现场只读检查确认本次训练/评测/服务进程
均已退出，8010 无监听，GPU 显存 30 MiB。

本次交付是可复现的微调与证据闭环，不是生产质量认证。默认主/子模型未切换；
当前 adapter 的回答质量失败与实验完成同时成立，不能用 85.71% 合成单步评分掩盖它。
