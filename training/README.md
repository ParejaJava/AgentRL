# 子 Agent LoRA SFT

这条流水线训练执行模型的工具调用和基于工具结果的回答。主 Agent 的默认模型不变。
当前使用 Qwen3-0.6B、BF16 LoRA、非思考模式；训练为 PyTorch + Transformers + PEFT，
不依赖 TRL，不访问付费教师 API，不自动上传模型。

## 环境与数据

以下命令在仓库根目录 PowerShell 执行。在线应用使用 `.venv`，训练使用
`training/.venv`。需要 CUDA；初始机器是 RTX 5060 Laptop 8 GB。

```powershell
uv venv training/.venv --python .venv/Scripts/python.exe
uv pip install --python training/.venv/Scripts/python.exe torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130 --cache-dir .uv-cache
uv pip install --python training/.venv/Scripts/python.exe -r training/requirements.txt --cache-dir .uv-cache
```

`environment-lock.txt` 保存本次实际安装版本。应用侧构建样本需要 dev 依赖中的 jsonschema。
公共基座来源：https://huggingface.co/Qwen/Qwen3-0.6B ，本次 revision：
`c1899de289a04d12100db370d81485cdf75e47ca`。权重在 `data/models/Qwen3-0.6B`，
不会进入 Git。新机器首次下载需要网络；下列命令不使用账号 token，并固定 revision。
下载后将该进程的 Hub 访问切为离线；训练和推理代码也使用 `local_files_only=True`。

```powershell
$env:HF_HUB_OFFLINE='0'
$env:HF_HUB_DISABLE_IMPLICIT_TOKEN='1'
training/.venv/Scripts/python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', revision='c1899de289a04d12100db370d81485cdf75e47ca', local_dir='data/models/Qwen3-0.6B', token=False)"
$env:HF_HUB_OFFLINE='1'
```

检查本地权重、配置和 tokenizer 哈希：

```powershell
training/.venv/Scripts/python.exe -m training.provenance data/models/Qwen3-0.6B --output eval/reports/executor/base_model_manifest.json
.venv/Scripts/python.exe -m training.build_dataset
training/.venv/Scripts/python.exe -m training.validate_dataset eval/executor/train.jsonl eval/executor/dev.jsonl eval/executor/test.jsonl
```

数据为明确标记的 `authored_synthetic`，当前 224 train / 56 dev / 84 test。
场景覆盖、实际样例与监督边界见 [DATASET.md](DATASET.md)。
工具定义从生产 `@tool` 工厂提取，预算参数通过生产输入模型校验，金额使用 Money 生成。
类别实体与主要请求表达模板跨 split 隔离；同一类别的多个步骤留在同一组。
任务类型与部分结构化上下文模板共享，因此这只是受控泛化实验，不代表真实用户分布。
尚无真实用户数据。修改数据时使用新目录或显式 `--replace-dataset`，不能在看过测试结果后
悄悄覆盖测试集；本次最终测试集冻结在正式完整训练之前。

## 训练与 loss 检查

```powershell
training/.venv/Scripts/python.exe -m training.check_loss
training/.venv/Scripts/python.exe -u -m training.train_sft --output training/outputs/my-executor --epochs 2
```

每个样本仅监督最后一个 assistant 目标，包括 tool calls；user/system/tool/历史 assistant
都不计算 loss。不训练隐藏思维链。模板文本前缀与 token 前缀均做检查，超长样本直接报错，
不会截断关键工具调用。对 Qwen 仅计算目标预测位置的词表 logits，降低显存；
`check_loss` 对比完整 masked loss 与优化后的 loss 和梯度。

训练保存 config（数据与代码哈希、显卡、版本、序列长度）、metrics（训练/开发 loss、
耗时、显存峰值）以及开发集 loss 最优 adapter。输出目录存在时拒绝覆盖。
开发 loss 下降不等于任务成功率提高，必须独立生成评测。

保存后的 adapter 可先在 CPU 上校验所有矩阵有限且 LoRA B 已更新：

```powershell
training/.venv/Scripts/python.exe -m training.check_adapter training/outputs/my-executor/adapter --output eval/reports/executor/adapter_validation.json
```

## 基线与微调模型评测

同一时刻仅运行一个 GPU 作业，避免显存竞争。先完成开发调试，再跑冻结测试集。

```powershell
training/.venv/Scripts/python.exe -u -m training.evaluate_executor --output eval/reports/executor/base-test
training/.venv/Scripts/python.exe -u -m training.evaluate_executor --adapter training/outputs/my-executor/adapter --output eval/reports/executor/adapter-test
.venv/Scripts/python.exe -m scripts.evidence.executor_sft --base eval/reports/executor/base-test --adapter eval/reports/executor/adapter-test --output eval/reports/executor/comparison.json
```

比较保持相同数据、工具、模板、解码和生成上限，关闭 fallback 和响应缓存。
报告保存每例原始生成、解析错误、截断、参数/格式检查、延迟、tokens、按场景切片和
Wilson 区间；同时绑定基座和 adapter 文件哈希。比较器拒绝不匹配的实验。
关键词检查仅能辅助判断回答，不证明完整语义正确或没有所有幻觉。没有付费主模型基线，
不能声称达到现有主模型质量；本地秒数和 tokens 不能直接换算成生产成本节省。

## 本地 Agent 接入

开发推理服务只绑定 `127.0.0.1`，串行使用一张 GPU，采用固定 greedy 解码。
支持非流式 `/v1/chat/completions` 的普通文本与工具调用，`tool_choice=auto/none`；
不支持音频、图片、结构化 response_format、n>1 或模型 token streaming。
应用的 Agent updates/SSE 仍可使用，但不是逐 token 输出。生产部署应另选并验证推理服务。

```powershell
training/.venv/Scripts/python.exe -m training.serve_executor --adapter training/outputs/my-executor/adapter --name globex-executor-v1
```

另一个终端运行无需付费服务的真实 Agent 接入检查：

```powershell
.venv/Scripts/python.exe -m training.integration_smoke --model globex-executor-v1 --output eval/reports/executor/integration-v1
```

检查采用脚本化 planner、真实微调 executor、生产 MainAgent/fork/上下文治理与生产商品
过滤服务，商品库是显式的合成词法夹具。这证明接入路径，不证明主模型的规划能力。
保存 planner/executor 调用轨迹、父子 run 链接及工具结果消费证据。

首次实际运行发现模型会把夹具品类“测试水杯”简化为“水杯”，导致生产精确品类过滤
返回空结果。`integration-v1` 保留该失败。可另外验证明确提供目录名称的场景：

```powershell
.venv/Scripts/python.exe -m training.integration_smoke --model globex-executor-v1 --output eval/reports/executor/integration-v2 --explicit-category
```

该选项只补充任务中的目录名称说明，所有国家、金额、数量、工具结果与父子关系检查
保持不变。此场景不能替代自然语言品类规范化测试；两次结果都应保留并报告。
实际第二次运行正确消费唯一商品，但最终重复列举该商品。现已增加回答质量检查，
分别报告 `integration_path_passed`、`answer_quality_passed`，总 `passed` 要求两者均通过。
历史 `integration-v2/integration.json` 的旧检查曾显示通过；追加的
`eval/reports/executor/integration_quality_audit.json` 明确判定回答质量未通过，原始记录不覆盖。

正式应用可在进程启动前设置以下环境变量（本次不会修改用户 `.env`）：

```dotenv
EXECUTOR_MODEL_NAME=globex-executor-v1
EXECUTOR_BASE_URL=http://127.0.0.1:8010/v1
EXECUTOR_API_KEY=local-no-key
EXECUTOR_CONTEXT_WINDOW=4096
TRAJECTORY_ROOT=data/trajectories
```

`EXECUTOR_FALLBACK_MODEL` / `EXECUTOR_LITE_MODEL` 可选，均使用 executor endpoint。
它们不会隐式继承 planner 的备用模型。新 endpoint 不继承 planner 密钥。
主子网关共享总请求/token 预算和并发信号量。开启 executor 覆盖或轨迹采集时，
SDK 内部重试关闭，重试由网关执行，
保证逐 attempt 记录与请求预算一致。executor 独立配置时默认上下文窗口 4096，
禁用继承来的供应商显式缓存标记；仍需按实际服务能力设置窗口。

装配顺序为：逻辑调用 scope → 模型路由/并发网关 → 上下文治理 → 实际 attempt
预算 → 轨迹记录 → 模型。应急压缩在治理内部重试也会消耗一次请求额度。
自定义装配时，网关的 `external_attempt_accounting=True` 必须与内层
`ModelAttemptBudgetMiddleware` 成对使用，避免重复计费或漏计。
未开启 executor 覆盖和轨迹采集时，保持原有 SDK 重试、网关计数与缓存命名空间。

## 运行轨迹转训练数据

TRAJECTORY_ROOT 默认关闭，启用后在本地记录实际治理后输入与响应。每个 attempt 一份
JSON；step_id 将重试、fallback 和应急请求归到同一逻辑调用。started/error/cancelled
不会自动进入训练。原始内容可能含个人信息，应保存在受控目录，禁止提交 Git。
生成设置只捕获白名单，不记录 API key 或 headers；自由文本的脱敏仍需显式审核。

审核文件示例（JSON 数组）：

```json
[{
  "attempt_id": "替换成实际32位十六进制ID",
  "approved": true,
  "privacy_reviewed": true,
  "group_id": "session-group-1",
  "split": "train",
  "replacements": {"实际个人字段": "PERSON_1"}
}]
```

```powershell
training/.venv/Scripts/python.exe -m training.export_trajectories --root data/trajectories --reviews data/reviews.json --output data/reviewed-decisions.jsonl
```

仅显式审核通过的 executor 成功 attempt 可以导出；必要时通过 `corrected_target`
提供经验证的修正动作。完整会话不能跨 split，export 额外记录会话哈希检查泄漏。
导出记录的 `target_origin` 区分 `captured_response` 和 `review_correction`，同时保留
原始响应与审核内容的哈希。修正答案不能计入捕获模型的表现。
工具返回先于目标的时序、消息配对、工具可见性和参数 JSON Schema 均需校验。
购物会话的各 fork 会记录相同 session_group_hash；导出时即使人工 group_id 不同，
也不允许同一捕获会话跨 split。没有会话身份的捕获不能直接导出。

可在独立训练环境检查实际 tokenizer 的完整前缀、监督目标与长度，并保存报告：

```powershell
training/.venv/Scripts/python.exe -m training.validate_dataset data/reviewed-decisions.jsonl --tokenizer data/models/Qwen3-0.6B --max-length 3072 --output data/reviewed-validation.json
```

本次实际导出位于 `eval/reports/executor/integration-v2/reviewed-decisions.jsonl`：
同一合成购物会话的两条真实模型调用，一条原始正确工具调用、一条明确修正的回答。
`reviews.json` 保存审核理由，`export-validation.json` 保存 tokenizer 校验结果。
这些运行后样本没有加入本次已完成的训练或冻结测试。

## 回滚

删除/清空 `EXECUTOR_MODEL_NAME`、`EXECUTOR_BASE_URL`、`EXECUTOR_API_KEY`、
`EXECUTOR_CONTEXT_WINDOW`、`EXECUTOR_FALLBACK_MODEL`、`EXECUTOR_LITE_MODEL` 后重启应用，
子 Agent 恢复使用主模型。清空 TRAJECTORY_ROOT 可关闭记录。
停止由自己启动的本地推理服务即可释放 GPU；不要终止其他 Python 进程。
模型发布使用新的不可变版本名，语义响应缓存 namespace 已包含 executor 模型名。
不要在同一版本名后直接替换 adapter；那会让缓存和观测记录失去版本区分。
