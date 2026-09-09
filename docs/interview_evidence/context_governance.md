# CTX-001：Cache-aware 上下文治理

## 四层投影

1. 稳定基线：系统提示、工具定义和 Epoch Baseline；同一 Epoch 内不改写。
2. 结构化任务状态：目标、约束、计划、失败工具和稳定引用。
3. 热消息与工作记忆：当前请求、完整工具调用组、近期纠正和增量偏好。
4. 冷事件与 Artifact：完整 Trace 和大工具结果落盘，默认只把引用放入 Prompt。

Breakpoint 不是“从这里以前删除”，而是稳定 Prompt 前缀的边界和哈希。确定性策略先做工具结果卸载、成组保护和候选选择；只有 `full` 模式达到摘要阈值时，LLM 才把 Breakpoint 之后的候选事件压成结构化增量摘要。语义基线改变、压缩次数或模型调用达到规则时滚动 Cache Epoch。

## 三种证据模式

- `off`：原始历史，不安装治理中间件。
- `deterministic`：卸载、裁剪、热区保护和 Epoch 规则生效，但不调用压缩 LLM。
- `full`：确定性治理加结构化 LLM 增量摘要。

三组 8 轮固定会话比较输入 Token、卸载量、压缩次数、Epoch、Breakpoint Hash、约束/状态/偏好保留率、延迟和模型请求数。只有供应商返回真实缓存 Token 时才能写“Prompt Cache 命中率”；否则只能写“Epoch 内稳定前缀 Hash 一致率”。

实验通过正式 `MainAgent` Runtime 驱动，而不是另写一套简化状态机；只把工具集缩减为一个确定性大结果工具，从而隔离测量治理策略。主模型调用、重试以及压缩 LLM 调用共享同一个并发安全预算账本，全部计入 250 次请求与 1,000,000 Token 的 live 总上限。

最新本地候选 `20260908T143344Z-live`：`off` 输入 56,337 Token，`deterministic` 输入 28,995 Token，`full` 输入 38,193 Token；`full` 相比 `off` 降低 32.21%。三种模式信息保留率均为 100%，稳定前缀率均为 100%；`full` 发生 7 次 LLM 摘要，确定性与 full 均卸载 7 个大工具结果。该结果尚需在干净 Commit 上重跑并发布。

## STAR 话术

- S：工具大结果和历史消息持续膨胀，而重写旧前缀会破坏供应商前缀缓存。
- T：在保留当前约束和任务状态的同时降低长会话输入规模。
- A：四层投影、Artifact Offload、确定性策略、增量摘要、Cache Epoch 和双 Breakpoint 哈希。
- R：本地候选已达到 Token 降低 32.21%、信息保留 100%、同 Epoch 前缀稳定率 100%；正式简历数字以干净 Commit 的 `context.json` 为准。
