# Task 四工具与确定性子 Agent 派发

## 1. 实现目标

本项目没有引入 AgentScope。参考实现中的 `TaskCreate`、`TaskGet`、`TaskList`、
`TaskUpdate` 被重新实现为 LangChain `@tool`，继续由 LangGraph `create_agent` 驱动。

任务系统负责保存 Work State，LangGraph Checkpointer 负责 Runtime State，上下文治理负责
模型当前看到的信息，三者互不替代：

```text
TaskBoard   = 工作完成到哪里
Checkpoint  = AgentLoop 运行到哪里
Context     = 模型当前看到什么
```

## 2. 分层结构

```text
app/application/tasking/
├── models.py       # Task、TaskBoard、派发结果等框架无关模型
├── ports.py        # TaskBoardRepository、ParallelTaskExecutor 端口
└── service.py      # CRUD、状态机、DAG 校验、认领和结果回写

app/infrastructure/tasking/
└── sqlite_repository.py  # SQLite WAL + 乐观版本仓储

app/infrastructure/langchain/tools/
└── task_tools.py          # 四个 @tool Driver

app/infrastructure/langchain/sub_agents/
└── fork.py                # LangGraph 子循环与确定性派发适配器
```

Application 层不依赖 LangChain、LangGraph、Pydantic 或 SQLite。具体工具和持久化方式均在
Infrastructure 层实现。

## 3. Task 数据

任务保存以下字段：

- `id`：当前 thread 内递增的字符串 ID；
- `subject`、`description`：标题和自包含任务描述；
- `status`：`pending / in_progress / completed / failed`；
- `owner`：当前认领者；
- `blocked_by`：当前任务依赖的任务 ID；
- `result`、`error`：最终结果或失败原因；
- `metadata`：扩展结构化数据；
- `created_at`、`updated_at`：UTC 时间。

仓储只保存 `blocked_by`。`blocks`、`active_blocked_by`、`runnable` 和
`effective_status=blocked` 均在读取时计算，避免双向依赖不一致。

## 4. 四个工具

### TaskCreate

创建 `pending` 任务。任务 ID 由 SQLite 事务和看板版本共同保护，并发创建不会重复。

```json
{
  "subject": "搜索亚马逊候选",
  "description": "搜索符合预算和收货地约束的旅行收纳袋"
}
```

### TaskGet

按 ID 获取完整任务。开始工作前应检查 `active_blocked_by` 和 `runnable`。

```json
{"task_id": "1"}
```

### TaskList

列出当前 LangGraph `thread_id` 下的整个任务看板。多个 `runnable=true` 的任务是并行
派发候选。

### TaskUpdate

修改状态、内容、owner、metadata、result 和依赖。

```json
{"task_id": "2", "add_blocked_by": ["1"]}
```

```json
{
  "task_id": "1",
  "status": "completed",
  "result": "已找到 8 个候选商品"
}
```

`metadata` 中值为 `null` 的键会被删除。`status=deleted` 会永久删除任务并清理相关
依赖。

## 5. DAG 与状态约束

新增依赖时会执行以下确定性检查：

1. 两端任务必须存在；
2. 任务不能依赖自身；
3. 新增边后不得形成环；
4. 有未完成依赖的任务不能进入 `in_progress`；
5. 上游任务完成后，下游任务的 `runnable` 会自动变为 `true`。

主要状态流转：

```text
pending ───────→ in_progress ───────→ completed
   │                  │
   └────→ failed ←────┘
             │
             └────→ pending / in_progress
```

`completed` 是终态；如果任务定义错误，可使用 `deleted` 删除后重新创建。

## 6. parallel 的确定性派发

生产装配中的 `fork_sub_agents(reason="parallel")` 不再接受任意自然语言任务绕过
TaskBoard，必须传入至少两个 `task_ids`：

```json
{
  "reason": "parallel",
  "task_ids": ["1", "2"]
}
```

执行顺序如下：

```text
TaskList 找到 runnable 任务
        ↓
fork_sub_agents(task_ids, parallel)
        ↓
一个 SQLite 事务内再次校验状态、owner、依赖
        ↓
原子设置 in_progress 和唯一 owner
        ↓
LangGraph 子 AgentLoop 按并发上限隔离执行
        ↓
逐项保留成功或失败，不因单个异常丢失整批结果
        ↓
一个事务内回写 completed/result 或 failed/error
        ↓
TaskList 查找新解锁任务
```

因此，“可并行”不再只是 LLM 填写的理由。代码至少保证：

- 任务均存在且为 `pending`；
- 任务没有 owner；
- 所有活动依赖都已完成；
- `parallel` 至少包含两个任务；
- 任务只会被一个派发批次认领。

资源互斥（例如两个任务写同一文件）尚未建模。需要这一能力时，可以在 metadata 中
增加资源声明，再由派发策略检查资源冲突。

## 7. MainAgent 与子 Agent 的权限

`MainAgent` 现在区分两类工具：

```python
MainAgent(
    tools=shared_business_tools,
    main_only_tools=task_tools,
)
```

- MainAgent：业务工具、Task 四工具、`fork_sub_agents`；
- Forked SubAgent：只有业务工具；
- 子 Agent 不能修改 TaskBoard，也不能继续 fork。

主 Agent 和子 Agent 使用不同系统提示词。任务规划、状态维护和派发属于 MainAgent，
子 Agent 只执行收到的自包含任务。

## 8. 会话隔离与持久化

任务以 `thread_id` 为 scope，数据库默认位于：

```text
data/sessions/task_boards.db
```

同一个 `thread_id` 的后续 Agent Run 可以继续读取任务；不同 thread 完全隔离。
`shopping_session_id` 仍用于购物身份和事件分区，不用于跨 thread 合并任务。当前实现是
会话内 Work State，不是跨会话长期记忆。

SQLite 使用 WAL、`BEGIN IMMEDIATE` 和看板版本号。发生并发版本冲突时，Application
服务会重新读取并重放修改，保证顺序 ID、依赖和认领状态不丢失。

## 9. 一次典型调用

```text
TaskCreate：搜索 Amazon
TaskCreate：搜索 Shopee
TaskCreate：比较结果
TaskUpdate：让“比较结果” blocked_by 前两个任务
TaskList：前两个任务 runnable=true
fork_sub_agents：task_ids=[1, 2], reason=parallel
TaskList：前两个 completed，“比较结果”被自动解锁
TaskUpdate：把比较任务设为 in_progress
执行比较
TaskUpdate：completed + result
```

## 10. 验证

```powershell
uv run ruff check app tests
uv run pytest
```

测试覆盖 CRUD、会话隔离、metadata 合并、状态流转、自依赖、循环依赖、自动解锁、
20 路并发创建、原子认领、部分失败回写、工具声明以及 MainAgent/子 Agent 工具隔离。
