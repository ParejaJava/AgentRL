"""后台 Agent 任务消费者入口。

Redis Stream 适配器尚未实现；Worker 入口先保持独立，后续由 Composition
注入 `TaskQueue` 实现，不允许在这里直接依赖 Redis 客户端。
"""


async def run_worker() -> None:
    """在队列适配器实现前明确拒绝启动空 Worker。"""

    raise RuntimeError("TaskQueue 基础设施尚未配置")
