"""演示主 AgentLoop fork 多个上下文隔离的子 AgentLoop。"""

import asyncio
import os
import sys
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool

from app.infrastructure.langchain.sub_agents import ForkedAgentLoop, create_fork_tool
from app.infrastructure.llm import get_llm

CHILD_PROMPT = """你是购物调研子 Agent。
先调用平台查询工具获取事实，再给出简短结论。只处理当前子任务，不假设其他线程的信息。
"""

MAIN_PROMPT = """你是购物调研主 Agent。
当任务可并行、需要隔离上下文，或预计调用链不少于三层时，调用 fork_sub_agents。
比较多个平台属于可并行任务：请为每个平台拆分一个完整子任务，一次性 fork，最后汇总结论。
"""


@tool(return_direct=True)
def platform_search(platform: str, query: str) -> str:
    """在指定电商平台查询商品的含运费价格和预计送达时间。"""

    # return_direct=True 让查询结果直接成为子循环最终答案，省去二次模型总结。
    # 使用固定数据模拟外部平台 API，便于观察并发和上下文隔离机制。
    platform_data = {
        "亚马逊": "89 元，预计 2 天送达",
        "Shopee": "65 元，预计 7 天送达",
        "速卖通": "72 元，预计 10 天送达",
    }
    platform_lower = platform.lower()
    if "amazon" in platform_lower or "亚马逊" in platform:
        canonical_platform = "亚马逊"
    elif "shopee" in platform_lower:
        canonical_platform = "Shopee"
    elif "aliexpress" in platform_lower or "速卖通" in platform:
        canonical_platform = "速卖通"
    else:
        canonical_platform = platform

    detail = platform_data.get(canonical_platform, "暂无报价")
    return f"{canonical_platform}上的「{query}」：{detail}"


def build_main_agent() -> Any:
    """组装包含 fork 工具的主 AgentLoop。"""

    # 并发数受模型供应商账户额度约束，可在 .env 中按实际配额调整。
    max_concurrency = int(os.getenv("SUB_AGENT_MAX_CONCURRENCY", "3"))

    # 所有子循环结构相同，但每次调用会获得独立 thread_id 和消息历史。
    sub_agent_loop = ForkedAgentLoop.create(
        model=get_llm(),
        tools=[platform_search],
        system_prompt=CHILD_PROMPT,
        max_concurrency=max_concurrency,
    )
    fork_tool = create_fork_tool(sub_agent_loop)

    # 主 Agent 只持有 fork 工具，不直接接触子循环的工具调用历史。
    return create_agent(
        model=get_llm(),
        tools=[fork_tool],
        system_prompt=MAIN_PROMPT,
        name="shopping_main_agent",
    )


async def main() -> None:
    """运行主循环，并显示主 Agent 调用 fork 工具的过程。"""

    # Windows 的 GBK 终端无法显示部分模型输出字符，替换后可避免打印中断。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    agent = build_main_agent()
    request = "并行调查亚马逊、Shopee、速卖通的旅行收纳袋价格和送达时间，并推荐一个。"

    # 子 Agent 的中间消息不会出现在这里，主循环只看到 fork 工具的最终结果。
    async for step in agent.astream(
        {"messages": [("user", request)]},
        stream_mode="updates",
    ):
        for node_name, update in step.items():
            print(f"\n===== 主循环节点：{node_name} =====")
            for message in update.get("messages", []):
                for tool_call in getattr(message, "tool_calls", []):
                    print(
                        f"[主 Agent 选择工具] {tool_call['name']}，"
                        f"参数：{tool_call['args']}"
                    )
                if message.content:
                    print(f"[{message.type}] {message.content}")


if __name__ == "__main__":
    asyncio.run(main())
