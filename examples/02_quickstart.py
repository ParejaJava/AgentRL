"""演示购物 Agent 连续调用“搜索”和“比价”工具的多轮循环。"""

import sys
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool

from app.agent.llm import get_llm

# 系统提示词不仅定义角色，也告诉模型多步骤任务应按什么顺序执行。
SYSTEM_PROMPT = """你是一个全球购物助手。
当用户同时要求搜索和比价时，必须先调用 item_search，等待搜索结果，
再从搜索结果中选择符合要求的商品名称并调用 price_compare，最后汇总回答。
"""


@tool
def item_search(query: str) -> str:
    """在全球电商平台搜索商品，返回匹配的商品列表。"""

    # 当前返回模拟数据；接入真实 API 时只需替换此处的函数体。
    return f"找到 3 件匹配「{query}」的商品：旅行收纳袋、防水洗漱包、便携衣物袋"


@tool
def price_compare(item_name: str) -> str:
    """跨平台比较商品价格，返回各平台包含运费的到手价。"""

    # 当前使用固定价格模拟多个平台；实际项目中可替换为比价 API。
    return f"「{item_name}」价格：亚马逊 89 元，Shopee 65 元（含运费），速卖通 72 元"


def build_agent() -> Any:
    """将统一模型、工具集和系统提示词组装成可循环执行的 Agent。"""

    # create_agent 底层使用 LangGraph，并自动处理“模型节点 ↔ 工具节点”循环。
    return create_agent(
        model=get_llm(),
        # 两个工具同时注册后，模型会根据工具描述决定调用对象和调用顺序。
        tools=[item_search, price_compare],
        system_prompt=SYSTEM_PROMPT,
    )


def print_step(step: dict[str, Any]) -> None:
    """打印一次图状态更新，使模型和工具节点的执行过程清晰可见。"""

    for node_name, update in step.items():
        print(f"\n===== 执行节点：{node_name} =====")

        for message in update.get("messages", []):
            # AIMessage.tool_calls 保存模型请求执行的工具名称和参数。
            for tool_call in getattr(message, "tool_calls", []):
                print(
                    f"[模型选择工具] {tool_call['name']}，参数：{tool_call['args']}"
                )

            if message.content:
                print(f"[{message.type}] {message.content}")


def main() -> None:
    """运行一次完整 Agent 循环，并逐节点输出执行过程。"""

    # Windows 的 GBK 终端无法显示部分货币符号或 Emoji，使用替换模式避免中断循环。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    agent = build_agent()

    # 该请求需要先搜索再比价，因此会形成多轮“模型 → 工具”循环。
    # stream_mode="updates" 会在每个图节点执行后返回该节点产生的增量状态。
    for step in agent.stream(
        {"messages": [("user", "帮我搜旅行收纳袋，然后跨平台比个价")]},
        stream_mode="updates",
    ):
        print_step(step)


if __name__ == "__main__":
    main()
