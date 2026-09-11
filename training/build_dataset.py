"""Author synthetic executor decisions against the application's real tool schemas.

Run with the application Python. No model/API calls. Category entities and surface
templates are disjoint across splits. Shared task families are intentional; this
is a small controlled generalization test, not a real-user benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_core.utils.function_calling import convert_to_openai_tool

from app.domain.catalog.money import Money
from app.infrastructure.langchain.prompts import SUB_AGENT_SYSTEM_PROMPT
from app.infrastructure.langchain.tools.category_insight import (
    create_category_insight_tool,
)
from app.infrastructure.langchain.tools.product_search import (
    ProductSearchToolInput,
    create_item_search_tool,
)
from app.infrastructure.langchain.tools.web_search import create_web_search_tool
from training.data import digest, validate_splits, write_jsonl

ENTITIES = {
    "train": [
        "咖啡杯",
        "旅行三件套",
        "威士忌酒杯",
        "保温壶",
        "帆布包",
        "登山杖",
        "充电器",
        "台灯",
        "收纳盒",
        "雨伞",
        "运动水壶",
        "枕头",
        "手套",
        "鼠标",
        "行李箱",
        "茶壶",
    ],
    "dev": ["咖啡机", "瑜伽垫", "餐具", "靠垫"],
    "test": ["蓝牙音箱", "露营椅", "电动牙刷", "背包", "键盘", "望远镜"],
}
COUNTRIES = [
    ("德国", "DE", "欧元", "EUR"),
    ("日本", "JP", "日元", "JPY"),
    ("美国", "US", "美元", "USD"),
    ("英国", "GB", "英镑", "GBP"),
]
SEARCH_TEMPLATES = {
    "train": "帮我找{category}，寄到{country}，预算{budget}{currency}，给我{count}个候选。",
    "dev": "收货地址在{country}；我需要{count}款{category}，每件不能超过{budget}{currency}。",
    "test": "要买的是{category}。请筛掉不配送{country}以及标价高于{budget}{currency}的款，最后列{count}款。",
}


def call(name: str, args: dict, ident: str = "call-1") -> dict:
    return {
        "id": ident,
        "type": "function",
        "function": {"name": name, "arguments": args},
    }


def assistant_call(name: str, args: dict) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [call(name, args)]}


def tool_history(name: str, args: dict, result: dict) -> list[dict]:
    return [
        assistant_call(name, args),
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": json.dumps(result, ensure_ascii=False),
        },
    ]


def build() -> list[dict]:
    # Factories are used only to extract the production @tool schemas.
    tools = [
        convert_to_openai_tool(t)
        for t in (
            create_item_search_tool(None),
            create_category_insight_tool(None),
            create_web_search_tool(None),
        )
    ]
    rows = []
    for split, categories in ENTITIES.items():
        for i, category in enumerate(categories):
            country, code, currency, currency_code = COUNTRIES[i % len(COUNTRIES)]
            budget = (i + 2) * (1000 if currency_code == "JPY" else 15)
            count = i % 3 + 2
            base_args = {
                "normalized_query": category,
                "category": category,
                "ship_to": code,
                "price_max_major": budget,
                "target_currency": currency_code,
                "top_k": count,
            }
            ProductSearchToolInput.model_validate(base_args)
            query = SEARCH_TEMPLATES[split].format(
                category=category,
                country=country,
                budget=budget,
                currency=currency,
                count=count,
            )

            def add(
                family: str,
                user: str,
                target: dict,
                *,
                history=None,
                expected_args=None,
                required=None,
                forbidden=None,
                prefix=None,
                split=split,
                i=i,
            ) -> None:
                messages = [{"role": "system", "content": SUB_AGENT_SYSTEM_PROMPT}]
                if prefix:
                    messages.append({"role": "system", "content": prefix})
                messages.append({"role": "user", "content": user})
                messages.extend(history or [])
                rows.append(
                    {
                        "case_id": f"{split}-{i:02d}-{family}",
                        "group_id": f"{split}-entity-{i:02d}",
                        "split": split,
                        "source": "authored_synthetic",
                        "family": family,
                        "messages": messages,
                        "tools": tools,
                        "target": target,
                        "rubric": {
                            "expected_args": expected_args or {},
                            "required_any": required or [],
                            "forbidden": forbidden or [],
                        },
                    }
                )

            add(
                "search",
                query,
                assistant_call("item_search", base_args),
                expected_args={
                    k: v for k, v in base_args.items() if k != "normalized_query"
                },
            )
            simple = {"normalized_query": category, "category": category, "top_k": 5}
            add(
                "simple",
                {
                    "train": f"搜一下{category}。",
                    "dev": f"有哪些{category}可选？",
                    "test": f"先查商品库里的{category}，其他筛选条件暂不限定。",
                }[split],
                assistant_call("item_search", simple),
                expected_args={"category": category},
            )
            updated = {**base_args, "price_max_major": budget * 2}
            update_text = {
                "train": f"预算改为{budget * 2}{currency}，其他不变。",
                "dev": f"刚才金额太低，现在上限是{budget * 2}{currency}。",
                "test": f"沿用之前的配送地和数量，唯独价格限制调整到{budget * 2}{currency}。",
            }[split]
            add(
                "modify",
                query,
                assistant_call("item_search", updated),
                history=[
                    {"role": "assistant", "content": "已记下筛选要求。"},
                    {"role": "user", "content": update_text},
                ],
                expected_args={
                    k: v for k, v in updated.items() if k != "normalized_query"
                },
            )
            for depth in ("quick", "deep"):
                wording = {
                    "train": ("有什么常见选购要点", "详细讲讲材质、参数和属性如何选择"),
                    "dev": ("通常怎么挑选", "想了解不同材质与规格的详细差异"),
                    "test": (
                        "先了解基础挑选常识",
                        "请深入分析各项技术指标和材料的取舍",
                    ),
                }[split][depth == "deep"]
                add(
                    f"category-{depth}",
                    f"{category}{wording}，暂时不查具体商品。",
                    assistant_call(
                        "category_insight", {"category": category, "depth": depth}
                    ),
                    expected_args={"category": category, "depth": depth},
                )
            add(
                "clarify",
                {
                    "train": f"找{category}，预算50，不知道用哪种币种，先问清楚再搜。",
                    "dev": f"我给{category}的上限是50，但还没决定按哪种货币计价，请先确认。",
                    "test": f"{category}最多花50；这个数字的货币单位尚未确定，确认好才能检索。",
                }[split],
                {"role": "assistant", "content": "请确认预算使用哪种币种？"},
                required=[["币种", "货币", "欧元", "美元"]],
            )
            add(
                "empty",
                query,
                {
                    "role": "assistant",
                    "content": f"当前没有同时满足预算和配送条件的{category}。你愿意提高预算或调整配送条件吗？",
                },
                history=tool_history(
                    "item_search",
                    base_args,
                    {
                        "status": "ok",
                        "items": [],
                        "total_candidates": 0,
                        "fixture_source": "authored_synthetic",
                    },
                ),
                required=[["没有", "未找到", "暂无", "无符合"], ["预算", "条件"]],
            )
            add(
                "error",
                query,
                {
                    "role": "assistant",
                    "content": "商品搜索连续失败，暂时无法确认可购买商品和价格。建议稍后重试。",
                },
                history=tool_history(
                    "item_search",
                    base_args,
                    {
                        "status": "error",
                        "error": "circuit_open",
                        "message": "搜索连续失败，工具暂时熔断，勿重复调用",
                    },
                ),
                required=[["失败", "不可用", "无法", "异常"], ["稍后", "重试", "恢复"]],
            )
            product_id = f"FIXTURE-{split}-{i:02d}"
            price = budget - 1
            add(
                "grounded",
                f"查询{category}后告诉我商品编号与标价，不要承诺到手总价。",
                {
                    "role": "assistant",
                    "content": f"候选商品 {product_id}（{category}）标价为 {price} {currency_code}。"
                    "运费和关税尚未确认，这不是到手总价或结算承诺。",
                },
                history=tool_history(
                    "item_search",
                    simple,
                    {
                        "status": "ok",
                        "items": [
                            {
                                "rank": 1,
                                "product": {
                                    "item_id": product_id,
                                    "title": category,
                                    "primary_price": Money.from_major_units(
                                        price, currency_code
                                    ).to_dict(),
                                },
                            }
                        ],
                        "fixture_source": "authored_synthetic",
                    },
                ),
                required=[
                    [product_id],
                    [str(price)],
                    [currency_code],
                    ["运费", "关税", "到手"],
                ],
                forbidden=["已下单", "已支付", "保证到手"],
            )
            add(
                "compressed",
                f"按已经确认的条件搜{category}。",
                assistant_call("item_search", base_args),
                prefix="<active_context>"
                + json.dumps(
                    {
                        "task_state": {
                            "goal": query,
                            "constraints": [f"{k}={v}" for k, v in base_args.items()],
                        },
                        "task_delta": {},
                        "working_memory": [],
                    },
                    ensure_ascii=False,
                )
                + "</active_context>",
                expected_args={
                    k: v for k, v in base_args.items() if k != "normalized_query"
                },
            )
            web_query = f"{country} {category} 进口政策 最新 官方"
            add(
                "preference-conflict",
                {
                    "train": f"买{category}，只看品牌甲，但以前的品牌黑名单也别撤销。",
                    "dev": f"这次{category}要品牌甲，同时继续保留原先所有品牌排除条件。",
                    "test": f"筛选{category}时，历史不喜欢的品牌继续排除；另外限定候选必须出自品牌甲。",
                }[split],
                {
                    "role": "assistant",
                    "content": "你之前将品牌甲列为不喜欢，但本次又限定品牌甲，条件存在冲突。请确认本次是否允许品牌甲？",
                },
                prefix='<active_context>{"task_state":{},"task_delta":{},"working_memory":[{"id":"buyer-preference:fixture","kind":"preference","content":"[dislike] 品牌甲"}]}</active_context>',
                required=[["品牌甲"], ["确认", "冲突", "矛盾"]],
            )
            add(
                "category-then-search",
                query,
                assistant_call("item_search", base_args),
                history=tool_history(
                    "category_insight",
                    {"category": category, "depth": "quick"},
                    {
                        "status": "ok",
                        "category": category,
                        "depth": "quick",
                        "summary": "比较材质和规格；具体库存、价格需查询商品工具。",
                        "fixture_source": "authored_synthetic",
                    },
                ),
                expected_args={
                    k: v for k, v in base_args.items() if k != "normalized_query"
                },
            )
            add(
                "retry-once",
                query,
                assistant_call("item_search", base_args),
                history=tool_history(
                    "item_search",
                    base_args,
                    {
                        "status": "error",
                        "error": "timeout",
                        "retryable": True,
                        "message": "第一次搜索超时，允许重试一次",
                    },
                ),
                expected_args={
                    k: v for k, v in base_args.items() if k != "normalized_query"
                },
            )
            add(
                "web",
                {
                    "train": f"查一下{country}关于{category}的最新进口政策，要官方来源。",
                    "dev": f"需要{category}进口到{country}的现行政策信息，找官方依据。",
                    "test": f"请核实近期{country}对进口{category}的政策有何规定，并保留官方来源。",
                }[split],
                assistant_call("web_search", {"query": web_query, "max_results": 5}),
                required=[[country], [category]],
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("eval/executor"))
    parser.add_argument(
        "--replace-dataset",
        action="store_true",
        help="Explicitly replace a changed dataset before any held-out evaluation",
    )
    args = parser.parse_args()
    rows = build()
    manifest = validate_splits(rows)
    existing = args.output / "manifest.json"
    if existing.exists() and not args.replace_dataset:
        old = json.loads(existing.read_text(encoding="utf-8"))
        if old["dataset_sha256"] != manifest["dataset_sha256"]:
            raise ValueError(
                "Dataset changed: use a new output directory or explicitly replace before evaluation"
            )
    for split in ENTITIES:
        selected = [r for r in rows if r["split"] == split]
        write_jsonl(args.output / f"{split}.jsonl", selected)
        manifest[f"{split}_sha256"] = digest(selected)
    manifest.update(
        {
            "source": "authored_synthetic",
            "entities": ENTITIES,
            "tool_schema_sha256": digest(rows[0]["tools"]),
            "limitations": "Small synthetic benchmark; shared task families, disjoint entities and surface templates. Keyword checks do not prove semantic correctness.",
        }
    )
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest["counts"]))


if __name__ == "__main__":
    main()
