"""Persist explicit requirements and validate actual shopping execution boundaries."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Command

from app.application.catalog.recommendations import render_selection, validate_selection
from app.application.catalog.requirements import (
    check_search_arguments,
    update_requirements,
)
from app.infrastructure.context import require_context, reset_context, set_context
from app.infrastructure.context_governance.schemas import SessionAgentState


class ShoppingGuardMiddleware(AgentMiddleware):
    """No extra model calls; keep raw model captures distinct from guarded answers."""

    state_schema = SessionAgentState

    def __init__(self, audit_root: Path | None = None) -> None:
        self.audit_root = audit_root
        # Only this middleware can issue a receipt, after rendering tool facts.
        # Scope by parent run and bound memory; text alone is not an attestation.
        self._child_receipts: dict[tuple[str, str], str] = {}

    async def abefore_model(
        self, state: SessionAgentState, runtime: Any
    ) -> dict | None:
        context = runtime.context
        prior = state.get("shopping_requirements")
        ledger = copy.deepcopy(
            prior or getattr(context, "confirmed_requirements", None) or {}
        )
        seen = set(ledger.get("seen_messages", []))
        # Fork task text is written by the planner, not by the buyer. It cannot
        # resolve pending facts or replace the trusted parent's requirements.
        if (
            getattr(context, "parent_run_id", None)
            and getattr(context, "confirmed_requirements", None) is not None
        ):
            return {"shopping_requirements": ledger}
        preferences_seen = set(ledger.get("preferences_seen", []))
        for ident, entry in state.get("working_memory", {}).items():
            content = entry.get("content", "")
            if ident not in preferences_seen and content.startswith("[dislike] "):
                statement = content.removeprefix("[dislike] ").strip()
                ledger = update_requirements(ledger, "不要" + statement, ident)
                preferences_seen.add(ident)
        ledger["preferences_seen"] = sorted(preferences_seen)
        for message in state.get("messages", []):
            if not isinstance(message, HumanMessage):
                continue
            ident = message.id or hashlib.sha256(message.text.encode()).hexdigest()
            if ident in seen:
                continue
            ledger = update_requirements(ledger, message.text, ident)
            ledger["revision"] = ident
            seen.add(ident)
        ledger["seen_messages"] = sorted(seen)
        return {"shopping_requirements": ledger}

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        ledger = request.state.get("shopping_requirements", {})
        name = request.tool_call["name"]

        def record(result: ToolMessage) -> Command[Any] | ToolMessage:
            try:
                payload = json.loads(result.text)
            except ValueError:
                payload = None
            if not isinstance(payload, dict):
                payload = {
                    "status": "error",
                    "message": "工具返回格式无效，无法确认商品事实。",
                    "raw_result_sha256": hashlib.sha256(
                        result.text.encode()
                    ).hexdigest(),
                }
            return Command(
                update={
                    "messages": [result],
                    "shopping_evidence": {
                        request.tool_call["id"]: {
                            "payload": payload,
                            "arguments": request.tool_call.get("args", {}),
                            "requirements": ledger.get("values", {}),
                            "revision": ledger.get("revision"),
                            "tool_call_id": request.tool_call["id"],
                        }
                    },
                }
            )

        if name == "item_search":
            failure = check_search_arguments(ledger, request.tool_call.get("args", {}))
            if failure:
                return record(
                    ToolMessage(
                        content=json.dumps(failure, ensure_ascii=False),
                        name=name,
                        tool_call_id=request.tool_call["id"],
                    )
                )
        context = require_context()
        token = set_context(
            replace(context, confirmed_requirements=copy.deepcopy(ledger))
        )
        try:
            result = await handler(request)
        finally:
            reset_context(token)
        if name == "resolve_product_category" and isinstance(result, ToolMessage):
            try:
                resolution = json.loads(result.text)
            except ValueError:
                return result
            if resolution.get("status") == "resolved" and ledger.get("values", {}).get(
                "category"
            ) == request.tool_call.get("args", {}).get("category"):
                updated = copy.deepcopy(ledger)
                updated["values"]["category"] = resolution["category"]
                updated.setdefault("sources", {})["category"] = {
                    "tool_call_id": request.tool_call["id"],
                    "source": resolution.get("source", "catalog"),
                }
                return Command(
                    update={"messages": [result], "shopping_requirements": updated}
                )
        if name == "item_search" and isinstance(result, ToolMessage):
            return record(result)
        if name == "fork_sub_agents" and isinstance(result, ToolMessage):
            try:
                payload = json.loads(result.text)
            except ValueError:
                return result
            answers = [
                entry.get("answer", entry.get("result"))
                for entry in payload.get("results", payload.get("tasks", []))
            ]
            keys = [
                (context.run_id, hashlib.sha256(answer.encode()).hexdigest())
                for answer in answers
                if isinstance(answer, str)
            ]
            if (
                keys
                and len(keys) == len(answers)
                and all(key in self._child_receipts for key in keys)
            ):
                protected = list(
                    dict.fromkeys(self._child_receipts[key] for key in keys)
                )
                for key in keys:
                    self._child_receipts.pop(key, None)
                return Command(
                    update={
                        "messages": [result],
                        "shopping_evidence": {
                            request.tool_call["id"]: {
                                "payload": {"protected_answers": protected},
                                "arguments": request.tool_call.get("args", {}),
                                "requirements": ledger.get("values", {}),
                                "revision": ledger.get("revision"),
                                "tool_call_id": request.tool_call["id"],
                            }
                        },
                    }
                )
        return result

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        ledger = request.state.get("shopping_requirements", {})
        if ledger.get("values") or ledger.get("pending"):
            instructions = json.dumps(
                {
                    "values": ledger.get("values", {}),
                    "pending": ledger.get("pending", []),
                },
                ensure_ascii=False,
            )
            original = request.system_message.text if request.system_message else ""
            request = request.override(
                system_message=SystemMessage(
                    content=original
                    + "\n<confirmed_shopping_constraints>"
                    + instructions
                    + "</confirmed_shopping_constraints>"
                )
            )
        response = await handler(request)
        evidence = request.state.get("shopping_evidence", {})
        relevant = [
            e
            for e in evidence.values()
            if e["requirements"] == ledger.get("values", {})
            and e.get("revision") == ledger.get("revision")
        ]
        # A serial retry supersedes its previous result. Concurrent searches in
        # the same tool-call batch are handled together, never by dict order.
        for prior in reversed(request.messages):
            if isinstance(prior, AIMessage):
                ids = {
                    call["id"]
                    for call in prior.tool_calls
                    if call["name"] in {"item_search", "fork_sub_agents"}
                }
                if ids:
                    relevant = [e for e in relevant if e["tool_call_id"] in ids]
                    break
        if not relevant or len(response.result) != 1:
            return response
        message = response.result[0]
        if not isinstance(message, AIMessage) or message.tool_calls:
            return response
        # Multiple parallel searches must not be silently combined across incompatible specs.
        signatures = {json.dumps(e["arguments"], sort_keys=True) for e in relevant}
        current = relevant[-1]
        products = [item["product"] for item in current["payload"].get("items", [])]
        top_k = current["arguments"].get("top_k", 5)
        ids, errors = validate_selection(message.text, products, top_k)
        if len(signatures) != 1:
            ids, errors = [], ["incompatible_search_scopes"]
            rendered = "本轮包含不同筛选条件的搜索结果，请确认要按哪一组条件推荐商品。"
        elif "protected_answers" in current["payload"]:
            ids = []
            rendered = "\n\n".join(current["payload"]["protected_answers"])
            errors = (
                []
                if message.text == rendered
                else ["parent_rewrite_replaced_by_verified_child_answer"]
            )
        elif "items" not in current["payload"]:
            ids, errors = [], ["search_not_successful"]
            payload = current["payload"]
            if payload.get("status") == "needs_clarification":
                rendered = "搜索条件尚未确认：" + str(
                    payload.get("message", "请确认缺失或冲突的条件。")
                )
                labels = {
                    "target_currency": "预算币种",
                    "ship_to": "配送国家",
                    "quantity": "购买件数",
                    "price_basis": "预算口径",
                    "price_max_major": "预算金额",
                    "brand_conflict": "品牌冲突",
                }
                if payload.get("fields"):
                    rendered += (
                        "待确认："
                        + "、".join(
                            labels.get(field, field) for field in payload["fields"]
                        )
                        + "。"
                    )
            else:
                rendered = "商品搜索或计价暂未成功，无法确认商品和价格。" + str(
                    payload.get("message", "请稍后重试。")
                )
        elif not products:
            ids = []
            # An empty recall is not evidence of stock or price across the catalog.
            rendered = "当前检索范围内没有符合已确认条件的商品，未更改筛选条件。"
            reasons = sorted(
                {item["reason"] for item in current["payload"].get("filtered_out", [])}
            )
            labels = {
                "over_price_cap": "超过预算",
                "ship_to_unavailable": "不支持配送目的地",
                "insufficient_stock": "库存不足以满足购买数量",
                "no_available_sku": "没有可用 SKU",
                "excluded_brand": "命中品牌排除条件",
                "brand_mismatch": "不符合指定品牌",
                "category_mismatch": "品类不匹配",
                "pricing_unavailable": "无法核验价格",
            }
            if reasons:
                rendered += (
                    "部分候选的已知过滤原因："
                    + "、".join(labels.get(r, r) for r in reasons)
                    + "。"
                )
            rendered += "是否愿意调整条件后再查询？"
        else:
            if errors:
                ids = list(dict.fromkeys(product["item_id"] for product in products))[
                    :top_k
                ]
            rendered = render_selection(
                ids, products, current["arguments"].get("quantity", 1)
            )
        audit = {
            "raw_selection_valid": not errors,
            "errors": errors,
            "raw_response_sha256": hashlib.sha256(message.text.encode()).hexdigest(),
            "tool_call_id": current["tool_call_id"],
            "rendered_item_ids": ids,
            "has_candidates": bool(products),
        }
        context = require_context()
        if context.parent_run_id:
            key = (context.parent_run_id, hashlib.sha256(rendered.encode()).hexdigest())
            self._child_receipts[key] = rendered
            while len(self._child_receipts) > 256:
                self._child_receipts.pop(next(iter(self._child_receipts)))
        if self.audit_root:
            self.audit_root.mkdir(parents=True, exist_ok=True)
            (self.audit_root / f"{uuid4().hex}.json").write_text(
                json.dumps(
                    {**audit, "thread_id": context.thread_id, "run_id": context.run_id},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        replacement = message.model_copy(
            update={
                "content": rendered,
                "additional_kwargs": {
                    **message.additional_kwargs,
                    "shopping_guard": audit,
                },
            }
        )
        return ModelResponse(
            result=[replacement], structured_response=response.structured_response
        )
