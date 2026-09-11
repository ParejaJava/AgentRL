"""Validate model selections and render prices only from retrieved product facts."""

import json


def validate_selection(
    text: str, products: list[dict], top_k: int
) -> tuple[list[str], list[str]]:
    """A model chooses IDs; it cannot supply arbitrary prices, SKU facts or prose."""
    try:
        choice = json.loads(text)
    except (ValueError, TypeError):
        return [], ["selection_not_json"]
    if not isinstance(choice, dict) or set(choice) != {"item_ids"}:
        return [], ["selection_schema_invalid"]
    ids = choice["item_ids"]
    if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
        return [], ["selection_ids_invalid"]
    errors = []
    if len(ids) != len(set(ids)):
        errors.append("duplicate_product")
    if len(ids) > top_k:
        errors.append("too_many_products")
    available = {product["item_id"] for product in products}
    if not set(ids).issubset(available):
        errors.append("unsupported_product")
    if not ids and products:
        errors.append("empty_selection_with_candidates")
    return ids, errors


def render_selection(ids: list[str], products: list[dict], quantity: int) -> str:
    """Render an already validated selection using the exact returned primary SKU."""
    cards = {product["item_id"]: product for product in products}
    lines = []
    for ident in ids:
        card = cards[ident]
        price = card["primary_price"]
        sku_id = card.get("primary_sku_id") or next(
            sku["sku_id"] for sku in card["skus"] if sku["stock"] > 0
        )
        line = f"{ident}（{card['title']}，SKU {sku_id}）：单件标价 {price['major']:g} {price['currency']}，购买 {quantity} 件。"
        quote = card.get("landed_price") or {}
        if quote.get("status") == "estimated":
            line += f"估算到手总价 {quote['landed_total_major']:g} {quote['currency']}；以结账时确认结果为准。"
        else:
            line += "到手价暂不可用，标价不是到手总价。"
        lines.append(line)
    return f"找到 {len(ids)} 个候选：\n" + "\n".join(lines)
