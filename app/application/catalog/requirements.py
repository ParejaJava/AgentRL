"""Conservative extraction of explicit shopping constraints with source evidence.

This is a bounded parser for explicit Chinese/English amounts and destinations,
not a general NLU model. Unrecognized language is left to the Agent to clarify;
contradictory recognized values are never guessed. Existing facts survive turns.
"""

import copy
import re

_CURRENCIES = {
    "人民币": "CNY",
    "元人民币": "CNY",
    "欧元": "EUR",
    "美元": "USD",
    "日元": "JPY",
    "英镑": "GBP",
    "CNY": "CNY",
    "EUR": "EUR",
    "USD": "USD",
    "JPY": "JPY",
    "GBP": "GBP",
}
_COUNTRIES = {
    "德国": "DE",
    "美国": "US",
    "日本": "JP",
    "英国": "GB",
    "法国": "FR",
    "中国": "CN",
    "Germany": "DE",
    "United States": "US",
    "Japan": "JP",
    "France": "FR",
    "UK": "GB",
    "DE": "DE",
    "US": "US",
    "JP": "JP",
    "GB": "GB",
    "FR": "FR",
    "CN": "CN",
}
_NUMBER = r"\d+(?:\.\d+)?"
_BRAND = r"(?:品牌[甲乙丙丁戊己庚辛壬癸]|品牌[A-Za-z0-9_-]+|[A-Za-z][A-Za-z0-9_-]*)"
_BUDGET = re.compile(
    rf"(?:预算(?:上限)?|最多花|不超过|不高于|低于|budget(?:\s+of)?|under)\s*[:：为是到改成改为不超过最多]*\s*({_NUMBER})",
    re.IGNORECASE,
)


def _mentions(text: str, vocabulary: dict[str, str]) -> set[str]:
    values = set()
    for word, value in vocabulary.items():
        pattern = re.escape(word)
        if word.isascii():
            pattern = r"(?<![A-Za-z])" + pattern + r"(?![A-Za-z])"
        if re.search(pattern, text, re.IGNORECASE):
            values.add(value)
    return values


def update_requirements(previous: dict, text: str, source_id: str) -> dict:
    """Update only explicit facts; maintain unresolved fields and provenance."""
    result = copy.deepcopy(previous)
    values = result.setdefault("values", {})
    sources = result.setdefault("sources", {})
    pending = set(result.get("pending", []))

    def put(field, value):
        values[field] = value
        sources[field] = {"message_id": source_id, "text": text}
        pending.discard(field)

    if re.search(
        r"(?:取消|不设|没有|不限).{0,3}预算|no budget limit", text, re.IGNORECASE
    ):
        put("price_max_major", None)
        pending.discard("price_max_major")
        pending.discard("target_currency")
    amounts = {float(value) for value in _BUDGET.findall(text)}
    if len(amounts) == 1:
        put("price_max_major", next(iter(amounts)))
    elif len(amounts) > 1:
        pending.add("price_max_major")
    currencies = _mentions(text, _CURRENCIES)
    if len(currencies) == 1:
        put("target_currency", currencies.pop())
    elif len(currencies) > 1:
        pending.add("target_currency")
    if re.search(
        r"(?:币种|货币(?:单位)?).{0,10}(?:不明|未定|未确定|不确定|没确定|不知道)|不知道.{0,8}(?:币种|货币)|currency.{0,10}(?:unknown|unspecified)",
        text,
        re.IGNORECASE,
    ):
        values.pop("target_currency", None)
        pending.add("target_currency")
    if values.get("price_max_major") is not None and not values.get("target_currency"):
        pending.add("target_currency")
    if re.search(r"配送|寄到|送到|寄往|收货|ship|deliver", text, re.IGNORECASE):
        destinations = _mentions(text, _COUNTRIES)
        if len(destinations) == 1:
            put("ship_to", destinations.pop())
        elif len(destinations) > 1:
            pending.add("ship_to")
    basis_text = re.sub(r"不是到手总价|非到手总价|不含税|不含运费", "", text)
    if re.search(
        r"含运费|含税|到手(?:总价|预算)|landed|including shipping",
        basis_text,
        re.IGNORECASE,
    ):
        put("price_basis", "landed")
    elif re.search(r"商品小计|商品总价|subtotal", text, re.IGNORECASE):
        put("price_basis", "subtotal")
    elif re.search(r"单件|标价|单价|unit price", text, re.IGNORECASE):
        put("price_basis", "unit")
    if re.search(
        r"口径.{0,8}(?:未定|未确定|不确定|还没定)|(?:单价|单件).{0,30}(?:还是|也可能|或).{0,15}(?:总价|到手)",
        text,
    ):
        values.pop("price_basis", None)
        pending.add("price_basis")
    for field, pattern in (
        (
            "quantity",
            r"(?:购买|买|purchase)\s*(\d+)\s*(?:件|个|只|套|支|瓶|items?|units?)",
        ),
        (
            "top_k",
            r"(?:最多|给我|推荐|列出|列|top)\s*(\d+)\s*(?:款|个候选|个商品|个|种)?",
        ),
    ):
        found = re.findall(pattern, text, re.IGNORECASE)
        if found:
            distinct = {int(value) for value in found}
            if len(distinct) == 1:
                put(field, distinct.pop())
            else:
                pending.add(field)
    alternatives = re.search(r"(\d+)\s*件\s*(?:还是|或)\s*(\d+)\s*件", text)
    if alternatives and alternatives.group(1) != alternatives.group(2):
        values.pop("quantity", None)
        pending.add("quantity")
    if (
        not amounts
        and re.search(r"(?:预算|金额).{0,40}(?:未确定|不确定|未定|尚未确定)", text)
        and not re.search(r"币种|货币", text)
    ):
        values.pop("price_max_major", None)
        pending.add("price_max_major")
    category = re.search(
        r"(?:目录品类|品类名称|标准品类)(?:名称)?(?:为|是|[:：])\s*[“\"]([^”\"，。；]+)",
        text,
    )
    if category:
        put("category", category.group(1).strip())
    excluded = set(values.get("excluded_brands", []))
    for match in re.finditer(
        rf"(?:排除|不要|不喜欢|dislike)\s*({_BRAND})", text, re.IGNORECASE
    ):
        excluded.add(match.group(1))
    for match in re.finditer(
        rf"(?:撤回|取消).{{0,8}}?({_BRAND}).{{0,8}}(?:排除|不喜欢|黑名单)",
        text,
        re.IGNORECASE,
    ):
        excluded = {
            brand for brand in excluded if brand.casefold() != match.group(1).casefold()
        }
    if excluded or "excluded_brands" in values:
        put("excluded_brands", sorted(excluded))
    brand = re.search(
        rf"(?:只要|只选|必须出自|限定(?:为)?|仅选)\s*({_BRAND})", text, re.IGNORECASE
    )
    if brand:
        put("required_brand", brand.group(1))
    if values.get("required_brand", "").casefold() in {
        brand.casefold() for brand in excluded
    }:
        pending.add("brand_conflict")
    else:
        pending.discard("brand_conflict")
    if values.get("price_basis") == "landed" and not values.get("ship_to"):
        pending.add("ship_to")
    result["pending"] = sorted(pending)
    return result


def check_search_arguments(requirements: dict, arguments: dict) -> dict | None:
    """Reject unresolved or lost constraints; never silently fill/relax model args."""
    pending = requirements.get("pending", [])
    if pending:
        return {
            "status": "needs_clarification",
            "code": "unresolved_requirements",
            "fields": pending,
            "message": "存在未确认或冲突的用户条件，请先追问；不能猜测或放宽条件。",
        }
    expected = requirements.get("values", {})
    mismatches = {}
    for field, value in expected.items():
        actual = arguments.get(
            field,
            {
                "quantity": 1,
                "top_k": 5,
                "price_basis": "unit",
                "excluded_brands": [],
            }.get(field),
        )
        if field == "excluded_brands":
            equal = {x.casefold() for x in value}.issubset(
                {x.casefold() for x in (actual or [])}
            )
        elif isinstance(value, str) and isinstance(actual, str):
            equal = value.casefold() == actual.casefold()
        else:
            equal = value == actual
        if not equal:
            mismatches[field] = value
    if mismatches:
        return {
            "status": "needs_clarification",
            "code": "constraint_mismatch",
            "expected": mismatches,
            "message": "工具参数遗漏或改变了已确认条件；按原条件修正，不得擅自取消。",
        }
    return None
