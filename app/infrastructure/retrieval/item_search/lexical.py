"""商品向量链路不可用时使用的确定性轻量词法评分。"""

import re

_TOKEN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]", re.IGNORECASE)


def lexical_terms(text: str) -> set[str]:
    """为中英文文本生成单词、中文单字和连续中文二元组。"""

    tokens = _TOKEN.findall(text.casefold())
    chinese = [token for token in tokens if "\u4e00" <= token <= "\u9fff"]
    bigrams = {
        "".join(chinese[index : index + 2])
        for index in range(max(0, len(chinese) - 1))
    }
    return set(tokens) | bigrams


def keyword_2gram_score(query: str, document: str) -> float:
    """返回 0 到 1 的词项覆盖率，并为完整短语命中提供稳定加权。"""

    normalized_query = query.strip().casefold()
    if not normalized_query:
        return 0.0
    normalized_document = document.casefold()
    query_terms = lexical_terms(normalized_query)
    if not query_terms:
        return 0.0
    overlap = len(query_terms & lexical_terms(normalized_document)) / len(query_terms)
    phrase_bonus = 0.2 if normalized_query in normalized_document else 0.0
    return round(min(1.0, overlap * 0.8 + phrase_bonus), 6)
