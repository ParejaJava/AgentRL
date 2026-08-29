"""结构化知识卡片的 JSONL 存储与开发期本地检索。"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

from app.application.catalog.category_insight_models import RetrievedCategoryCard
from app.domain.catalog import CategoryCard

from .schemas import category_card_from_dict, category_card_to_dict

_WORD = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]", re.IGNORECASE)


class JsonlCategoryCardStore:
    """把一次性结构化结果持久化，供多次在线查询复用。"""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> tuple[CategoryCard, ...]:
        """加载全部卡片；尚未执行摄取时返回空集合。"""

        if not self.path.exists():
            return ()
        cards: list[CategoryCard] = []
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                cards.append(category_card_from_dict(raw))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"无效知识卡片：{self.path}:{line_number}: {exc}"
                ) from exc
        return tuple(cards)

    def save(self, cards: Sequence[CategoryCard]) -> None:
        """按 card_id 排序并原子替换 JSONL，避免读到半写入文件。"""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        content = "\n".join(
            json.dumps(category_card_to_dict(card), ensure_ascii=False)
            for card in sorted(cards, key=lambda item: item.card_id)
        )
        temporary.write_text(f"{content}\n" if content else "", encoding="utf-8")
        temporary.replace(self.path)


class LocalCategoryCardRetriever:
    """OpenSearch 接入前使用的确定性词法检索适配器。"""

    def __init__(self, store: JsonlCategoryCardStore) -> None:
        self._store = store

    def search(self, category: str, limit: int) -> Sequence[RetrievedCategoryCard]:
        """检索具体品类卡片，并只在命中后补充全局跨境通则。"""

        if limit < 1:
            raise ValueError("limit 必须大于 0")
        query = category.strip().casefold()
        if not query:
            raise ValueError("category 不能为空")

        specific: list[RetrievedCategoryCard] = []
        general: list[RetrievedCategoryCard] = []
        for card in self._store.load():
            if card.applies_to_all_categories:
                general.append(
                    RetrievedCategoryCard(
                        card=card,
                        score=0.1,
                        recall_strategy="keyword_2gram",
                    )
                )
                continue
            score = self._score(query, card)
            if score > 0:
                specific.append(
                    RetrievedCategoryCard(
                        card=card,
                        score=score,
                        recall_strategy="keyword_2gram",
                    )
                )
        if not specific:
            return ()
        specific.sort(key=lambda item: (item.score, item.card.confidence), reverse=True)
        general.sort(key=lambda item: item.card.confidence, reverse=True)
        return tuple([*specific, *general][:limit])

    @classmethod
    def _score(cls, query: str, card: CategoryCard) -> float:
        """用精确包含和字符/单词覆盖率产生 0 到 1 的临时相关度。"""

        corpus = " ".join(
            (
                card.category,
                card.summary,
                *card.raw_evidence,
                *(item.name for item in card.bestsellers),
                *(item.attribute for item in card.attributes),
                *(item.description for item in card.price_tiers),
            )
        ).casefold()
        if query == card.category.casefold():
            return 1.0
        if query in corpus or card.category.casefold() in query:
            return 0.95
        terms = cls._terms(query)
        if not terms:
            return 0.0
        overlap = len(terms & cls._terms(corpus)) / len(terms)
        return round(min(overlap, 0.9), 4) if overlap >= 0.5 else 0.0

    @staticmethod
    def _terms(text: str) -> set[str]:
        """为中英文内容生成轻量 unigram/bigram 词项。"""

        tokens = _WORD.findall(text.casefold())
        chinese = [token for token in tokens if "\u4e00" <= token <= "\u9fff"]
        bigrams = {
            "".join(chinese[index : index + 2])
            for index in range(max(0, len(chinese) - 1))
        }
        return set(tokens) | bigrams
