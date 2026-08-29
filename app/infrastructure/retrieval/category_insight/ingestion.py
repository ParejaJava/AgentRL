"""把 Markdown 品类知识一次性结构化为可长期复用的知识卡片。"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.domain.catalog import CategoryCard

from .local_store import JsonlCategoryCardStore
from .schemas import StructuredCategoryDocument, extracted_card_to_domain


class CategoryDocumentExtractor(Protocol):
    """把一份变更后的知识文档提炼成结构化卡片。"""

    async def extract(
        self,
        *,
        source_document: str,
        source_hash: str,
        last_updated: str,
        markdown: str,
    ) -> Sequence[CategoryCard]: ...


class CategoryCardPublisher(Protocol):
    """把已经结构化的完整知识快照发布到在线检索设施。"""

    def sync(self, cards: Sequence[CategoryCard]) -> None: ...


class StructuredLLMCategoryExtractor:
    """仅在离线摄取阶段调用结构化 LLM 的适配器。"""

    def __init__(self, model: BaseChatModel) -> None:
        self._model = model.with_structured_output(StructuredCategoryDocument)

    async def extract(
        self,
        *,
        source_document: str,
        source_hash: str,
        last_updated: str,
        markdown: str,
    ) -> Sequence[CategoryCard]:
        """根据原文生成三类卡片，并验证引用证据确实来自原文。"""

        system = SystemMessage(
            content=(
                "你是电商品类知识摄取器。只允许根据输入 Markdown 提取事实，"
                "不得补充外部知识或实时信息。为每个明确品类生成 bestseller、"
                "attribute、price_range 三类卡片；缺少可靠内容时不要生成对应卡片。"
                "category 使用简洁、标准化的中文品类名。跨所有品类生效的规则设置"
                " applies_to_all_categories=true。raw_evidence 必须包含 1-3 段原文原句。"
                "价格统一拆成 entry、mainstream、premium；原文没有上下界时填 null。"
                "confidence 根据证据明确程度给出 0-1 数值。pitfalls 只写原文明示风险。"
            )
        )
        human = HumanMessage(
            content=json.dumps(
                {
                    "source_document": source_document,
                    "last_updated": last_updated,
                    "markdown": markdown,
                },
                ensure_ascii=False,
            )
        )
        result = await self._model.ainvoke([system, human])
        document = StructuredCategoryDocument.model_validate(result)
        normalized_source = self._normalize(markdown)
        cards: list[CategoryCard] = []
        for extracted in document.cards:
            for evidence in extracted.raw_evidence:
                if self._normalize(evidence) not in normalized_source:
                    raise ValueError(
                        f"结构化结果包含不属于 {source_document} 的 raw_evidence"
                    )
            cards.append(
                extracted_card_to_domain(
                    extracted,
                    source_document=source_document,
                    source_hash=source_hash,
                    last_updated=last_updated,
                )
            )
        return tuple(cards)

    @staticmethod
    def _normalize(text: str) -> str:
        """忽略 Markdown 换行差异，但不改变证据中的实际字符。"""

        return " ".join(text.split())


@dataclass(frozen=True, slots=True)
class CategoryIngestionReport:
    """一次增量摄取的可审计结果。"""

    processed_documents: int
    skipped_documents: int
    removed_documents: int
    card_count: int
    published_cards: int = 0


class CategoryKnowledgeIngestor:
    """使用内容哈希保证未变化文档永远不会重复调用 LLM。"""

    def __init__(
        self,
        extractor: CategoryDocumentExtractor,
        store: JsonlCategoryCardStore,
        manifest_path: Path,
        publisher: CategoryCardPublisher | None = None,
    ) -> None:
        self._extractor = extractor
        self._store = store
        self._manifest_path = manifest_path
        self._publisher = publisher

    async def ingest(self, knowledge_root: Path) -> CategoryIngestionReport:
        """增量处理 Markdown，并在全部成功后原子更新卡片和 manifest。"""

        root = knowledge_root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"知识目录不存在：{root}")
        documents = sorted(
            path
            for path in root.rglob("*.md")
            if path.name.casefold() != "readme.md"
        )
        if not documents:
            raise FileNotFoundError(f"知识目录没有可摄取的 Markdown：{root}")

        manifest = self._load_manifest()
        previous_sources = dict(manifest.get("sources", {}))
        cards_by_id = {card.card_id: card for card in self._store.load()}
        current_sources: dict[str, dict[str, Any]] = {}
        processed = 0
        skipped = 0

        for path in documents:
            source_document = path.relative_to(root).as_posix()
            markdown = path.read_text(encoding="utf-8")
            source_hash = hashlib.sha256(markdown.encode()).hexdigest()
            previous = previous_sources.get(source_document, {})
            previous_card_ids = [str(item) for item in previous.get("card_ids", [])]
            unchanged = (
                previous.get("sha256") == source_hash
                and previous_card_ids
                and all(card_id in cards_by_id for card_id in previous_card_ids)
            )
            if unchanged:
                current_sources[source_document] = previous
                skipped += 1
                continue

            last_updated = datetime.fromtimestamp(
                path.stat().st_mtime,
                tz=UTC,
            ).isoformat()
            extracted = tuple(
                await self._extractor.extract(
                    source_document=source_document,
                    source_hash=source_hash,
                    last_updated=last_updated,
                    markdown=markdown,
                )
            )
            if not extracted:
                raise ValueError(f"结构化模型没有为 {source_document} 生成任何卡片")
            for card_id in previous_card_ids:
                cards_by_id.pop(card_id, None)
            for card in extracted:
                cards_by_id[card.card_id] = card
            current_sources[source_document] = {
                "sha256": source_hash,
                "card_ids": [card.card_id for card in extracted],
                "last_updated": last_updated,
            }
            processed += 1

        removed_sources = set(previous_sources) - set(current_sources)
        for source_document in removed_sources:
            previous = previous_sources[source_document]
            for card_id in previous.get("card_ids", []):
                cards_by_id.pop(str(card_id), None)

        self._store.save(tuple(cards_by_id.values()))
        self._save_manifest({"version": 1, "sources": current_sources})
        # OpenSearch/BGE 为同步阻塞客户端，放在线程中避免阻塞异步摄取入口。
        published_cards = 0
        if self._publisher is not None:
            snapshot = tuple(cards_by_id.values())
            await asyncio.to_thread(self._publisher.sync, snapshot)
            published_cards = len(snapshot)
        return CategoryIngestionReport(
            processed_documents=processed,
            skipped_documents=skipped,
            removed_documents=len(removed_sources),
            card_count=len(cards_by_id),
            published_cards=published_cards,
        )

    def _load_manifest(self) -> dict[str, Any]:
        """读取摄取状态；首次运行时返回空 manifest。"""

        if not self._manifest_path.exists():
            return {"version": 1, "sources": {}}
        try:
            raw = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"摄取 manifest 不是合法 JSON：{self._manifest_path}") from exc
        if raw.get("version") != 1 or not isinstance(raw.get("sources"), dict):
            raise ValueError(f"不支持的摄取 manifest：{self._manifest_path}")
        return raw

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        """原子更新内容哈希与 card_id 映射。"""

        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._manifest_path.with_suffix(
            f"{self._manifest_path.suffix}.tmp"
        )
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self._manifest_path)
