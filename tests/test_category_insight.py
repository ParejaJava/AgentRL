"""验证品类洞察聚合、工具边界和增量摄取。"""

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

from langchain_core.tools import BaseTool

from app.application.catalog import (
    CategoryInsightConfig,
    CategoryInsightRequest,
    GetCategoryInsight,
)
from app.application.catalog.category_insight_models import RetrievedCategoryCard
from app.domain.catalog import (
    AttributeDistribution,
    Bestseller,
    CategoryCard,
    CategoryCardType,
    PriceTier,
    PriceTierName,
)
from app.infrastructure.langchain.tools.category_insight import (
    create_category_insight_tool,
)
from app.infrastructure.retrieval.category_insight import (
    CategoryKnowledgeIngestor,
    JsonlCategoryCardStore,
    LocalCategoryCardRetriever,
)

TEST_ROOT = Path("data/test-output/category-insight-tests").resolve()


def _fresh_file(relative_path: str) -> Path:
    """在仓库内可写测试目录创建一个不存在的文件路径。"""

    path = TEST_ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    return path


def _cards(
    *,
    source_hash: str = "hash-1",
    source_document: str = "travel.md",
) -> tuple[CategoryCard, ...]:
    """创建覆盖三类知识的旅行装备测试卡片。"""

    common = {
        "category": "旅行装备",
        "raw_evidence": ("旅行三件套是长途飞行刚需组合。",),
        "last_updated": "2026-08-26T00:00:00+00:00",
        "confidence": 0.9,
        "source_document": source_document,
        "source_hash": source_hash,
        "pitfalls": ("20 寸并非所有航司都免托运。",),
    }
    return (
        CategoryCard(
            card_id="card-bestseller",
            card_type=CategoryCardType.BESTSELLER,
            summary="旅行三件套适合长途飞行",
            bestsellers=(
                Bestseller(
                    name="旅行三件套",
                    components=("收纳袋", "颈枕", "眼罩"),
                    use_cases=("长途飞行",),
                    selling_points=("轻量", "好收纳"),
                    confidence=0.92,
                ),
            ),
            **common,
        ),
        CategoryCard(
            card_id="card-attribute",
            card_type=CategoryCardType.ATTRIBUTE,
            summary="材质和自重是关键属性",
            attributes=(
                AttributeDistribution(
                    attribute="材质",
                    mainstream_values=("帆布", "再生尼龙"),
                    selection_guide="材质敏感买家优先非塑料材质。",
                    caveats=("低价皮质可能是 PU 涂层。",),
                    confidence=0.88,
                ),
            ),
            **common,
        ),
        CategoryCard(
            card_id="card-price",
            card_type=CategoryCardType.PRICE_RANGE,
            summary="旅行三件套主力价格为 180-260 元",
            price_tiers=(
                PriceTier(
                    tier=PriceTierName.MAINSTREAM,
                    min_price=180,
                    max_price=260,
                    currency="CNY",
                    representative_products=("旅行三件套",),
                    description="竞争最激烈的主力档。",
                    confidence=0.9,
                ),
            ),
            **common,
        ),
    )


class FakeRetriever:
    """返回固定卡片的 Application 端口替身。"""

    def search(self, category: str, limit: int) -> Sequence[RetrievedCategoryCard]:
        assert category == "旅行三件套"
        return tuple(
            RetrievedCategoryCard(card=card, score=0.95)
            for card in _cards()[:limit]
        )


def test_quick_and_deep_have_different_detail_levels() -> None:
    """quick 不返回属性图谱，deep 返回完整属性选择口径。"""

    service = GetCategoryInsight(
        FakeRetriever(),
        CategoryInsightConfig(quick_recall_k=3, deep_recall_k=5),
    )

    quick = service.execute(CategoryInsightRequest("旅行三件套", "quick"))
    deep = service.execute(CategoryInsightRequest("旅行三件套", "deep"))

    assert quick.components == ("收纳袋", "颈枕", "眼罩")
    assert quick.attributes == ()
    assert deep.attributes[0].attribute == "材质"
    assert deep.price_tiers[0].min_price == 180
    assert deep.confidence == 0.9


def test_tool_uses_decorator_and_never_returns_raw_evidence() -> None:
    """Agent 只能看到压缩结果，不能看到知识库原文。"""

    tool = create_category_insight_tool(GetCategoryInsight(FakeRetriever()))
    result = asyncio.run(
        tool.ainvoke({"category": "旅行三件套", "depth": "deep"})
    )
    payload = json.loads(result)

    assert isinstance(tool, BaseTool)
    assert tool.name == "category_insight"
    assert set(tool.args_schema.model_json_schema()["properties"]) == {
        "category",
        "depth",
    }
    assert payload["status"] == "ok"
    assert payload["attributes"][0]["attribute"] == "材质"
    assert "raw_evidence" not in result


def test_jsonl_store_round_trip_and_local_retrieval() -> None:
    """结构化卡片持久化后可被本地适配器重复查询。"""

    store = JsonlCategoryCardStore(_fresh_file("round-trip/cards.jsonl"))
    store.save(_cards())
    retriever = LocalCategoryCardRetriever(store)

    restored = store.load()
    results = retriever.search("旅行三件套", 8)

    assert {card.card_id: card for card in restored} == {
        card.card_id: card for card in _cards()
    }
    assert {item.card.card_id for item in results} == {
        "card-bestseller",
        "card-attribute",
        "card-price",
    }


class FakeExtractor:
    """记录调用次数，证明内容哈希能阻止重复结构化。"""

    def __init__(self) -> None:
        self.calls = 0

    async def extract(
        self,
        *,
        source_document: str,
        source_hash: str,
        last_updated: str,
        markdown: str,
    ) -> Sequence[CategoryCard]:
        del last_updated
        self.calls += 1
        assert markdown.startswith("# 旅行装备")
        return _cards(source_hash=source_hash, source_document=source_document)


class FakePublisher:
    """记录每次发布的完整卡片快照。"""

    def __init__(self) -> None:
        self.snapshots: list[tuple[CategoryCard, ...]] = []

    def sync(self, cards: Sequence[CategoryCard]) -> None:
        self.snapshots.append(tuple(cards))


def test_ingestion_calls_structurer_only_when_document_changes() -> None:
    """同一 Markdown 未变化时，后续摄取不会再次调用 LLM。"""

    base = TEST_ROOT / "ingestion"
    knowledge_root = base / "knowledge"
    knowledge_root.mkdir(parents=True, exist_ok=True)
    source = knowledge_root / "travel.md"
    source.write_text("# 旅行装备\n旅行三件套是长途飞行刚需组合。", encoding="utf-8")
    extractor = FakeExtractor()
    publisher = FakePublisher()
    store = JsonlCategoryCardStore(_fresh_file("ingestion/data/cards.jsonl"))
    ingestor = CategoryKnowledgeIngestor(
        extractor=extractor,
        store=store,
        manifest_path=_fresh_file("ingestion/data/manifest.json"),
        publisher=publisher,
    )

    first = asyncio.run(ingestor.ingest(knowledge_root))
    second = asyncio.run(ingestor.ingest(knowledge_root))
    source.write_text(
        "# 旅行装备\n旅行三件套是长途飞行刚需组合。\n新增材质说明。",
        encoding="utf-8",
    )
    third = asyncio.run(ingestor.ingest(knowledge_root))

    assert first.processed_documents == 1
    assert second.skipped_documents == 1
    assert third.processed_documents == 1
    assert extractor.calls == 2
    assert second.published_cards == 3
    assert len(publisher.snapshots) == 3
