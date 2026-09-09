"""验证 OpenSearch 品类知识适配器的索引、Pipeline、同步与重排。"""

import json
from collections.abc import Sequence
from typing import Any

from app.application.catalog.category_insight_models import RetrievedCategoryCard
from app.domain.catalog import Bestseller, CategoryCard, CategoryCardType
from app.infrastructure.retrieval.category_insight.opensearch import (
    OpenSearchCategoryCardRepository,
)
from app.infrastructure.retrieval.category_insight.schemas import (
    category_card_to_dict,
)


def _card(card_id: str, summary: str, *, general: bool = False) -> CategoryCard:
    """创建可通过领域校验的测试知识卡。"""

    return CategoryCard(
        card_id=card_id,
        category="通用品类" if general else "旅行装备",
        card_type=CategoryCardType.BESTSELLER,
        summary=summary,
        raw_evidence=(summary,),
        last_updated="2026-08-27T00:00:00+00:00",
        confidence=0.9,
        source_document="travel.md",
        source_hash="hash",
        applies_to_all_categories=general,
        bestsellers=(Bestseller(name="旅行三件套", confidence=0.9),),
    )


class FakeEncoder:
    """用固定二维向量替代 BGE 模型。"""

    dimension = 2

    def embed_queries(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[0.5, 0.5] for _ in texts]


class FakeReranker:
    """让包含“最佳”的卡片排在混合召回首位。"""

    def score(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
        return [0.95 if "最佳" in document else 0.2 for _, document in pairs]


class FailingEncoder(FakeEncoder):
    """模拟查询 embedding 服务不可用。"""

    def embed_queries(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        del texts
        raise RuntimeError("embedding unavailable")


class FailingReranker(FakeReranker):
    """模拟交叉编码器重排服务不可用。"""

    def score(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
        del pairs
        raise RuntimeError("reranker unavailable")


class FakeKeywordFallback:
    """模拟无需模型的本地 JSONL 二元组检索。"""

    def search(self, category: str, limit: int) -> Sequence[RetrievedCategoryCard]:
        assert category == "旅行装备"
        return (
            RetrievedCategoryCard(
                card=_card("keyword", "本地关键词命中"),
                score=0.8,
                recall_strategy="keyword_2gram",
            ),
        )[:limit]


class FakeIndices:
    """记录索引创建请求。"""

    def __init__(self) -> None:
        self.created: tuple[str, dict[str, Any]] | None = None

    def exists(self, *, index: str) -> bool:
        del index
        return False

    def create(self, *, index: str, body: dict[str, Any]) -> None:
        self.created = (index, body)


class FakeTransport:
    """记录 Search Pipeline 的 PUT 请求。"""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, dict[str, Any]]] = []

    def perform_request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any],
    ) -> dict[str, bool]:
        self.requests.append((method, path, body))
        return {"acknowledged": True}


class FakeOpenSearch:
    """模拟适配器使用到的官方客户端最小表面。"""

    def __init__(self, hybrid_cards: Sequence[CategoryCard] = ()) -> None:
        self.indices = FakeIndices()
        self.transport = FakeTransport()
        self.hybrid_cards = tuple(hybrid_cards)
        self.search_calls: list[dict[str, Any]] = []
        self.bulk_body = ""

    def search(self, **kwargs: Any) -> dict[str, Any]:
        self.search_calls.append(kwargs)
        query = kwargs["body"]["query"]
        if "match_all" in query:
            return {
                "hits": {
                    "hits": [
                        {
                            "_id": "stale-card",
                            "_source": {
                                "source_hash": "old-hash",
                                "embedding_model": "old-model",
                            },
                        }
                    ]
                }
            }
        if "term" in query:
            cards = [card for card in self.hybrid_cards if card.applies_to_all_categories]
        else:
            cards = list(self.hybrid_cards)
        return {
            "hits": {
                "hits": [
                    {"_id": card.card_id, "_source": {"card_payload": category_card_to_dict(card)}}
                    for card in cards
                ]
            }
        }

    def bulk(self, *, body: str, refresh: bool) -> dict[str, Any]:
        assert refresh is True
        self.bulk_body = body
        return {"errors": False, "items": []}


def _repository(
    client: FakeOpenSearch,
    *,
    encoder: FakeEncoder | None = None,
    reranker: FakeReranker | None = None,
    keyword_fallback: FakeKeywordFallback | None = None,
    min_relevance_score: float = 0.1,
) -> OpenSearchCategoryCardRepository:
    """使用测试替身装配被测适配器。"""

    return OpenSearchCategoryCardRepository(
        client,
        encoder or FakeEncoder(),
        reranker or FakeReranker(),
        index_name="category-index",
        pipeline_name="category-rrf",
        embedding_dimension=2,
        hybrid_recall_k=20,
        min_relevance_score=min_relevance_score,
        bm25_weight=0.4,
        knn_weight=0.6,
        keyword_fallback=keyword_fallback,
    )


def test_ensure_resources_creates_knn_index_and_weighted_rrf_pipeline() -> None:
    """索引使用 KNN，Pipeline 使用两路加权 RRF。"""

    client = FakeOpenSearch()
    repository = _repository(client)

    repository.ensure_resources()

    assert client.indices.created is not None
    _, definition = client.indices.created
    embedding = definition["mappings"]["properties"]["embedding"]
    assert embedding["type"] == "knn_vector"
    assert embedding["dimension"] == 2
    _, path, pipeline = client.transport.requests[0]
    assert path == "/_search/pipeline/category-rrf"
    combination = pipeline["phase_results_processors"][0][
        "score-ranker-processor"
    ]["combination"]
    assert combination["technique"] == "rrf"
    assert combination["parameters"]["weights"] == [0.4, 0.6]


def test_sync_embeds_cards_and_removes_stale_documents() -> None:
    """同步会写入结构化卡、BGE 向量，并删除不在快照中的旧 ID。"""

    client = FakeOpenSearch()
    repository = _repository(client)

    repository.sync((_card("current-card", "当前知识"),))

    lines = [json.loads(line) for line in client.bulk_body.splitlines()]
    assert lines[0]["index"]["_id"] == "current-card"
    assert lines[1]["embedding"] == [0.5, 0.5]
    assert lines[1]["card_payload"]["summary"] == "当前知识"
    assert lines[2]["delete"]["_id"] == "stale-card"


def test_search_uses_pipeline_then_reranks_and_keeps_global_rule() -> None:
    """在线查询先走 Hybrid Pipeline，再由 BGE 重排并保留一条全局规则。"""

    client = FakeOpenSearch(
        (
            _card("ordinary", "普通候选"),
            _card("best", "最佳候选"),
            _card("global", "通用避坑", general=True),
        )
    )
    repository = _repository(client)

    results = repository.search("旅行装备", 3)

    hybrid_call = client.search_calls[0]
    assert hybrid_call["params"] == {"search_pipeline": "category-rrf"}
    assert "hybrid" in hybrid_call["body"]["query"]
    assert [item.card.card_id for item in results] == ["best", "ordinary", "global"]
    assert results[0].score == 0.95
    assert results[0].recall_strategy == "embedding_rerank"


def test_search_keeps_hybrid_order_when_reranker_fails() -> None:
    """第二级保留 OpenSearch Hybrid 返回顺序。"""

    client = FakeOpenSearch(
        (
            _card("ordinary", "普通候选"),
            _card("best", "最佳候选"),
        )
    )
    repository = _repository(client, reranker=FailingReranker())

    results = repository.search("旅行装备", 2)

    assert [item.card.card_id for item in results] == ["ordinary", "best"]
    assert {item.recall_strategy for item in results} == {"embedding_only"}


def test_search_uses_local_keyword_fallback_when_embedding_fails() -> None:
    """第三级从本地知识卡快照检索，不依赖 OpenSearch 查询。"""

    client = FakeOpenSearch()
    repository = _repository(
        client,
        encoder=FailingEncoder(),
        keyword_fallback=FakeKeywordFallback(),
    )

    results = repository.search("旅行装备", 3)

    assert [item.card.card_id for item in results] == ["keyword"]
    assert results[0].recall_strategy == "keyword_2gram"
    assert client.search_calls == []


def test_search_rejects_when_all_reranker_scores_are_below_threshold() -> None:
    """域外查询不能因为全局规则存在而始终返回知识卡。"""

    client = FakeOpenSearch(
        (
            _card("ordinary", "普通候选"),
            _card("global", "通用避坑", general=True),
        )
    )
    repository = _repository(client, min_relevance_score=0.5)

    assert repository.search("完全无关的问题", 3) == ()


def test_relevant_global_rule_can_pass_the_same_relevance_gate() -> None:
    """全局规则与普通卡使用同一阈值，可单独回答到手价类问题。"""

    client = FakeOpenSearch((_card("global", "最佳到手价规则", general=True),))
    repository = _repository(client, min_relevance_score=0.5)

    results = repository.search("到手价如何计算", 3)

    assert [item.card.card_id for item in results] == ["global"]
    assert results[0].score == 0.95


def test_ablation_queries_do_not_trigger_hidden_fallbacks() -> None:
    """BM25、KNN 和 Hybrid 消融分别发出可辨识的 OpenSearch 查询。"""

    client = FakeOpenSearch((_card("best", "最佳候选"),))
    repository = _repository(client)

    assert repository.search_with_strategy("旅行装备", 2, "bm25")
    assert "multi_match" in client.search_calls[-2]["body"]["query"]
    assert repository.search_with_strategy("旅行装备", 2, "knn")
    assert "knn" in client.search_calls[-2]["body"]["query"]
    assert repository.search_with_strategy("旅行装备", 2, "hybrid_rrf")
    assert "hybrid" in client.search_calls[-2]["body"]["query"]
    assert client.search_calls[-2]["params"] == {
        "search_pipeline": "category-rrf"
    }
