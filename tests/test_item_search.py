"""Verify ItemSearch fusion, scoped recall, reranking, and tool boundaries."""

import asyncio
import json
from collections.abc import Mapping, Sequence

import numpy as np
import pytest
from langchain_core.tools import BaseTool
from pydantic import ValidationError

from app.application.catalog.config import ItemSearchConfig
from app.application.catalog.fusion import fuse_request_vector
from app.application.catalog.models import (
    FloatVector,
    ItemSearchCommand,
    RecallHit,
    UserSignal,
)
from app.application.catalog.search_catalog import ItemSearchService
from app.application.runtime import AgentExecutionContext, ShoppingContextSnapshot
from app.domain.catalog import Money, Product, ProductSearchSpec, Sku
from app.infrastructure.context import reset_context, set_context
from app.infrastructure.langchain.tools.product_search import create_item_search_tool
from app.infrastructure.pricing import create_static_pricing_provider
from app.infrastructure.retrieval.item_search.faiss_index import (
    InMemoryIndexRegistry,
    validate_index_id,
)
from app.infrastructure.retrieval.item_search.lexical import keyword_2gram_score
from app.infrastructure.retrieval.item_search.user_tower import (
    InMemoryUserSignalProvider,
)


class FakeEncoder:
    dimension = 2

    def embed_queries(self, texts: Sequence[str]) -> FloatVector:
        vectors = {
            "旅行收纳袋": [1.0, 0.0],
            "偏好轻便和低价": [0.0, 1.0],
        }
        return np.asarray([vectors[text] for text in texts], dtype=np.float32)

    def embed_documents(self, texts: Sequence[str]) -> FloatVector:
        raise AssertionError("online search must not rebuild item embeddings")


class FakeReranker:
    def score(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
        scores = []
        for query, document in pairs:
            assert "当前查询：旅行收纳袋" in query
            assert "用户偏好：偏好轻便和低价" in query
            if "轻量收纳袋" in document:
                scores.append(0.95)
            elif "耐用收纳袋" in document:
                scores.append(0.80)
            else:
                scores.append(0.20)
        return scores


class FakeIndex:
    index_id = "amazon-cn"
    dimension = 2

    def __init__(
        self,
        products: Mapping[str, Product],
        vectors: Mapping[str, Sequence[float]],
    ) -> None:
        self._products = dict(products)
        self._vectors = {
            item_id: np.asarray(vector, dtype=np.float32)
            for item_id, vector in vectors.items()
        }
        self.last_vector: FloatVector | None = None
        self.last_top_k: int | None = None

    @property
    def size(self) -> int:
        return len(self._products)

    def search(self, vector: FloatVector, top_k: int) -> Sequence[RecallHit]:
        self.last_vector = vector
        self.last_top_k = top_k
        scores = [
            RecallHit(item_id=item_id, score=float(item_vector @ vector))
            for item_id, item_vector in self._vectors.items()
        ]
        return sorted(scores, key=lambda hit: hit.score, reverse=True)[:top_k]

    def get_product(self, item_id: str) -> Product:
        return self._products[item_id]

    def keyword_search(self, query: str, top_k: int) -> Sequence[RecallHit]:
        """使用生产词法评分模拟商品索引的三级召回。"""

        hits = [
            RecallHit(item_id, keyword_2gram_score(query, product.to_search_text()))
            for item_id, product in self._products.items()
        ]
        return tuple(
            sorted(
                (hit for hit in hits if hit.score > 0),
                key=lambda hit: hit.score,
                reverse=True,
            )[:top_k]
        )


class FailingEncoder(FakeEncoder):
    """模拟在线 embedding 服务故障。"""

    def embed_queries(self, texts: Sequence[str]) -> FloatVector:
        del texts
        raise RuntimeError("embedding unavailable")


class FailingReranker(FakeReranker):
    """模拟在线 reranker 服务故障。"""

    def score(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
        del pairs
        raise RuntimeError("reranker unavailable")


class ConstantEncoder:
    """为结构化过滤集成测试返回固定向量。"""

    dimension = 2

    def embed_queries(self, texts: Sequence[str]) -> FloatVector:
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    def embed_documents(self, texts: Sequence[str]) -> FloatVector:
        raise AssertionError("在线搜索不应重建商品向量")


class StableReranker:
    """按照候选原顺序给出稳定递减分数。"""

    def score(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
        return [1.0 - index * 0.1 for index, _ in enumerate(pairs)]


def _build_service(
    retrieval_mode: str = "embedding_rerank",
) -> tuple[ItemSearchService, FakeIndex]:
    def product(item_id: str, title: str) -> Product:
        """构造具备可售主 SKU 的最小商品测试夹具。"""

        sku = Sku(
            sku_id=f"{item_id}-sku",
            spec="默认款",
            price=Money.from_major_units(99, "CNY"),
            stock=10,
        )
        return Product(
            item_id=item_id,
            title=title,
            skus=(sku,),
            primary_sku_id=sku.sku_id,
        )

    products = {
        "durable": product("durable", "耐用收纳袋"),
        "light": product("light", "轻量收纳袋"),
        "other": product("other", "普通整理盒"),
    }
    index = FakeIndex(
        products,
        {
            "durable": [1.0, 0.0],
            "light": [0.0, 1.0],
            "other": [0.7, 0.7],
        },
    )
    signals = InMemoryUserSignalProvider(
        {
            ("user-1", "amazon-cn"): UserSignal(
                vector=np.asarray([0.0, 1.0], dtype=np.float32),
                summary="偏好轻便和低价",
            )
        }
    )
    config = ItemSearchConfig(
        embedding_dimension=2,
        recall_k=100,
        result_k=2,
        max_result_k=3,
        query_weight=0.8,
        retrieval_mode=retrieval_mode,
    )
    service = ItemSearchService(
        encoder=FakeEncoder(),
        reranker=FakeReranker(),
        indexes=InMemoryIndexRegistry({"amazon-cn": index}),
        user_signals=signals,
        config=config,
    )
    return service, index


def test_retrieval_ablation_can_run_lexical_without_embedding() -> None:
    """词法消融必须完全绕过 embedding 与 reranker。"""

    service, _ = _build_service("lexical")
    service._encoder = FailingEncoder()  # type: ignore[attr-defined]
    service._reranker = FailingReranker()  # type: ignore[attr-defined]

    response = service.search(
        ItemSearchCommand(
            spec=ProductSearchSpec(normalized_query="旅行收纳袋", top_k=2),
            index_id="amazon-cn",
        )
    )

    assert response.recall_strategy == "keyword_2gram"
    assert response.items


def test_retrieval_ablation_can_skip_reranker() -> None:
    """向量消融保留 FAISS 顺序并且不调用交叉编码器。"""

    service, _ = _build_service("embedding")
    service._reranker = FailingReranker()  # type: ignore[attr-defined]

    response = service.search(
        ItemSearchCommand(
            spec=ProductSearchSpec(normalized_query="旅行收纳袋", top_k=2),
            index_id="amazon-cn",
        )
    )

    assert response.recall_strategy == "embedding_only"


def test_fusion_falls_back_to_query_without_user_signal() -> None:
    query = np.asarray([3.0, 4.0], dtype=np.float32)

    fused = fuse_request_vector(query, None, query_weight=0.8)

    assert np.allclose(fused, np.asarray([0.6, 0.8], dtype=np.float32))


def test_service_recalls_selected_domain_then_reranks() -> None:
    service, index = _build_service()

    response = service.search(
        ItemSearchCommand(
            spec=ProductSearchSpec(
                normalized_query="旅行收纳袋",
                top_k=2,
            ),
            index_id="amazon-cn",
            buyer_id="user-1",
        )
    )

    assert response.index_id == "amazon-cn"
    assert response.recall_count == 3
    assert [item.product.item_id for item in response.items] == ["light", "durable"]
    assert index.last_top_k == 3
    assert index.last_vector is not None
    assert index.last_vector[0] > index.last_vector[1] > 0


def test_item_search_tool_exposes_only_business_search_arguments() -> None:
    service, _ = _build_service()
    item_search = create_item_search_tool(service, index_id="amazon-cn")

    token = set_context(
        AgentExecutionContext(
            thread_id="tool-test",
            shopping=ShoppingContextSnapshot("shopping-1", "user-1"),
        )
    )
    try:
        result = asyncio.run(
            item_search.ainvoke(
                {
                    "normalized_query": "旅行收纳袋",
                    "top_k": 2,
                }
            )
        )
    finally:
        reset_context(token)
    schema_properties = item_search.args_schema.model_json_schema()["properties"]
    payload = json.loads(result)

    assert isinstance(item_search, BaseTool)
    assert item_search.name == "item_search"
    assert set(schema_properties) == {
        "normalized_query",
        "category",
        "ship_to",
        "top_k",
        "price_max_major",
        "target_currency",
        "quantity",
        "price_basis",
        "excluded_brands",
        "required_brand",
    }
    assert payload["index_id"] == "amazon-cn"
    assert payload["items"][0]["product"]["item_id"] == "light"


def test_item_search_tool_validates_arguments_before_service_call() -> None:
    """工具边界拒绝越界数量和非法国家代码。"""

    service, _ = _build_service()
    item_search = create_item_search_tool(service, index_id="amazon-cn")

    with pytest.raises(ValidationError):
        asyncio.run(
            item_search.ainvoke(
                {
                    "normalized_query": "旅行收纳袋",
                    "ship_to": "CHINA",
                    "top_k": 0,
                }
            )
        )


def test_product_constraints_use_structured_domain_fields() -> None:
    """配送和预算过滤读取领域字段，而不是解析展示文本。"""

    product = Product(
        item_id="structured",
        title="结构化商品",
        category="旅行装备",
        ships_to=("cn", "US"),
        skus=(
            Sku(
                "sku-1",
                "标准版",
                Money.from_major_units(89, "cny"),
                stock=2,
            ),
        ),
    )

    assert product.ships_to == ("CN", "US")
    assert product.primary_sku().price.major == 89
    assert Product.from_dict(product.to_dict()) == product


def test_index_id_rejects_paths_and_filter_syntax() -> None:
    for invalid_index_id in ("../amazon", "amazon/category=bag", "", "含空格"):
        try:
            validate_index_id(invalid_index_id)
        except ValueError:
            continue
        raise AssertionError(f"invalid index_id was accepted: {invalid_index_id}")


def test_service_rejects_zero_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        ProductSearchSpec(
            normalized_query="旅行收纳袋",
            top_k=0,
        )


def test_item_search_falls_back_to_embedding_order_when_reranker_fails() -> None:
    """第二级不重新召回，直接保留 embedding 相似度顺序。"""

    service, _ = _build_service()
    service._reranker = FailingReranker()

    response = service.search(
        ItemSearchCommand(
            spec=ProductSearchSpec(
                normalized_query="旅行收纳袋",
                top_k=2,
            ),
            index_id="amazon-cn",
        )
    )

    assert response.recall_strategy == "embedding_only"
    assert response.items[0].product.item_id == "durable"


def test_item_search_falls_back_to_keyword_when_embedding_fails() -> None:
    """第三级完全不依赖 embedding 和 reranker。"""

    _, index = _build_service()
    service = ItemSearchService(
        encoder=FailingEncoder(),
        reranker=FailingReranker(),
        indexes=InMemoryIndexRegistry({"amazon-cn": index}),
        config=ItemSearchConfig(
            embedding_dimension=2,
            recall_k=3,
            result_k=2,
            max_result_k=3,
        ),
    )

    response = service.search(
        ItemSearchCommand(
            spec=ProductSearchSpec(
                normalized_query="轻量收纳袋",
                top_k=2,
            ),
            index_id="amazon-cn",
        )
    )

    assert response.recall_strategy == "keyword_2gram"
    assert response.items[0].product.item_id == "light"


def test_item_search_filters_cross_currency_and_inlines_landed_price() -> None:
    """搜索卡使用同一汇率做预算过滤，并在可配送命中中内联到手价。"""

    products = {
        "eligible": Product(
            item_id="eligible",
            title="可配送旅行包",
            category="旅行装备",
            ships_to=("CN",),
            skus=(
                Sku(
                    "eligible-sku",
                    "标准版",
                    Money.from_major_units(100, "USD"),
                    10,
                ),
            ),
            primary_sku_id="eligible-sku",
        ),
        "expensive": Product(
            item_id="expensive",
            title="高价旅行包",
            category="旅行装备",
            ships_to=("CN",),
            skus=(
                Sku(
                    "expensive-sku",
                    "标准版",
                    Money.from_major_units(200, "USD"),
                    10,
                ),
            ),
            primary_sku_id="expensive-sku",
        ),
        "unshippable": Product(
            item_id="unshippable",
            title="不可配送旅行包",
            category="旅行装备",
            ships_to=("US",),
            skus=(
                Sku(
                    "unshippable-sku",
                    "标准版",
                    Money.from_major_units(50, "USD"),
                    10,
                ),
            ),
            primary_sku_id="unshippable-sku",
        ),
        "no-stock": Product(
            item_id="no-stock",
            title="缺货旅行包",
            category="旅行装备",
            ships_to=("CN",),
            skus=(
                Sku(
                    "no-stock-sku",
                    "标准版",
                    Money.from_major_units(20, "USD"),
                    0,
                ),
            ),
            primary_sku_id="no-stock-sku",
        ),
    }
    index = FakeIndex(
        products,
        {
            "eligible": [1.0, 0.0],
            "expensive": [0.9, 0.0],
            "unshippable": [0.8, 0.0],
            "no-stock": [0.7, 0.0],
        },
    )
    service = ItemSearchService(
        encoder=ConstantEncoder(),
        reranker=StableReranker(),
        indexes=InMemoryIndexRegistry({"amazon-cn": index}),
        pricing=create_static_pricing_provider(),
        config=ItemSearchConfig(
            embedding_dimension=2,
            recall_k=4,
            result_k=4,
            max_result_k=4,
        ),
    )

    response = service.search(
        ItemSearchCommand(
            spec=ProductSearchSpec(
                normalized_query="旅行包",
                category="旅行装备",
                ship_to="cn",
                top_k=4,
                price_max_major=800,
                target_currency="cny",
            ),
            index_id="amazon-cn",
        )
    )
    payload = response.to_dict()

    assert [item.product.item_id for item in response.items] == ["eligible"]
    assert {item.reason for item in response.filtered_out} == {
        "over_price_cap",
        "ship_to_unavailable",
        "no_available_sku",
    }
    landed = payload["items"][0]["product"]["landed_price"]
    assert landed["subtotal_major"] == 710
    assert landed["freight_major"] == 25
    assert landed["landed_total_major"] == 735
    assert landed["tariff_rule_version"] == "static-2026-08"
    assert landed["exchange_rate_version"] == "static-2026-08"
