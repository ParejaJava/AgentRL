"""Verify ItemSearch fusion, scoped recall, reranking, and tool boundaries."""

import asyncio
import json
from collections.abc import Mapping, Sequence

import numpy as np
from langchain_core.tools import BaseTool

from app.tools.item_search.config import ItemSearchConfig
from app.tools.item_search.faiss_index import InMemoryIndexRegistry, validate_index_id
from app.tools.item_search.fusion import fuse_request_vector
from app.tools.item_search.schemas import (
    FloatVector,
    ItemSearchRequest,
    Product,
    RecallHit,
    UserSignal,
)
from app.tools.item_search.service import ItemSearchService
from app.tools.item_search.tool import create_item_search_tool
from app.tools.item_search.user_tower import InMemoryUserSignalProvider


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


def _build_service() -> tuple[ItemSearchService, FakeIndex]:
    products = {
        "durable": Product(item_id="durable", title="耐用收纳袋"),
        "light": Product(item_id="light", title="轻量收纳袋"),
        "other": Product(item_id="other", title="普通整理盒"),
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
    )
    service = ItemSearchService(
        encoder=FakeEncoder(),
        reranker=FakeReranker(),
        indexes=InMemoryIndexRegistry({"amazon-cn": index}),
        user_signals=signals,
        config=config,
    )
    return service, index


def test_fusion_falls_back_to_query_without_user_signal() -> None:
    query = np.asarray([3.0, 4.0], dtype=np.float32)

    fused = fuse_request_vector(query, None, query_weight=0.8)

    assert np.allclose(fused, np.asarray([0.6, 0.8], dtype=np.float32))


def test_service_recalls_selected_domain_then_reranks() -> None:
    service, index = _build_service()

    response = service.search(
        ItemSearchRequest(
            query="旅行收纳袋",
            index_id="amazon-cn",
            user_id="user-1",
            top_k=2,
        )
    )

    assert response.index_id == "amazon-cn"
    assert response.recall_count == 3
    assert [item.product.item_id for item in response.items] == ["light", "durable"]
    assert index.last_top_k == 3
    assert index.last_vector is not None
    assert index.last_vector[0] > index.last_vector[1] > 0


def test_item_search_tool_has_no_metadata_filter_arguments() -> None:
    service, _ = _build_service()
    item_search = create_item_search_tool(service)

    result = asyncio.run(
        item_search.ainvoke(
            {
                "query": "旅行收纳袋",
                "index_id": "amazon-cn",
                "user_id": "user-1",
                "top_k": 2,
            }
        )
    )
    schema_properties = item_search.args_schema.model_json_schema()["properties"]
    payload = json.loads(result)

    assert isinstance(item_search, BaseTool)
    assert item_search.name == "item_search"
    assert set(schema_properties) == {"query", "index_id", "user_id", "top_k"}
    assert payload["index_id"] == "amazon-cn"
    assert payload["items"][0]["product"]["item_id"] == "light"


def test_index_id_rejects_paths_and_filter_syntax() -> None:
    for invalid_index_id in ("../amazon", "amazon/category=bag", "", "含空格"):
        try:
            validate_index_id(invalid_index_id)
        except ValueError:
            continue
        raise AssertionError(f"invalid index_id was accepted: {invalid_index_id}")


def test_service_rejects_zero_top_k() -> None:
    service, _ = _build_service()

    try:
        service.search(
            ItemSearchRequest(
                query="旅行收纳袋",
                index_id="amazon-cn",
                user_id="user-1",
                top_k=0,
            )
        )
    except ValueError as exc:
        assert "top_k" in str(exc)
    else:
        raise AssertionError("top_k=0 should be rejected")
