"""OpenSearch 品类知识库：混合召回、Search Pipeline 融合与 BGE 重排。"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from threading import Lock
from typing import Any, Literal, TypeAlias
from urllib.parse import urlparse

from app.application.catalog.category_insight_models import RetrievedCategoryCard
from app.application.catalog.category_insight_ports import CategoryKnowledgeRetriever
from app.application.catalog.ports import EmbeddingEncoder, Reranker
from app.domain.catalog import CategoryCard

from .schemas import category_card_from_dict, category_card_to_dict

logger = logging.getLogger(__name__)

CategoryRetrievalStrategy: TypeAlias = Literal[
    "keyword",
    "bm25",
    "knn",
    "hybrid_rrf",
    "hybrid_rerank",
]


def create_opensearch_client(
    url: str,
    *,
    username: str | None = None,
    password: str | None = None,
    verify_certs: bool = False,
    timeout_seconds: float = 10.0,
) -> Any:
    """按环境配置创建官方 Python 客户端，并把可选依赖延迟到真正使用时。"""

    try:
        from opensearchpy import OpenSearch
    except ImportError as exc:
        raise RuntimeError(
            "OpenSearch 支持尚未安装；请运行 `uv sync --extra search --extra rag`"
        ) from exc

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("OPENSEARCH_URL 必须是合法的 http/https URL")
    authentication = (
        (username, password or "") if username is not None else None
    )
    return OpenSearch(
        hosts=[
            {
                "host": parsed.hostname,
                "port": parsed.port or (443 if parsed.scheme == "https" else 80),
            }
        ],
        http_auth=authentication,
        use_ssl=parsed.scheme == "https",
        verify_certs=verify_certs,
        ssl_assert_hostname=verify_certs,
        ssl_show_warn=verify_certs,
        http_compress=True,
        timeout=timeout_seconds,
    )


def category_card_search_text(card: CategoryCard) -> str:
    """把结构化卡片展开成供向量编码和交叉编码器重排的检索文本。"""

    parts = [card.category, card.summary, *card.raw_evidence, *card.pitfalls]
    for item in card.bestsellers:
        parts.extend(
            (
                item.name,
                *item.components,
                *item.use_cases,
                *item.selling_points,
                *item.typical_attributes.keys(),
                *item.typical_attributes.values(),
            )
        )
    for item in card.attributes:
        parts.extend(
            (
                item.attribute,
                *item.mainstream_values,
                item.selection_guide,
                *item.caveats,
            )
        )
    for item in card.price_tiers:
        parts.extend(
            (
                item.tier.value,
                item.currency,
                *item.representative_products,
                item.description,
            )
        )
    return "\n".join(part.strip() for part in parts if part.strip())


class OpenSearchCategoryCardRepository:
    """同时实现知识卡发布端口和在线检索端口的 OpenSearch 适配器。"""

    def __init__(
        self,
        client: Any,
        encoder: EmbeddingEncoder,
        reranker: Reranker,
        *,
        index_name: str,
        pipeline_name: str,
        number_of_replicas: int = 0,
        embedding_dimension: int,
        embedding_model: str | None = None,
        hybrid_recall_k: int = 50,
        min_relevance_score: float = 0.1,
        bm25_weight: float = 0.4,
        knn_weight: float = 0.6,
        keyword_fallback: CategoryKnowledgeRetriever | None = None,
    ) -> None:
        if encoder.dimension != embedding_dimension:
            raise ValueError("编码器维度与 OpenSearch 索引维度不一致")
        if hybrid_recall_k < 1:
            raise ValueError("hybrid_recall_k 必须大于 0")
        if number_of_replicas < 0:
            raise ValueError("number_of_replicas 不能小于 0")
        if not 0.0 <= min_relevance_score <= 1.0:
            raise ValueError("min_relevance_score 必须位于 0 到 1 之间")
        if abs(bm25_weight + knn_weight - 1.0) > 1e-9:
            raise ValueError("BM25 与 KNN 权重之和必须为 1")
        self._client = client
        self._encoder = encoder
        self._reranker = reranker
        self._index_name = index_name
        self._pipeline_name = pipeline_name
        self._number_of_replicas = number_of_replicas
        self._embedding_dimension = embedding_dimension
        self._embedding_model = embedding_model or str(
            getattr(encoder, "model_name", type(encoder).__name__)
        )
        self._hybrid_recall_k = hybrid_recall_k
        self._min_relevance_score = min_relevance_score
        self._bm25_weight = bm25_weight
        self._knn_weight = knn_weight
        self._keyword_fallback = keyword_fallback
        self._ready = False
        self._ready_lock = Lock()

    def ensure_resources(self) -> None:
        """幂等创建向量索引和基于加权 RRF 的 Search Pipeline。"""

        if self._ready:
            return
        with self._ready_lock:
            if self._ready:
                return
            if not self._client.indices.exists(index=self._index_name):
                self._client.indices.create(
                    index=self._index_name,
                    body=self._index_definition(),
                )
            else:
                # 本地单节点默认零副本；生产环境可通过配置提高副本数。
                self._client.indices.put_settings(
                    index=self._index_name,
                    body={"index": {"number_of_replicas": self._number_of_replicas}},
                )
            # Pipeline 是服务器端融合环节；PUT 可重复执行并更新权重。
            self._client.transport.perform_request(
                "PUT",
                f"/_search/pipeline/{self._pipeline_name}",
                body=self._pipeline_definition(),
            )
            self._ready = True

    def sync(self, cards: Sequence[CategoryCard]) -> None:
        """把本地完整知识快照同步到 OpenSearch，并删除已失效卡片。"""

        self.ensure_resources()
        existing = self._existing_documents()
        changed_cards = [
            card
            for card in cards
            if existing.get(card.card_id)
            != (card.source_hash, self._embedding_model)
        ]
        search_texts = [category_card_search_text(card) for card in changed_cards]
        vectors = self._encoder.embed_documents(search_texts)
        if len(vectors) != len(changed_cards):
            raise ValueError("BGE 返回的文档向量数量与知识卡数量不一致")

        operations: list[str] = []
        for card, search_text, vector in zip(
            changed_cards,
            search_texts,
            vectors,
            strict=True,
        ):
            numeric_vector = [float(value) for value in vector]
            if len(numeric_vector) != self._embedding_dimension:
                raise ValueError("BGE 返回的文档向量维度与索引配置不一致")
            operations.append(
                json.dumps(
                    {"index": {"_index": self._index_name, "_id": card.card_id}}
                )
            )
            operations.append(
                json.dumps(
                    self._document(card, search_text, numeric_vector),
                    ensure_ascii=False,
                )
            )

        for stale_id in sorted(set(existing) - {card.card_id for card in cards}):
            operations.append(
                json.dumps(
                    {"delete": {"_index": self._index_name, "_id": stale_id}}
                )
            )
        if not operations:
            return
        response = self._client.bulk(
            body="\n".join(operations) + "\n",
            refresh=True,
        )
        if response.get("errors"):
            failures = [
                item
                for item in response.get("items", [])
                if next(iter(item.values())).get("error")
            ]
            raise RuntimeError(f"OpenSearch 批量同步失败：{failures[:3]}")

    def search(
        self,
        category: str,
        limit: int,
    ) -> Sequence[RetrievedCategoryCard]:
        """按重排、向量原序、关键词二元组三层策略检索知识卡。"""

        if limit < 1:
            return ()
        try:
            self.ensure_resources()
            query_vectors = self._encoder.embed_queries([category])
            if len(query_vectors) != 1:
                raise RuntimeError("BGE 没有为品类查询返回唯一向量")
            query_vector = [float(value) for value in query_vectors[0]]
            if len(query_vector) != self._embedding_dimension:
                raise RuntimeError("BGE 返回的查询向量维度与索引配置不一致")

            response = self._client.search(
                index=self._index_name,
                params={"search_pipeline": self._pipeline_name},
                body=self._hybrid_query(category, query_vector),
            )
            hybrid_cards = self._cards_from_hits(response)
        except Exception:
            logger.warning(
                "品类 embedding/hybrid 召回失败，降级到 keyword_2gram",
                exc_info=True,
            )
            return self._keyword_search(category, limit)
        if not hybrid_cards:
            return self._keyword_search(category, limit)

        general_cards = [
            card for card in hybrid_cards if card.applies_to_all_categories
        ]
        try:
            general_cards.extend(self._general_cards())
        except Exception:
            # 全局规则补查失败不应丢弃已经成功召回的具体品类卡。
            logger.warning("品类全局规则补查失败，继续使用现有候选", exc_info=True)
        candidates = list(
            {
                card.card_id: card
                for card in (*hybrid_cards, *general_cards)
            }.values()
        )

        try:
            # 全局规则也参与相关性门槛：它可以回答到手价/关税类问题，
            # 但不会再无条件附加并令域外查询命中。
            ranked = self._rerank(category, candidates)
        except Exception:
            logger.warning(
                "品类 reranker 失败，降级到 embedding_only",
                exc_info=True,
            )
            ranked = [
                RetrievedCategoryCard(
                    card=card,
                    score=round(1.0 / rank, 6),
                    recall_strategy="embedding_only",
                )
                for rank, card in enumerate(hybrid_cards, start=1)
            ]
        else:
            # 最高相关分承担查询级域外拒答。入域后保留完整 Top-K，避免
            # 单卡绝对分过滤破坏 Recall 与需要多张卡联合回答的场景。
            if not ranked or ranked[0].score < self._min_relevance_score:
                return ()
        return tuple(ranked[:limit])

    def search_with_strategy(
        self,
        category: str,
        limit: int,
        strategy: CategoryRetrievalStrategy,
    ) -> Sequence[RetrievedCategoryCard]:
        """按指定单一路径检索，供离线消融使用且不触发隐式降级。"""

        if limit < 1:
            return ()
        if strategy == "keyword":
            return self._keyword_search(category, limit)

        self.ensure_resources()
        params: dict[str, str] | None = None
        if strategy == "bm25":
            body = self._bm25_query(category)
        else:
            query_vectors = self._encoder.embed_queries([category])
            if len(query_vectors) != 1:
                raise RuntimeError("BGE 没有为品类查询返回唯一向量")
            query_vector = [float(value) for value in query_vectors[0]]
            if len(query_vector) != self._embedding_dimension:
                raise RuntimeError("BGE 返回的查询向量维度与索引配置不一致")
            if strategy == "knn":
                body = self._knn_query(query_vector)
            else:
                body = self._hybrid_query(category, query_vector)
                params = {"search_pipeline": self._pipeline_name}

        response = self._client.search(
            index=self._index_name,
            body=body,
            **({"params": params} if params is not None else {}),
        )
        cards = self._cards_from_hits(response)
        if strategy == "hybrid_rerank":
            general = [card for card in cards if card.applies_to_all_categories]
            general.extend(self._general_cards())
            candidates = list(
                {
                    card.card_id: card
                    for card in (*cards, *general)
                }.values()
            )
            ranked = self._rerank(category, candidates)
            if not ranked or ranked[0].score < self._min_relevance_score:
                return ()
            return tuple(ranked[:limit])
        else:
            specific = [
                card for card in cards if not card.applies_to_all_categories
            ]
            recall_strategy = {
                "bm25": "bm25",
                "knn": "knn",
                "hybrid_rrf": "hybrid_rrf",
            }[strategy]
            ranked = [
                RetrievedCategoryCard(
                    card=card,
                    score=round(1.0 / rank, 6),
                    recall_strategy=recall_strategy,
                )
                for rank, card in enumerate(specific, start=1)
            ]

        general = [card for card in cards if card.applies_to_all_categories]
        general.extend(self._general_cards())
        specific_ids = {item.card.card_id for item in ranked}
        unique_general = {
            card.card_id: card
            for card in general
            if card.card_id not in specific_ids
        }
        selected_general = list(unique_general.values())[:1]
        result = ranked[: max(0, limit - len(selected_general))]
        result.extend(
            RetrievedCategoryCard(
                card=card,
                score=0.1,
                recall_strategy=ranked[0].recall_strategy,
            )
            for card in selected_general
            if ranked
        )
        return tuple(result[:limit])

    def _keyword_search(
        self,
        category: str,
        limit: int,
    ) -> Sequence[RetrievedCategoryCard]:
        """调用无模型本地检索；未装配本地快照时返回空结果。"""

        if self._keyword_fallback is None:
            return ()
        return self._keyword_fallback.search(category, limit)

    def _rerank(
        self,
        category: str,
        cards: Sequence[CategoryCard],
    ) -> list[RetrievedCategoryCard]:
        """用交叉编码器对混合召回结果做最终相关性排序。"""

        if not cards:
            return []
        scores = self._reranker.score(
            [(category, category_card_search_text(card)) for card in cards]
        )
        if len(scores) != len(cards):
            raise ValueError("BGE reranker 返回的分数数量与候选卡数量不一致")
        if not all(math.isfinite(float(score)) for score in scores):
            raise ValueError("BGE reranker 返回了非有限分数")
        ranked = [
            RetrievedCategoryCard(
                card=card,
                score=min(1.0, max(0.0, float(score))),
                recall_strategy="embedding_rerank",
            )
            for card, score in zip(cards, scores, strict=True)
        ]
        ranked.sort(
            key=lambda item: (item.score, item.card.confidence),
            reverse=True,
        )
        return ranked

    def _general_cards(self) -> list[CategoryCard]:
        """单独查询适用于所有品类的规则，避免语义相似度将其漏召回。"""

        response = self._client.search(
            index=self._index_name,
            body={
                "size": 5,
                "query": {"term": {"applies_to_all_categories": True}},
            },
        )
        return self._cards_from_hits(response)

    def _existing_documents(self) -> dict[str, tuple[str, str]]:
        """读取已有卡片版本，跳过未变化文档的重复 BGE 编码。"""

        response = self._client.search(
            index=self._index_name,
            body={
                "size": 10_000,
                "_source": ["source_hash", "embedding_model"],
                "query": {"match_all": {}},
            },
        )
        return {
            str(hit["_id"]): (
                str(hit.get("_source", {}).get("source_hash", "")),
                str(hit.get("_source", {}).get("embedding_model", "")),
            )
            for hit in response.get("hits", {}).get("hits", [])
        }

    @staticmethod
    def _cards_from_hits(response: dict[str, Any]) -> list[CategoryCard]:
        """从 OpenSearch 命中中恢复经过领域校验的知识卡。"""

        cards: list[CategoryCard] = []
        for hit in response.get("hits", {}).get("hits", []):
            payload = hit.get("_source", {}).get("card_payload")
            if isinstance(payload, dict):
                cards.append(category_card_from_dict(payload))
        return cards

    def _hybrid_query(
        self,
        category: str,
        query_vector: list[float],
    ) -> dict[str, Any]:
        """构造由 Search Pipeline 做加权 RRF 融合的 Hybrid Query。"""

        return {
            "size": self._hybrid_recall_k,
            "_source": {"excludes": ["embedding"]},
            "query": {
                "hybrid": {
                    "pagination_depth": self._hybrid_recall_k,
                    "queries": [
                        {
                            "multi_match": {
                                "query": category,
                                "fields": [
                                    "category^4",
                                    "summary^2",
                                    "search_text",
                                    "raw_evidence",
                                ],
                            }
                        },
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_vector,
                                    "k": self._hybrid_recall_k,
                                }
                            }
                        },
                    ],
                }
            },
        }

    def _bm25_query(self, category: str) -> dict[str, Any]:
        """构造只使用 OpenSearch BM25 的消融查询。"""

        return {
            "size": self._hybrid_recall_k,
            "_source": {"excludes": ["embedding"]},
            "query": {
                "multi_match": {
                    "query": category,
                    "fields": [
                        "category^4",
                        "summary^2",
                        "search_text",
                        "raw_evidence",
                    ],
                }
            },
        }

    def _knn_query(self, query_vector: list[float]) -> dict[str, Any]:
        """构造只使用 BGE 向量近邻的消融查询。"""

        return {
            "size": self._hybrid_recall_k,
            "_source": {"excludes": ["embedding"]},
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_vector,
                        "k": self._hybrid_recall_k,
                    }
                }
            },
        }

    def _index_definition(self) -> dict[str, Any]:
        """定义兼顾中文 BM25 与 BGE-M3 向量召回的索引映射。"""

        text = {"type": "text", "analyzer": "cjk"}
        return {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100,
                    "number_of_replicas": self._number_of_replicas,
                }
            },
            "mappings": {
                "dynamic": "strict",
                "properties": {
                    "card_id": {"type": "keyword"},
                    "category": {
                        **text,
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "card_type": {"type": "keyword"},
                    "summary": text,
                    "search_text": text,
                    "raw_evidence": text,
                    "confidence": {"type": "float"},
                    "last_updated": {"type": "date"},
                    "source_document": {"type": "keyword"},
                    "source_hash": {"type": "keyword"},
                    "embedding_model": {"type": "keyword"},
                    "applies_to_all_categories": {"type": "boolean"},
                    "card_payload": {"type": "object", "enabled": False},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": self._embedding_dimension,
                        "method": {
                            "name": "hnsw",
                            "engine": "faiss",
                            "space_type": "innerproduct",
                            "parameters": {"ef_construction": 128, "m": 24},
                        },
                    },
                },
            },
        }

    def _pipeline_definition(self) -> dict[str, Any]:
        """用加权 RRF 融合 BM25 与 KNN 两路排名，规避量纲不一致。"""

        return {
            "phase_results_processors": [
                {
                    "score-ranker-processor": {
                        "combination": {
                            "technique": "rrf",
                            "rank_constant": 60,
                            "parameters": {
                                "weights": [
                                    self._bm25_weight,
                                    self._knn_weight,
                                ]
                            },
                        }
                    }
                }
            ]
        }

    def _document(
        self,
        card: CategoryCard,
        search_text: str,
        embedding: list[float],
    ) -> dict[str, Any]:
        """把领域卡片转换为 OpenSearch 可索引文档。"""

        return {
            "card_id": card.card_id,
            "category": card.category,
            "card_type": card.card_type.value,
            "summary": card.summary,
            "search_text": search_text,
            "raw_evidence": list(card.raw_evidence),
            "confidence": card.confidence,
            "last_updated": card.last_updated,
            "source_document": card.source_document,
            "source_hash": card.source_hash,
            "embedding_model": self._embedding_model,
            "applies_to_all_categories": card.applies_to_all_categories,
            "card_payload": category_card_to_dict(card),
            "embedding": embedding,
        }
