"""品类洞察本地/OpenSearch 适配器的默认装配函数。"""

from pathlib import Path

from app.application.catalog import CategoryInsightConfig, GetCategoryInsight
from app.application.catalog.category_insight_ports import (
    CategoryKnowledgeRetriever,
)
from app.infrastructure.retrieval.item_search.bge import (
    BGEEmbeddingEncoder,
    BGEReranker,
)
from app.infrastructure.settings import Settings

from .local_store import JsonlCategoryCardStore, LocalCategoryCardRetriever
from .opensearch import OpenSearchCategoryCardRepository, create_opensearch_client


def create_local_category_insight_service(
    card_store_path: Path,
    *,
    config: CategoryInsightConfig | None = None,
) -> GetCategoryInsight:
    """创建无需 OpenSearch、不会触发模型加载的本地查询服务。"""

    store = JsonlCategoryCardStore(card_store_path)
    retriever = LocalCategoryCardRetriever(store)
    return GetCategoryInsight(retriever, config)


def create_opensearch_category_repository(
    settings: Settings,
) -> OpenSearchCategoryCardRepository:
    """装配 OpenSearch Hybrid Search 与 BGE 重排适配器。"""

    client = create_opensearch_client(
        settings.opensearch_url,
        username=settings.opensearch_username,
        password=settings.opensearch_password,
        verify_certs=settings.opensearch_verify_certs,
        timeout_seconds=settings.opensearch_timeout_seconds,
    )
    encoder = BGEEmbeddingEncoder(
        settings.category_embedding_model,
        dimension=settings.category_embedding_dimension,
        batch_size=settings.retrieval_embedding_batch_size,
        use_fp16=settings.retrieval_use_fp16,
        device=settings.retrieval_device,
    )
    reranker = BGEReranker(
        settings.category_reranker_model,
        use_fp16=settings.retrieval_use_fp16,
        device=settings.retrieval_device,
        batch_size=settings.retrieval_reranker_batch_size,
    )
    keyword_fallback = LocalCategoryCardRetriever(
        JsonlCategoryCardStore(settings.category_card_store)
    )
    return OpenSearchCategoryCardRepository(
        client,
        encoder,
        reranker,
        index_name=settings.opensearch_category_index,
        pipeline_name=settings.opensearch_category_pipeline,
        embedding_dimension=settings.category_embedding_dimension,
        embedding_model=settings.category_embedding_model,
        hybrid_recall_k=settings.category_hybrid_recall_k,
        bm25_weight=settings.opensearch_bm25_weight,
        knn_weight=settings.opensearch_knn_weight,
        keyword_fallback=keyword_fallback,
    )


def create_category_insight_service(
    settings: Settings,
    *,
    config: CategoryInsightConfig | None = None,
) -> GetCategoryInsight:
    """根据配置选择本地降级检索或 OpenSearch 生产检索。"""

    return GetCategoryInsight(
        create_category_knowledge_retriever(settings),
        config,
    )


def create_category_knowledge_retriever(
    settings: Settings,
) -> CategoryKnowledgeRetriever:
    """根据配置装配评测和在线服务共用的检索端口。"""

    if settings.category_retriever_backend == "opensearch":
        return create_opensearch_category_repository(settings)
    return LocalCategoryCardRetriever(
        JsonlCategoryCardStore(settings.category_card_store)
    )
