"""Default wiring for the production BGE and FAISS adapters."""

from dataclasses import replace
from pathlib import Path

from app.application.catalog.config import ItemRetrievalMode, ItemSearchConfig
from app.application.catalog.ports import (
    EmbeddingEncoder,
    IndexRegistry,
    NullUserSignalProvider,
    UserSignalProvider,
)
from app.application.catalog.search_catalog import ItemSearchService
from app.infrastructure.cache import CachedEmbeddingEncoder, VectorCache
from app.infrastructure.pricing import create_static_pricing_provider

from .bge import BGEEmbeddingEncoder, BGEReranker
from .builder import ItemIndexBuilder
from .faiss_index import FaissDirectoryIndexRegistry
from .user_tower import EmbeddingUserSignalProvider, UserProfileSource


def create_item_search_service(
    index_root: Path,
    *,
    config: ItemSearchConfig | None = None,
    user_signals: UserSignalProvider | None = None,
    profile_source: UserProfileSource | None = None,
    indexes: IndexRegistry | None = None,
    vector_cache: VectorCache | None = None,
    encoder: EmbeddingEncoder | None = None,
    use_fp16: bool = False,
    device: str = "auto",
    embedding_batch_size: int = 16,
    reranker_batch_size: int = 8,
    retrieval_mode: ItemRetrievalMode | None = None,
) -> ItemSearchService:
    """按统一设备、精度和批大小装配商品搜索服务。"""

    resolved_config = config or ItemSearchConfig()
    if retrieval_mode is not None:
        resolved_config = replace(resolved_config, retrieval_mode=retrieval_mode)
    resolved_encoder = encoder or create_item_embedding_encoder(
        config=resolved_config,
        vector_cache=vector_cache,
        use_fp16=use_fp16,
        device=device,
        embedding_batch_size=embedding_batch_size,
    )
    reranker = BGEReranker(
        resolved_config.reranker_model,
        use_fp16=use_fp16,
        device=device,
        batch_size=reranker_batch_size,
    )
    resolved_indexes = indexes or create_item_index_registry(
        index_root,
        config=resolved_config,
    )
    resolved_user_signals = user_signals
    if resolved_user_signals is None and profile_source is not None:
        resolved_user_signals = EmbeddingUserSignalProvider(
            resolved_encoder,
            profile_source,
        )
    return ItemSearchService(
        encoder=resolved_encoder,
        reranker=reranker,
        indexes=resolved_indexes,
        user_signals=resolved_user_signals or NullUserSignalProvider(),
        pricing=create_static_pricing_provider(),
        config=resolved_config,
    )


def create_item_embedding_encoder(
    *,
    config: ItemSearchConfig | None = None,
    vector_cache: VectorCache | None = None,
    use_fp16: bool = False,
    device: str = "auto",
    embedding_batch_size: int = 16,
) -> EmbeddingEncoder:
    """创建可被商品检索、User Tower 与语义缓存共享的编码器。"""

    resolved_config = config or ItemSearchConfig()
    encoder = BGEEmbeddingEncoder(
        resolved_config.embedding_model,
        dimension=resolved_config.embedding_dimension,
        batch_size=embedding_batch_size,
        use_fp16=use_fp16,
        device=device,
    )
    if vector_cache is None:
        return encoder
    return CachedEmbeddingEncoder(
        encoder,
        vector_cache,
        model_revision=resolved_config.embedding_model,
    )


def create_item_index_registry(
    index_root: Path,
    *,
    config: ItemSearchConfig | None = None,
) -> FaissDirectoryIndexRegistry:
    """创建可被搜索与订单读取共同复用的商品索引注册表。"""

    resolved_config = config or ItemSearchConfig()
    return FaissDirectoryIndexRegistry(
        index_root,
        expected_embedding_model=resolved_config.embedding_model,
        expected_dimension=resolved_config.embedding_dimension,
    )


def create_item_index_builder(
    *,
    config: ItemSearchConfig | None = None,
    use_fp16: bool = False,
    device: str = "auto",
    embedding_batch_size: int = 16,
) -> ItemIndexBuilder:
    """按线上同款推理配置装配离线 BGE-M3 索引构建器。"""

    resolved_config = config or ItemSearchConfig()
    encoder = BGEEmbeddingEncoder(
        resolved_config.embedding_model,
        dimension=resolved_config.embedding_dimension,
        batch_size=embedding_batch_size,
        use_fp16=use_fp16,
        device=device,
    )
    return ItemIndexBuilder(encoder, config=resolved_config)
