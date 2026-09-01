"""Default wiring for the production BGE and FAISS adapters."""

from pathlib import Path

from app.application.catalog.config import ItemSearchConfig
from app.application.catalog.ports import NullUserSignalProvider, UserSignalProvider
from app.application.catalog.search_catalog import ItemSearchService
from app.infrastructure.pricing import create_static_pricing_provider

from .bge import BGEEmbeddingEncoder, BGEReranker
from .builder import ItemIndexBuilder
from .faiss_index import FaissDirectoryIndexRegistry


def create_item_search_service(
    index_root: Path,
    *,
    config: ItemSearchConfig | None = None,
    user_signals: UserSignalProvider | None = None,
    use_fp16: bool = False,
    device: str = "auto",
    embedding_batch_size: int = 16,
    reranker_batch_size: int = 8,
) -> ItemSearchService:
    """按统一设备、精度和批大小装配商品搜索服务。"""

    resolved_config = config or ItemSearchConfig()
    encoder = BGEEmbeddingEncoder(
        resolved_config.embedding_model,
        dimension=resolved_config.embedding_dimension,
        batch_size=embedding_batch_size,
        use_fp16=use_fp16,
        device=device,
    )
    reranker = BGEReranker(
        resolved_config.reranker_model,
        use_fp16=use_fp16,
        device=device,
        batch_size=reranker_batch_size,
    )
    indexes = FaissDirectoryIndexRegistry(
        index_root,
        expected_embedding_model=resolved_config.embedding_model,
        expected_dimension=resolved_config.embedding_dimension,
    )
    return ItemSearchService(
        encoder=encoder,
        reranker=reranker,
        indexes=indexes,
        user_signals=user_signals or NullUserSignalProvider(),
        pricing=create_static_pricing_provider(),
        config=resolved_config,
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
