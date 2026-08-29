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
) -> ItemSearchService:
    """装配生产商品搜索服务，但不提前加载两个 BGE 模型。"""

    resolved_config = config or ItemSearchConfig()
    encoder = BGEEmbeddingEncoder(
        resolved_config.embedding_model,
        dimension=resolved_config.embedding_dimension,
        use_fp16=use_fp16,
    )
    reranker = BGEReranker(
        resolved_config.reranker_model,
        use_fp16=use_fp16,
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
) -> ItemIndexBuilder:
    """装配与在线检索使用相同 BGE-M3 配置的离线索引构建器。"""

    resolved_config = config or ItemSearchConfig()
    encoder = BGEEmbeddingEncoder(
        resolved_config.embedding_model,
        dimension=resolved_config.embedding_dimension,
        use_fp16=use_fp16,
    )
    return ItemIndexBuilder(encoder, config=resolved_config)
