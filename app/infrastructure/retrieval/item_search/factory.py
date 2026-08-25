"""Default wiring for the production BGE and FAISS adapters."""

from pathlib import Path

from app.application.catalog.config import ItemSearchConfig
from app.application.catalog.ports import NullUserSignalProvider, UserSignalProvider
from app.application.catalog.search_catalog import ItemSearchService

from .bge import BGEEmbeddingEncoder, BGEReranker
from .faiss_index import FaissDirectoryIndexRegistry


def create_item_search_service(
    index_root: Path,
    *,
    config: ItemSearchConfig | None = None,
    user_signals: UserSignalProvider | None = None,
    use_fp16: bool = False,
) -> ItemSearchService:
    """Create the default service without eagerly loading either BGE model."""

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
        config=resolved_config,
    )
