"""品类知识的摄取与本地检索适配器。"""

from .ingestion import (
    CategoryIngestionReport,
    CategoryKnowledgeIngestor,
    StructuredLLMCategoryExtractor,
)
from .local_store import JsonlCategoryCardStore, LocalCategoryCardRetriever
from .opensearch import CategoryRetrievalStrategy, OpenSearchCategoryCardRepository

__all__ = [
    "CategoryIngestionReport",
    "CategoryKnowledgeIngestor",
    "CategoryRetrievalStrategy",
    "JsonlCategoryCardStore",
    "LocalCategoryCardRetriever",
    "OpenSearchCategoryCardRepository",
    "StructuredLLMCategoryExtractor",
]
