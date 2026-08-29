"""品类知识的摄取与本地检索适配器。"""

from .ingestion import (
    CategoryIngestionReport,
    CategoryKnowledgeIngestor,
    StructuredLLMCategoryExtractor,
)
from .local_store import JsonlCategoryCardStore, LocalCategoryCardRetriever
from .opensearch import OpenSearchCategoryCardRepository

__all__ = [
    "CategoryIngestionReport",
    "CategoryKnowledgeIngestor",
    "JsonlCategoryCardStore",
    "LocalCategoryCardRetriever",
    "OpenSearchCategoryCardRepository",
    "StructuredLLMCategoryExtractor",
]
