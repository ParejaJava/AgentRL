"""Configuration for the ItemSearch recall and reranking pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ItemSearchConfig:
    """Runtime defaults shared by offline indexing and online retrieval."""

    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    embedding_dimension: int = 1024
    recall_k: int = 100
    result_k: int = 10
    max_result_k: int = 50
    query_weight: float = 0.8
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 128

    def __post_init__(self) -> None:
        if self.embedding_dimension < 1:
            raise ValueError("embedding_dimension must be positive")
        if self.recall_k < 1:
            raise ValueError("recall_k must be positive")
        if not 1 <= self.result_k <= self.max_result_k:
            raise ValueError("result_k must be between 1 and max_result_k")
        if self.max_result_k > self.recall_k:
            raise ValueError("max_result_k cannot exceed recall_k")
        if not 0.0 <= self.query_weight <= 1.0:
            raise ValueError("query_weight must be between 0 and 1")
        if self.hnsw_m < 2:
            raise ValueError("hnsw_m must be at least 2")
        if self.hnsw_ef_construction < 1 or self.hnsw_ef_search < 1:
            raise ValueError("HNSW ef values must be positive")
