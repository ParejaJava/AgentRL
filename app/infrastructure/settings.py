"""在 Composition Root 使用的强类型环境配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from app.infrastructure.context_governance.config import GovernanceConfig


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    """应用启动时一次性读取的配置快照。"""

    llm_model_name: str
    llm_api_key: str | None
    llm_base_url: str | None
    llm_temperature: float
    compression_llm_model: str
    compression_llm_max_tokens: int
    context_llm_enabled: bool
    sub_agent_max_concurrency: int
    item_index_root: Path
    item_search_index_id: str
    category_knowledge_root: Path
    category_card_store: Path
    category_ingestion_manifest: Path
    category_structuring_model: str
    category_structuring_max_tokens: int
    category_quick_recall_k: int
    category_deep_recall_k: int
    category_min_confidence: float
    category_retriever_backend: str
    category_embedding_model: str
    category_embedding_dimension: int
    category_reranker_model: str
    category_hybrid_recall_k: int
    opensearch_url: str
    opensearch_username: str | None
    opensearch_password: str | None
    opensearch_verify_certs: bool
    opensearch_timeout_seconds: float
    opensearch_category_index: str
    opensearch_category_pipeline: str
    opensearch_bm25_weight: float
    opensearch_knn_weight: float
    governance: GovernanceConfig

    @classmethod
    def from_env(cls) -> Settings:
        """加载 `.env` 并校验平台启动配置。"""

        load_dotenv()
        model_name = os.getenv("LLM_MODEL_NAME", "qwen-max")
        settings = cls(
            llm_model_name=model_name,
            llm_api_key=os.getenv("LLM_API_KEY"),
            llm_base_url=os.getenv("LLM_BASE_URL"),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            compression_llm_model=os.getenv("COMPRESSION_LLM_MODEL", model_name),
            compression_llm_max_tokens=int(
                os.getenv("COMPRESSION_LLM_MAX_TOKENS", "2048")
            ),
            context_llm_enabled=_env_bool("CONTEXT_LLM_ENABLED", True),
            sub_agent_max_concurrency=int(os.getenv("SUB_AGENT_MAX_CONCURRENCY", "50")),
            item_index_root=Path(os.getenv("ITEM_INDEX_ROOT", "data/indexes")),
            item_search_index_id=os.getenv(
                "ITEM_SEARCH_INDEX_ID", "evaluation-products"
            ),
            category_knowledge_root=Path(
                os.getenv("CATEGORY_KNOWLEDGE_ROOT", "knowledge")
            ),
            category_card_store=Path(
                os.getenv(
                    "CATEGORY_CARD_STORE",
                    "data/category_insight/cards.jsonl",
                )
            ),
            category_ingestion_manifest=Path(
                os.getenv(
                    "CATEGORY_INGESTION_MANIFEST",
                    "data/category_insight/manifest.json",
                )
            ),
            category_structuring_model=os.getenv(
                "CATEGORY_STRUCTURING_MODEL",
                model_name,
            ),
            category_structuring_max_tokens=int(
                os.getenv("CATEGORY_STRUCTURING_MAX_TOKENS", "4096")
            ),
            category_quick_recall_k=int(
                os.getenv("CATEGORY_QUICK_RECALL_K", "8")
            ),
            category_deep_recall_k=int(
                os.getenv("CATEGORY_DEEP_RECALL_K", "20")
            ),
            category_min_confidence=float(
                os.getenv("CATEGORY_MIN_CONFIDENCE", "0.6")
            ),
            category_retriever_backend=os.getenv(
                "CATEGORY_RETRIEVER_BACKEND",
                "local",
            ).strip().lower(),
            category_embedding_model=os.getenv(
                "CATEGORY_EMBEDDING_MODEL",
                "BAAI/bge-m3",
            ),
            category_embedding_dimension=int(
                os.getenv("CATEGORY_EMBEDDING_DIMENSION", "1024")
            ),
            category_reranker_model=os.getenv(
                "CATEGORY_RERANKER_MODEL",
                "BAAI/bge-reranker-v2-m3",
            ),
            category_hybrid_recall_k=int(
                os.getenv("CATEGORY_HYBRID_RECALL_K", "50")
            ),
            opensearch_url=os.getenv(
                "OPENSEARCH_URL",
                "http://localhost:9200",
            ),
            opensearch_username=os.getenv("OPENSEARCH_USERNAME") or None,
            opensearch_password=os.getenv("OPENSEARCH_PASSWORD") or None,
            opensearch_verify_certs=_env_bool("OPENSEARCH_VERIFY_CERTS", False),
            opensearch_timeout_seconds=float(
                os.getenv("OPENSEARCH_TIMEOUT_SECONDS", "10")
            ),
            opensearch_category_index=os.getenv(
                "OPENSEARCH_CATEGORY_INDEX",
                "globex_category_kb",
            ),
            opensearch_category_pipeline=os.getenv(
                "OPENSEARCH_CATEGORY_PIPELINE",
                "globex-category-rrf",
            ),
            opensearch_bm25_weight=float(
                os.getenv("OPENSEARCH_BM25_WEIGHT", "0.4")
            ),
            opensearch_knn_weight=float(
                os.getenv("OPENSEARCH_KNN_WEIGHT", "0.6")
            ),
            governance=GovernanceConfig.from_env(),
        )
        if settings.sub_agent_max_concurrency < 1:
            raise ValueError("SUB_AGENT_MAX_CONCURRENCY 必须大于 0")
        if not settings.item_search_index_id.strip():
            raise ValueError("ITEM_SEARCH_INDEX_ID 不能为空")
        if settings.compression_llm_max_tokens < 1:
            raise ValueError("COMPRESSION_LLM_MAX_TOKENS 必须大于 0")
        if settings.category_structuring_max_tokens < 1:
            raise ValueError("CATEGORY_STRUCTURING_MAX_TOKENS 必须大于 0")
        if settings.category_quick_recall_k < 1:
            raise ValueError("CATEGORY_QUICK_RECALL_K 必须大于 0")
        if settings.category_deep_recall_k < settings.category_quick_recall_k:
            raise ValueError(
                "CATEGORY_DEEP_RECALL_K 不能小于 CATEGORY_QUICK_RECALL_K"
            )
        if not 0.0 <= settings.category_min_confidence <= 1.0:
            raise ValueError("CATEGORY_MIN_CONFIDENCE 必须位于 0 到 1 之间")
        if settings.category_retriever_backend not in {"local", "opensearch"}:
            raise ValueError(
                "CATEGORY_RETRIEVER_BACKEND 只能是 local 或 opensearch"
            )
        if settings.category_embedding_dimension < 1:
            raise ValueError("CATEGORY_EMBEDDING_DIMENSION 必须大于 0")
        if settings.category_hybrid_recall_k < settings.category_deep_recall_k:
            raise ValueError(
                "CATEGORY_HYBRID_RECALL_K 不能小于 CATEGORY_DEEP_RECALL_K"
            )
        if settings.opensearch_timeout_seconds <= 0:
            raise ValueError("OPENSEARCH_TIMEOUT_SECONDS 必须大于 0")
        if not settings.opensearch_category_index.strip():
            raise ValueError("OPENSEARCH_CATEGORY_INDEX 不能为空")
        if not settings.opensearch_category_pipeline.strip():
            raise ValueError("OPENSEARCH_CATEGORY_PIPELINE 不能为空")
        weights = (
            settings.opensearch_bm25_weight,
            settings.opensearch_knn_weight,
        )
        if any(weight < 0.0 or weight > 1.0 for weight in weights):
            raise ValueError("OpenSearch Hybrid 权重必须位于 0 到 1 之间")
        if abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("OPENSEARCH_BM25_WEIGHT 与 KNN 权重之和必须为 1")
        return settings
