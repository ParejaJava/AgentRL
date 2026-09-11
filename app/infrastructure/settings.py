"""在 Composition Root 使用的强类型环境配置。"""

from __future__ import annotations

import os
import re
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
    model_max_concurrency: int
    model_min_interval_seconds: float
    model_max_retries: int
    model_run_max_requests: int
    model_run_max_observed_tokens: int
    fallback_llm_model: str | None
    lite_llm_model: str | None
    token_budget_total: int
    tool_timeout_seconds: float
    tool_failure_threshold: int
    tool_recovery_seconds: float
    tool_repeated_call_limit: int
    breaker_shared: bool
    drift_detect_enabled: bool
    drift_check_interval: int
    langfuse_enabled: bool
    redis_enabled: bool
    redis_url: str
    redis_key_prefix: str
    queue_max_attempts: int
    queue_large_request_turns: int
    checkpoint_backend: str
    checkpoint_database_path: Path
    tavily_api_key: str | None
    embedding_cache_enabled: bool
    embedding_cache_ttl_seconds: int
    semantic_cache_enabled: bool
    semantic_cache_threshold: float
    semantic_cache_ttl_seconds: int
    item_index_root: Path
    item_search_index_id: str
    retrieval_device: str
    retrieval_use_fp16: bool
    retrieval_embedding_batch_size: int
    retrieval_reranker_batch_size: int
    category_knowledge_root: Path
    category_card_store: Path
    category_ingestion_manifest: Path
    category_structuring_model: str
    category_structuring_max_tokens: int
    category_ingestion_max_concurrency: int
    category_quick_recall_k: int
    category_deep_recall_k: int
    category_min_confidence: float
    category_retriever_backend: str
    category_embedding_model: str
    category_embedding_dimension: int
    category_reranker_model: str
    category_hybrid_recall_k: int
    category_min_relevance_score: float
    opensearch_url: str
    opensearch_username: str | None
    opensearch_password: str | None
    opensearch_verify_certs: bool
    opensearch_timeout_seconds: float
    opensearch_category_index: str
    opensearch_category_pipeline: str
    opensearch_number_of_replicas: int
    opensearch_bm25_weight: float
    opensearch_knn_weight: float
    preference_database_path: Path
    preference_max_likes: int
    preference_prompt_like_limit: int
    order_database_path: Path
    governance: GovernanceConfig
    executor_model_name: str | None = None
    executor_base_url: str | None = None
    executor_api_key: str | None = None
    executor_context_window: int | None = None
    executor_fallback_model: str | None = None
    executor_lite_model: str | None = None
    trajectory_root: Path | None = None

    @classmethod
    def from_env(cls) -> Settings:
        """加载 `.env` 并校验平台启动配置。"""

        load_dotenv()
        model_name = os.getenv("LLM_MODEL_NAME", "qwen-max")
        settings = cls(
            executor_model_name=os.getenv("EXECUTOR_MODEL_NAME") or None,
            executor_base_url=os.getenv("EXECUTOR_BASE_URL") or None,
            executor_api_key=os.getenv("EXECUTOR_API_KEY") or None,
            executor_context_window=(
                int(os.environ["EXECUTOR_CONTEXT_WINDOW"])
                if os.getenv("EXECUTOR_CONTEXT_WINDOW")
                else None
            ),
            executor_fallback_model=os.getenv("EXECUTOR_FALLBACK_MODEL") or None,
            executor_lite_model=os.getenv("EXECUTOR_LITE_MODEL") or None,
            trajectory_root=(
                Path(os.environ["TRAJECTORY_ROOT"])
                if os.getenv("TRAJECTORY_ROOT")
                else None
            ),
            llm_model_name=model_name,
            llm_api_key=os.getenv("LLM_API_KEY"),
            llm_base_url=os.getenv("LLM_BASE_URL"),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            compression_llm_model=os.getenv("COMPRESSION_LLM_MODEL", model_name),
            compression_llm_max_tokens=int(
                os.getenv("COMPRESSION_LLM_MAX_TOKENS", "2048")
            ),
            context_llm_enabled=_env_bool("CONTEXT_LLM_ENABLED", True),
            sub_agent_max_concurrency=int(os.getenv("SUB_AGENT_MAX_CONCURRENCY", "10")),
            model_max_concurrency=int(os.getenv("MODEL_MAX_CONCURRENCY", "12")),
            model_min_interval_seconds=float(
                os.getenv("MODEL_MIN_INTERVAL_SECONDS", "0")
            ),
            model_max_retries=int(os.getenv("MODEL_MAX_RETRIES", "2")),
            model_run_max_requests=int(os.getenv("MODEL_RUN_MAX_REQUESTS", "0")),
            model_run_max_observed_tokens=int(
                os.getenv("MODEL_RUN_MAX_OBSERVED_TOKENS", "0")
            ),
            fallback_llm_model=os.getenv("FALLBACK_LLM_MODEL") or None,
            lite_llm_model=os.getenv("LITE_LLM_MODEL") or None,
            token_budget_total=int(os.getenv("TOKEN_BUDGET_TOTAL", "0")),
            tool_timeout_seconds=float(os.getenv("TOOL_TIMEOUT_SECONDS", "30")),
            tool_failure_threshold=int(os.getenv("TOOL_FAILURE_THRESHOLD", "3")),
            tool_recovery_seconds=float(os.getenv("TOOL_RECOVERY_SECONDS", "30")),
            tool_repeated_call_limit=int(os.getenv("TOOL_REPEATED_CALL_LIMIT", "3")),
            breaker_shared=_env_bool("BREAKER_SHARED", False),
            drift_detect_enabled=_env_bool("DRIFT_DETECT_ENABLED", False),
            drift_check_interval=int(os.getenv("DRIFT_CHECK_INTERVAL", "3")),
            langfuse_enabled=_env_bool("LANGFUSE_ENABLED", False),
            redis_enabled=_env_bool("REDIS_ENABLED", False),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            redis_key_prefix=os.getenv("REDIS_KEY_PREFIX", "globex").strip(),
            queue_max_attempts=int(os.getenv("QUEUE_MAX_ATTEMPTS", "3")),
            queue_large_request_turns=int(os.getenv("QUEUE_LARGE_REQUEST_TURNS", "30")),
            checkpoint_backend=os.getenv("CHECKPOINT_BACKEND", "memory")
            .strip()
            .lower(),
            checkpoint_database_path=Path(
                os.getenv(
                    "CHECKPOINT_DATABASE_PATH",
                    "data/sessions/langgraph-checkpoints.db",
                )
            ),
            tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
            embedding_cache_enabled=_env_bool("EMBEDDING_CACHE_ENABLED", False),
            embedding_cache_ttl_seconds=int(
                os.getenv("EMBEDDING_CACHE_TTL_SECONDS", "604800")
            ),
            semantic_cache_enabled=_env_bool("SEMANTIC_CACHE_ENABLED", False),
            semantic_cache_threshold=float(
                os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.95")
            ),
            semantic_cache_ttl_seconds=int(
                os.getenv("SEMANTIC_CACHE_TTL_SECONDS", "86400")
            ),
            item_index_root=Path(os.getenv("ITEM_INDEX_ROOT", "data/indexes")),
            item_search_index_id=os.getenv(
                "ITEM_SEARCH_INDEX_ID", "evaluation-products"
            ),
            retrieval_device=os.getenv("RETRIEVAL_DEVICE", "auto").strip().lower(),
            retrieval_use_fp16=_env_bool("RETRIEVAL_USE_FP16", False),
            retrieval_embedding_batch_size=int(
                os.getenv("RETRIEVAL_EMBEDDING_BATCH_SIZE", "16")
            ),
            retrieval_reranker_batch_size=int(
                os.getenv("RETRIEVAL_RERANKER_BATCH_SIZE", "8")
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
            category_ingestion_max_concurrency=int(
                os.getenv("CATEGORY_INGESTION_MAX_CONCURRENCY", "5")
            ),
            category_quick_recall_k=int(os.getenv("CATEGORY_QUICK_RECALL_K", "8")),
            category_deep_recall_k=int(os.getenv("CATEGORY_DEEP_RECALL_K", "20")),
            category_min_confidence=float(os.getenv("CATEGORY_MIN_CONFIDENCE", "0.6")),
            category_retriever_backend=os.getenv(
                "CATEGORY_RETRIEVER_BACKEND",
                "local",
            )
            .strip()
            .lower(),
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
            category_hybrid_recall_k=int(os.getenv("CATEGORY_HYBRID_RECALL_K", "50")),
            category_min_relevance_score=float(
                os.getenv("CATEGORY_MIN_RELEVANCE_SCORE", "0.01")
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
            opensearch_number_of_replicas=int(
                os.getenv("OPENSEARCH_NUMBER_OF_REPLICAS", "0")
            ),
            opensearch_bm25_weight=float(os.getenv("OPENSEARCH_BM25_WEIGHT", "0.4")),
            opensearch_knn_weight=float(os.getenv("OPENSEARCH_KNN_WEIGHT", "0.6")),
            preference_database_path=Path(
                os.getenv(
                    "PREFERENCE_DATABASE_PATH",
                    "data/sessions/preferences.db",
                )
            ),
            preference_max_likes=int(os.getenv("PREFERENCE_MAX_LIKES", "50")),
            preference_prompt_like_limit=int(
                os.getenv("PREFERENCE_PROMPT_LIKE_LIMIT", "8")
            ),
            order_database_path=Path(
                os.getenv("ORDER_DATABASE_PATH", "data/orders/orders.db")
            ),
            governance=GovernanceConfig.from_env(),
        )
        if settings.sub_agent_max_concurrency < 1:
            raise ValueError("SUB_AGENT_MAX_CONCURRENCY 必须大于 0")
        if settings.model_max_concurrency < 1:
            raise ValueError("MODEL_MAX_CONCURRENCY 必须大于 0")
        if settings.model_min_interval_seconds < 0 or settings.model_max_retries < 0:
            raise ValueError("模型请求间隔和重试次数不能小于 0")
        if (
            min(
                settings.model_run_max_requests,
                settings.model_run_max_observed_tokens,
            )
            < 0
        ):
            raise ValueError("证据运行请求和 Token 硬上限不能小于 0")
        if settings.token_budget_total < 0:
            raise ValueError("TOKEN_BUDGET_TOTAL 不能小于 0")
        if settings.tool_timeout_seconds <= 0:
            raise ValueError("TOOL_TIMEOUT_SECONDS 必须大于 0")
        if (
            min(
                settings.tool_failure_threshold,
                settings.tool_repeated_call_limit,
                settings.drift_check_interval,
            )
            < 1
            or settings.tool_recovery_seconds < 0
        ):
            raise ValueError("工具熔断和重复调用参数不合法")
        if not settings.redis_url or not settings.redis_key_prefix:
            raise ValueError("REDIS_URL 和 REDIS_KEY_PREFIX 不能为空")
        if min(settings.queue_max_attempts, settings.queue_large_request_turns) < 1:
            raise ValueError("Redis 队列重试次数和大请求轮次必须大于 0")
        if settings.queue_max_attempts < 1:
            raise ValueError("QUEUE_MAX_ATTEMPTS 必须大于 0")
        if settings.checkpoint_backend not in {"memory", "sqlite"}:
            raise ValueError("CHECKPOINT_BACKEND 只能是 memory 或 sqlite")
        if settings.embedding_cache_ttl_seconds < 1:
            raise ValueError("EMBEDDING_CACHE_TTL_SECONDS 必须大于 0")
        if settings.embedding_cache_enabled and not settings.redis_enabled:
            raise ValueError("Embedding 缓存需要同时启用 Redis")
        if not 0.0 < settings.semantic_cache_threshold <= 1.0:
            raise ValueError("SEMANTIC_CACHE_THRESHOLD 必须位于 0 到 1 之间")
        if settings.semantic_cache_ttl_seconds < 1:
            raise ValueError("SEMANTIC_CACHE_TTL_SECONDS 必须大于 0")
        if settings.semantic_cache_enabled and not settings.redis_enabled:
            raise ValueError("语义响应缓存需要同时启用 Redis")
        if not settings.item_search_index_id.strip():
            raise ValueError("ITEM_SEARCH_INDEX_ID 不能为空")
        if not settings.retrieval_device:
            raise ValueError("RETRIEVAL_DEVICE 不能为空")
        if not re.fullmatch(r"auto|cpu|mps|cuda(?::\d+)?", settings.retrieval_device):
            raise ValueError(
                "RETRIEVAL_DEVICE 只能是 auto、cpu、mps、cuda 或 cuda:<序号>"
            )
        if settings.retrieval_embedding_batch_size < 1:
            raise ValueError("RETRIEVAL_EMBEDDING_BATCH_SIZE 必须大于 0")
        if settings.retrieval_reranker_batch_size < 1:
            raise ValueError("RETRIEVAL_RERANKER_BATCH_SIZE 必须大于 0")
        if (
            settings.executor_context_window is not None
            and settings.executor_context_window < 1
        ):
            raise ValueError("EXECUTOR_CONTEXT_WINDOW must be positive")
        if settings.executor_base_url and not settings.executor_model_name:
            raise ValueError("EXECUTOR_BASE_URL requires EXECUTOR_MODEL_NAME")
        if settings.compression_llm_max_tokens < 1:
            raise ValueError("COMPRESSION_LLM_MAX_TOKENS 必须大于 0")
        if settings.category_structuring_max_tokens < 1:
            raise ValueError("CATEGORY_STRUCTURING_MAX_TOKENS 必须大于 0")
        if settings.category_ingestion_max_concurrency < 1:
            raise ValueError("CATEGORY_INGESTION_MAX_CONCURRENCY 必须大于 0")
        if settings.category_quick_recall_k < 1:
            raise ValueError("CATEGORY_QUICK_RECALL_K 必须大于 0")
        if settings.category_deep_recall_k < settings.category_quick_recall_k:
            raise ValueError("CATEGORY_DEEP_RECALL_K 不能小于 CATEGORY_QUICK_RECALL_K")
        if not 0.0 <= settings.category_min_confidence <= 1.0:
            raise ValueError("CATEGORY_MIN_CONFIDENCE 必须位于 0 到 1 之间")
        if settings.category_retriever_backend not in {"local", "opensearch"}:
            raise ValueError("CATEGORY_RETRIEVER_BACKEND 只能是 local 或 opensearch")
        if settings.category_embedding_dimension < 1:
            raise ValueError("CATEGORY_EMBEDDING_DIMENSION 必须大于 0")
        if settings.category_hybrid_recall_k < settings.category_deep_recall_k:
            raise ValueError("CATEGORY_HYBRID_RECALL_K 不能小于 CATEGORY_DEEP_RECALL_K")
        if not 0.0 <= settings.category_min_relevance_score <= 1.0:
            raise ValueError("CATEGORY_MIN_RELEVANCE_SCORE 必须位于 0 到 1 之间")
        if settings.opensearch_timeout_seconds <= 0:
            raise ValueError("OPENSEARCH_TIMEOUT_SECONDS 必须大于 0")
        if not settings.opensearch_category_index.strip():
            raise ValueError("OPENSEARCH_CATEGORY_INDEX 不能为空")
        if not settings.opensearch_category_pipeline.strip():
            raise ValueError("OPENSEARCH_CATEGORY_PIPELINE 不能为空")
        if settings.opensearch_number_of_replicas < 0:
            raise ValueError("OPENSEARCH_NUMBER_OF_REPLICAS 不能小于 0")
        weights = (
            settings.opensearch_bm25_weight,
            settings.opensearch_knn_weight,
        )
        if any(weight < 0.0 or weight > 1.0 for weight in weights):
            raise ValueError("OpenSearch Hybrid 权重必须位于 0 到 1 之间")
        if abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("OPENSEARCH_BM25_WEIGHT 与 KNN 权重之和必须为 1")
        if settings.preference_max_likes < 1:
            raise ValueError("PREFERENCE_MAX_LIKES 必须大于 0")
        if settings.preference_prompt_like_limit < 0:
            raise ValueError("PREFERENCE_PROMPT_LIKE_LIMIT 不能小于 0")
        return settings
