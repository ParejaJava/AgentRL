"""API 与 Worker 共用的唯一依赖装配入口。"""

import hashlib
from dataclasses import dataclass

from app.application.agents import DriftDetector, RunAgent
from app.application.catalog import CategoryInsightConfig, ItemRetrievalMode
from app.application.catalog.category_insight_ports import (
    CategoryKnowledgeRetriever,
)
from app.application.catalog.ports import EmbeddingEncoder, IndexRegistry
from app.application.catalog.search_catalog import ItemSearchService
from app.application.memory import PreferenceSelector
from app.application.orders import OrderService
from app.application.tasking import TaskBoardService
from app.infrastructure.cache import (
    RedisSemanticResponseCache,
    RedisVectorCache,
    VectorCache,
)
from app.infrastructure.checkpoint import LazyAsyncSqliteSaver
from app.infrastructure.context_governance.compressor import StructuredLLMCompressor
from app.infrastructure.eventbus import InMemoryTradeEventBus, RedisTradeEventBus
from app.infrastructure.evidence_budget import EvidenceUsageBudget
from app.infrastructure.health import OpenSearchReadinessProbe
from app.infrastructure.langchain import MainAgent
from app.infrastructure.langchain.preference_middleware import (
    PreferenceMemoryMiddleware,
)
from app.infrastructure.langchain.prompts import MAIN_SYSTEM_PROMPT
from app.infrastructure.langchain.reliability_middleware import (
    ModelGatewayMiddleware,
    ToolHarnessMiddleware,
    ToolResilienceConfig,
    ToolResilienceMiddleware,
)
from app.infrastructure.langchain.tools import (
    create_category_insight_tool,
    create_item_search_tool,
    create_order_tools,
    create_preference_tools,
    create_task_tools,
    create_web_search_tool,
)
from app.infrastructure.llm import (
    create_category_structuring_model,
    create_chat_model,
    create_compression_model,
)
from app.infrastructure.memory import SQLitePreferenceStore
from app.infrastructure.observability import LangfuseCallbacks
from app.infrastructure.orders import IndexedProductReader, SQLiteOrderRepository
from app.infrastructure.queue import RedisStreamTaskQueue
from app.infrastructure.resilience import RedisSharedCircuitBreaker
from app.infrastructure.retrieval.category_insight import (
    CategoryKnowledgeIngestor,
    JsonlCategoryCardStore,
    StructuredLLMCategoryExtractor,
)
from app.infrastructure.retrieval.category_insight.factory import (
    create_category_insight_service,
    create_category_knowledge_retriever,
    create_opensearch_category_repository,
)
from app.infrastructure.retrieval.item_search.builder import ItemIndexBuilder
from app.infrastructure.retrieval.item_search.factory import (
    create_item_embedding_encoder,
    create_item_index_builder,
    create_item_index_registry,
    create_item_search_service,
)
from app.infrastructure.retrieval.item_search.user_tower import (
    StoredPreferenceProfileSource,
    UserProfileSource,
)
from app.infrastructure.settings import Settings
from app.infrastructure.tasking import SQLiteTaskBoardRepository
from app.infrastructure.web_search import TavilyWebSearch


@dataclass(frozen=True, slots=True)
class ApplicationContainer:
    """保存已经完成装配的应用用例和基础设施 Driver。"""

    settings: Settings
    run_agent: RunAgent
    runtime: MainAgent
    event_bus: InMemoryTradeEventBus | RedisTradeEventBus
    orders: OrderService
    task_queue: RedisStreamTaskQueue | None
    opensearch_probe: OpenSearchReadinessProbe | None
    model_gateway: ModelGatewayMiddleware


def build_container(settings: Settings | None = None) -> ApplicationContainer:
    """创建平台所有具体依赖；其他模块不得自行组装实现。"""

    resolved = settings or Settings.from_env()
    model = create_chat_model(resolved)
    fallback_model = (
        create_chat_model(resolved, model_name=resolved.fallback_llm_model)
        if resolved.fallback_llm_model
        else None
    )
    lite_model = (
        create_chat_model(resolved, model_name=resolved.lite_llm_model)
        if resolved.lite_llm_model
        else None
    )
    # 主/子 Agent 与压缩 LLM 必须共享总请求数和 Token 上限。
    evidence_budget = EvidenceUsageBudget(
        max_total_requests=resolved.model_run_max_requests,
        max_observed_tokens=resolved.model_run_max_observed_tokens,
        min_interval_seconds=resolved.model_min_interval_seconds,
    )
    compressor = (
        StructuredLLMCompressor(
            create_compression_model(resolved),
            evidence_budget=evidence_budget,
        )
        if resolved.context_llm_enabled
        else None
    )
    preference_store = SQLitePreferenceStore(
        resolved.preference_database_path,
        max_likes_per_buyer=resolved.preference_max_likes,
    )
    preference_selector = PreferenceSelector(
        like_limit=resolved.preference_prompt_like_limit
    )
    preference_profiles = StoredPreferenceProfileSource(
        preference_store,
        preference_selector,
    )
    item_indexes = create_item_index_registry(resolved.item_index_root)
    vector_cache: VectorCache | None = (
        RedisVectorCache(
            resolved.redis_url,
            prefix=f"{resolved.redis_key_prefix}:embedding",
            ttl_seconds=resolved.embedding_cache_ttl_seconds,
        )
        if resolved.embedding_cache_enabled
        else None
    )
    item_encoder = create_item_embedding_encoder(
        vector_cache=vector_cache,
        use_fp16=resolved.retrieval_use_fp16,
        device=resolved.retrieval_device,
        embedding_batch_size=resolved.retrieval_embedding_batch_size,
    )
    item_search_service = build_item_search_service(
        resolved,
        profile_source=preference_profiles,
        indexes=item_indexes,
        vector_cache=vector_cache,
        encoder=item_encoder,
    )
    order_service = OrderService(
        IndexedProductReader(item_indexes),
        SQLiteOrderRepository(resolved.order_database_path),
    )
    category_insight_service = create_category_insight_service(
        resolved,
        config=CategoryInsightConfig(
            quick_recall_k=resolved.category_quick_recall_k,
            deep_recall_k=resolved.category_deep_recall_k,
            min_confidence=resolved.category_min_confidence,
        ),
    )
    tools = [
        create_item_search_tool(
            item_search_service,
            index_id=resolved.item_search_index_id,
        ),
        create_category_insight_tool(category_insight_service),
    ]
    if resolved.tavily_api_key:
        tools.append(create_web_search_tool(TavilyWebSearch(resolved.tavily_api_key)))
    task_service = TaskBoardService(
        SQLiteTaskBoardRepository(
            resolved.governance.session_root / "task_boards.db"
        )
    )
    task_tools = create_task_tools(task_service)
    preference_tools = create_preference_tools(preference_store)
    order_tools = create_order_tools(
        order_service,
        index_id=resolved.item_search_index_id,
    )
    preference_middleware = PreferenceMemoryMiddleware(
        preference_store,
        preference_selector,
    )
    model_gateway = ModelGatewayMiddleware(
        max_concurrency=resolved.model_max_concurrency,
        min_interval_seconds=resolved.model_min_interval_seconds,
        max_retries=resolved.model_max_retries,
        max_total_requests=resolved.model_run_max_requests,
        max_observed_tokens=resolved.model_run_max_observed_tokens,
        evidence_budget=evidence_budget,
        fallback_model=fallback_model,
        lite_model=lite_model,
    )
    shared_breaker = (
        RedisSharedCircuitBreaker(
            resolved.redis_url,
            prefix=f"{resolved.redis_key_prefix}:breaker",
            failure_threshold=resolved.tool_failure_threshold,
            recovery_seconds=resolved.tool_recovery_seconds,
        )
        if resolved.redis_enabled and resolved.breaker_shared
        else None
    )
    tool_resilience = ToolResilienceMiddleware(
        ToolResilienceConfig(
            default_timeout_seconds=resolved.tool_timeout_seconds,
            failure_threshold=resolved.tool_failure_threshold,
            recovery_seconds=resolved.tool_recovery_seconds,
            repeated_call_limit=resolved.tool_repeated_call_limit,
        ),
        shared_breaker=shared_breaker,
    )
    tool_harness = ToolHarnessMiddleware()
    observability = LangfuseCallbacks(enabled=resolved.langfuse_enabled)
    checkpointer = (
        LazyAsyncSqliteSaver(resolved.checkpoint_database_path)
        if resolved.checkpoint_backend == "sqlite"
        else None
    )
    runtime = MainAgent(
        model=model,
        tools=tools,
        main_only_tools=(*task_tools, *preference_tools, *order_tools),
        task_service=task_service,
        compressor=compressor,
        governance_config=resolved.governance,
        sub_agent_max_concurrency=resolved.sub_agent_max_concurrency,
        shared_middleware=(
            model_gateway,
            tool_harness,
            tool_resilience,
            preference_middleware,
        ),
        observability=observability,
        checkpointer=checkpointer,
    )
    event_bus: InMemoryTradeEventBus | RedisTradeEventBus
    task_queue: RedisStreamTaskQueue | None
    if resolved.redis_enabled:
        event_bus = RedisTradeEventBus(
            resolved.redis_url,
            channel_prefix=f"{resolved.redis_key_prefix}:events",
        )
        task_queue = RedisStreamTaskQueue(
            resolved.redis_url,
            stream=f"{resolved.redis_key_prefix}:agent:jobs",
            group=f"{resolved.redis_key_prefix}-workers",
            dead_letter_stream=f"{resolved.redis_key_prefix}:agent:jobs:dead",
            max_attempts=resolved.queue_max_attempts,
            large_stream=f"{resolved.redis_key_prefix}:agent:jobs:large",
            large_request_turns=resolved.queue_large_request_turns,
        )
    else:
        event_bus = InMemoryTradeEventBus()
        task_queue = None
    semantic_cache = (
        RedisSemanticResponseCache(
            resolved.redis_url,
            item_encoder,
            preference_profiles,
            namespace=(
                f"{resolved.redis_key_prefix}:semantic:"
                f"{resolved.llm_model_name}:"
                f"{hashlib.sha256(MAIN_SYSTEM_PROMPT.encode()).hexdigest()[:12]}"
            ),
            threshold=resolved.semantic_cache_threshold,
            ttl_seconds=resolved.semantic_cache_ttl_seconds,
        )
        if resolved.semantic_cache_enabled
        else None
    )
    return ApplicationContainer(
        settings=resolved,
        run_agent=RunAgent(
            runtime,
            event_bus,
            response_cache=semantic_cache,
            token_budget_total=resolved.token_budget_total,
            drift_detector=(
                DriftDetector(check_interval=resolved.drift_check_interval)
                if resolved.drift_detect_enabled
                else None
            ),
        ),
        runtime=runtime,
        event_bus=event_bus,
        orders=order_service,
        task_queue=task_queue,
        opensearch_probe=(
            OpenSearchReadinessProbe(resolved)
            if resolved.category_retriever_backend == "opensearch"
            else None
        ),
        model_gateway=model_gateway,
    )


def build_category_ingestor(
    settings: Settings | None = None,
) -> CategoryKnowledgeIngestor:
    """为离线脚本装配一次性结构化模型和本地卡片存储。"""

    resolved = settings or Settings.from_env()
    extractor = StructuredLLMCategoryExtractor(
        create_category_structuring_model(resolved)
    )
    publisher = (
        create_opensearch_category_repository(resolved)
        if resolved.category_retriever_backend == "opensearch"
        else None
    )
    return CategoryKnowledgeIngestor(
        extractor=extractor,
        store=JsonlCategoryCardStore(resolved.category_card_store),
        manifest_path=resolved.category_ingestion_manifest,
        publisher=publisher,
        max_concurrency=resolved.category_ingestion_max_concurrency,
        progress=lambda document, status: print(
            f"[category-ingestion] {status}: {document}",
            flush=True,
        ),
    )


def build_category_retriever(
    settings: Settings | None = None,
) -> CategoryKnowledgeRetriever:
    """为离线评测装配与线上 CategoryInsight 完全一致的检索实现。"""

    resolved = settings or Settings.from_env()
    return create_category_knowledge_retriever(resolved)


def build_item_search_service(
    settings: Settings | None = None,
    *,
    profile_source: UserProfileSource | None = None,
    indexes: IndexRegistry | None = None,
    vector_cache: VectorCache | None = None,
    encoder: EmbeddingEncoder | None = None,
    retrieval_mode: ItemRetrievalMode | None = None,
) -> ItemSearchService:
    """为在线工具和离线评测装配同一套商品搜索服务。"""

    resolved = settings or Settings.from_env()
    return create_item_search_service(
        resolved.item_index_root,
        use_fp16=resolved.retrieval_use_fp16,
        device=resolved.retrieval_device,
        embedding_batch_size=resolved.retrieval_embedding_batch_size,
        reranker_batch_size=resolved.retrieval_reranker_batch_size,
        profile_source=profile_source,
        indexes=indexes,
        vector_cache=vector_cache,
        encoder=encoder,
        retrieval_mode=retrieval_mode,
    )


def build_item_index_builder(
    settings: Settings | None = None,
) -> ItemIndexBuilder:
    """为离线索引脚本装配与线上一致的 BGE-M3 编码器。"""

    resolved = settings or Settings.from_env()
    return create_item_index_builder(
        use_fp16=resolved.retrieval_use_fp16,
        device=resolved.retrieval_device,
        embedding_batch_size=resolved.retrieval_embedding_batch_size,
    )
