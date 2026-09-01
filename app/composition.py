"""API 与 Worker 共用的唯一依赖装配入口。"""

from dataclasses import dataclass

from app.application.agents import RunAgent
from app.application.catalog import CategoryInsightConfig
from app.application.catalog.category_insight_ports import (
    CategoryKnowledgeRetriever,
)
from app.application.catalog.search_catalog import ItemSearchService
from app.application.tasking import TaskBoardService
from app.infrastructure.context_governance.compressor import StructuredLLMCompressor
from app.infrastructure.eventbus import InMemoryTradeEventBus
from app.infrastructure.langchain import MainAgent
from app.infrastructure.langchain.tools import (
    create_category_insight_tool,
    create_item_search_tool,
    create_task_tools,
)
from app.infrastructure.llm import (
    create_category_structuring_model,
    create_chat_model,
    create_compression_model,
)
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
    create_item_index_builder,
    create_item_search_service,
)
from app.infrastructure.settings import Settings
from app.infrastructure.tasking import SQLiteTaskBoardRepository


@dataclass(frozen=True, slots=True)
class ApplicationContainer:
    """保存已经完成装配的应用用例和基础设施 Driver。"""

    settings: Settings
    run_agent: RunAgent
    runtime: MainAgent
    event_bus: InMemoryTradeEventBus


def build_container(settings: Settings | None = None) -> ApplicationContainer:
    """创建平台所有具体依赖；其他模块不得自行组装实现。"""

    resolved = settings or Settings.from_env()
    model = create_chat_model(resolved)
    compressor = (
        StructuredLLMCompressor(create_compression_model(resolved))
        if resolved.context_llm_enabled
        else None
    )
    item_search_service = build_item_search_service(resolved)
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
    task_service = TaskBoardService(
        SQLiteTaskBoardRepository(
            resolved.governance.session_root / "task_boards.db"
        )
    )
    task_tools = create_task_tools(task_service)
    runtime = MainAgent(
        model=model,
        tools=tools,
        main_only_tools=task_tools,
        task_service=task_service,
        compressor=compressor,
        governance_config=resolved.governance,
        sub_agent_max_concurrency=resolved.sub_agent_max_concurrency,
    )
    event_bus = InMemoryTradeEventBus()
    return ApplicationContainer(
        settings=resolved,
        run_agent=RunAgent(runtime, event_bus),
        runtime=runtime,
        event_bus=event_bus,
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
    )


def build_category_retriever(
    settings: Settings | None = None,
) -> CategoryKnowledgeRetriever:
    """为离线评测装配与线上 CategoryInsight 完全一致的检索实现。"""

    resolved = settings or Settings.from_env()
    return create_category_knowledge_retriever(resolved)


def build_item_search_service(
    settings: Settings | None = None,
) -> ItemSearchService:
    """为在线工具和离线评测装配同一套商品搜索服务。"""

    resolved = settings or Settings.from_env()
    return create_item_search_service(
        resolved.item_index_root,
        use_fp16=resolved.retrieval_use_fp16,
        device=resolved.retrieval_device,
        embedding_batch_size=resolved.retrieval_embedding_batch_size,
        reranker_batch_size=resolved.retrieval_reranker_batch_size,
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
