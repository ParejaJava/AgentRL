"""API 与 Worker 共用的唯一依赖装配入口。"""

from dataclasses import dataclass

from app.application.agents import RunAgent
from app.infrastructure.context_governance.compressor import StructuredLLMCompressor
from app.infrastructure.langchain import MainAgent
from app.infrastructure.langchain.tools.product_search import create_item_search_tool
from app.infrastructure.llm import create_chat_model, create_compression_model
from app.infrastructure.retrieval.item_search.factory import create_item_search_service
from app.infrastructure.settings import Settings


@dataclass(frozen=True, slots=True)
class ApplicationContainer:
    """保存已经完成装配的应用用例和基础设施 Driver。"""

    settings: Settings
    run_agent: RunAgent
    runtime: MainAgent


def build_container(settings: Settings | None = None) -> ApplicationContainer:
    """创建平台所有具体依赖；其他模块不得自行组装实现。"""

    resolved = settings or Settings.from_env()
    model = create_chat_model(resolved)
    compressor = (
        StructuredLLMCompressor(create_compression_model(resolved))
        if resolved.context_llm_enabled
        else None
    )
    item_search_service = create_item_search_service(resolved.item_index_root)
    tools = [create_item_search_tool(item_search_service)]
    runtime = MainAgent(
        model=model,
        tools=tools,
        compressor=compressor,
        governance_config=resolved.governance,
        sub_agent_max_concurrency=resolved.sub_agent_max_concurrency,
    )
    return ApplicationContainer(
        settings=resolved,
        run_agent=RunAgent(runtime),
        runtime=runtime,
    )
