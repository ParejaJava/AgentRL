"""增量摄取 knowledge/*.md，生成可长期复用的结构化品类卡片。"""

import asyncio
import json

from app.composition import build_category_ingestor
from app.infrastructure.settings import Settings


async def main() -> None:
    """只对新增或内容哈希变化的 Markdown 调用一次结构化 LLM。"""

    settings = Settings.from_env()
    ingestor = build_category_ingestor(settings)
    report = await ingestor.ingest(settings.category_knowledge_root)
    print(
        json.dumps(
            {
                "processed_documents": report.processed_documents,
                "skipped_documents": report.skipped_documents,
                "removed_documents": report.removed_documents,
                "card_count": report.card_count,
                "published_cards": report.published_cards,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
