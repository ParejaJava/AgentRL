"""时效性网页检索用例端口。"""

from typing import Protocol


class WebSearch(Protocol):
    """查询跨境政策、平台规则等易变化的公开网页信息。"""

    async def search(self, query: str, max_results: int) -> list[dict[str, object]]: ...
