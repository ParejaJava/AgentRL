"""商品检索用例使用的出站端口。"""

from collections.abc import Sequence
from typing import Protocol

from app.domain.catalog import Product

from .models import FloatVector, RecallHit, UserSignal


class EmbeddingEncoder(Protocol):
    """把查询或文档编码到同一向量空间。"""

    @property
    def dimension(self) -> int: ...

    def embed_queries(self, texts: Sequence[str]) -> Sequence[FloatVector]: ...

    def embed_documents(self, texts: Sequence[str]) -> Sequence[FloatVector]: ...


class Reranker(Protocol):
    """为查询与商品文本对计算相关性分数。"""

    def score(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]: ...


class UserSignalProvider(Protocol):
    """读取指定用户在检索域内的偏好信号。"""

    def get(self, user_id: str, index_id: str) -> UserSignal | None: ...


class ItemVectorIndex(Protocol):
    """访问一个已经由 Orchestrator 选定的商品索引域。"""

    @property
    def index_id(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    @property
    def size(self) -> int: ...

    def search(self, vector: FloatVector, top_k: int) -> Sequence[RecallHit]: ...

    def keyword_search(self, query: str, top_k: int) -> Sequence[RecallHit]: ...

    def get_product(self, item_id: str) -> Product: ...


class IndexRegistry(Protocol):
    """解析子 Agent 选择的商品索引域。"""

    def get(self, index_id: str) -> ItemVectorIndex: ...


class NullUserSignalProvider:
    """在未配置买家画像时关闭个性化。"""

    def get(self, user_id: str, index_id: str) -> None:
        del user_id, index_id
