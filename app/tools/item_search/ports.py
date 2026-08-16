"""Dependency boundaries for models, user signals, and scoped indexes."""

from collections.abc import Sequence
from typing import Protocol

from .schemas import FloatVector, Product, RecallHit, UserSignal


class EmbeddingEncoder(Protocol):
    """Dense encoder shared by query, user, and item towers."""

    @property
    def dimension(self) -> int: ...

    def embed_queries(self, texts: Sequence[str]) -> FloatVector: ...

    def embed_documents(self, texts: Sequence[str]) -> FloatVector: ...


class Reranker(Protocol):
    """Cross-encoder scoring query-document pairs."""

    def score(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]: ...


class UserSignalProvider(Protocol):
    """Provide a user-tower signal within the selected retrieval domain."""

    def get(self, user_id: str, index_id: str) -> UserSignal | None: ...


class ItemVectorIndex(Protocol):
    """A vector index containing only one fork-selected retrieval domain."""

    @property
    def index_id(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    @property
    def size(self) -> int: ...

    def search(self, vector: FloatVector, top_k: int) -> Sequence[RecallHit]: ...

    def get_product(self, item_id: str) -> Product: ...


class IndexRegistry(Protocol):
    """Resolve the index domain selected by a child Agent."""

    def get(self, index_id: str) -> ItemVectorIndex: ...


class NullUserSignalProvider:
    """Disable personalization while preserving the three-tower interface."""

    def get(self, user_id: str, index_id: str) -> None:
        return None

