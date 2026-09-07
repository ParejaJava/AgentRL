"""User-tower providers kept independent from ANN metadata filtering."""

from collections.abc import Mapping
from typing import Protocol

from app.application.catalog.models import UserSignal
from app.application.catalog.ports import EmbeddingEncoder
from app.application.memory import (
    PreferenceSelector,
    SyncPreferenceReader,
    render_preference_profile,
)


class UserProfileSource(Protocol):
    """Fetch the text summary used to build a user vector."""

    def get_profile(self, user_id: str, index_id: str) -> str | None: ...


class InMemoryUserProfileSource:
    """Small profile source for development and deterministic tests."""

    def __init__(self, profiles: Mapping[tuple[str, str], str]) -> None:
        self._profiles = dict(profiles)

    def get_profile(self, user_id: str, index_id: str) -> str | None:
        return self._profiles.get((user_id, index_id))


class StoredPreferenceProfileSource:
    """把持久化偏好投影为商品 User Tower 使用的画像文本。"""

    def __init__(
        self,
        reader: SyncPreferenceReader,
        selector: PreferenceSelector,
    ) -> None:
        self._reader = reader
        self._selector = selector

    def get_profile(self, user_id: str, index_id: str) -> str | None:
        """读取买家画像；index_id 保留为未来按检索域分区的扩展点。"""

        del index_id
        selected = self._selector.select(self._reader.list_by_buyer_sync(user_id))
        profile = render_preference_profile(selected)
        return profile or None


class EmbeddingUserSignalProvider:
    """Encode a scoped user-profile summary in the shared BGE vector space."""

    def __init__(
        self,
        encoder: EmbeddingEncoder,
        profile_source: UserProfileSource,
    ) -> None:
        self._encoder = encoder
        self._profile_source = profile_source

    def get(self, user_id: str, index_id: str) -> UserSignal | None:
        profile = self._profile_source.get_profile(user_id, index_id)
        if profile is None or not profile.strip():
            return None
        vector = self._encoder.embed_queries([profile])[0]
        return UserSignal(vector=vector, summary=profile)


class InMemoryUserSignalProvider:
    """Serve precomputed user vectors keyed by user and retrieval domain."""

    def __init__(self, signals: Mapping[tuple[str, str], UserSignal]) -> None:
        self._signals = dict(signals)

    def get(self, user_id: str, index_id: str) -> UserSignal | None:
        return self._signals.get((user_id, index_id))
