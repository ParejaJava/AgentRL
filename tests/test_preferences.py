"""长期偏好、动态注入和商品 User Tower 接线测试。"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np

from app.application.memory import PreferenceSelector
from app.application.runtime import AgentExecutionContext, ShoppingContextSnapshot
from app.domain.buyer import BuyerPreference, PreferenceKind
from app.infrastructure.context import reset_context, set_context
from app.infrastructure.langchain.preference_middleware import (
    PreferenceMemoryMiddleware,
)
from app.infrastructure.langchain.tools import create_preference_tools
from app.infrastructure.memory import SQLitePreferenceStore
from app.infrastructure.retrieval.item_search.user_tower import (
    EmbeddingUserSignalProvider,
    StoredPreferenceProfileSource,
)


class _Encoder:
    dimension = 2

    def embed_queries(self, texts: list[str]) -> list[np.ndarray]:
        return [np.asarray([float(len(text)), 1.0]) for text in texts]

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        return self.embed_queries(texts)


def _preference(
    statement: str,
    kind: PreferenceKind = PreferenceKind.LIKE,
    *,
    offset: int = 0,
) -> BuyerPreference:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=offset)
    return BuyerPreference(
        buyer_id="buyer-1",
        kind=kind,
        statement=statement,
        created_at=timestamp,
        updated_at=timestamp,
    )


def test_selector_keeps_all_dislikes_and_limits_likes() -> None:
    selector = PreferenceSelector(like_limit=2)
    preferences = (
        _preference("喜欢红色", offset=1),
        _preference("不要塑料", PreferenceKind.DISLIKE, offset=2),
        _preference("喜欢轻量", offset=3),
        _preference("不要皮革", PreferenceKind.DISLIKE, offset=4),
        _preference("偏好小众品牌", offset=5),
    )

    selected = selector.select(preferences)

    assert [item.statement for item in selected] == [
        "不要塑料",
        "不要皮革",
        "喜欢轻量",
        "偏好小众品牌",
    ]


def test_sqlite_store_deduplicates_deletes_exactly_and_prunes_likes(
    tmp_path,
) -> None:
    store = SQLitePreferenceStore(tmp_path / "preferences.db", max_likes_per_buyer=2)

    async def exercise() -> None:
        await store.save(_preference("喜欢红色", offset=1))
        await store.save(_preference("喜欢轻量", offset=2))
        await store.save(_preference("不要塑料", PreferenceKind.DISLIKE, offset=3))
        await store.save(_preference("偏好小众品牌", offset=4))
        # 同一原文更新为 dislike，不创建重复行。
        await store.save(
            _preference("偏好小众品牌", PreferenceKind.DISLIKE, offset=5)
        )

        current = await store.list_by_buyer("buyer-1")
        assert [(item.kind.value, item.statement) for item in current] == [
            ("like", "喜欢轻量"),
            ("dislike", "不要塑料"),
            ("dislike", "偏好小众品牌"),
        ]
        assert not await store.delete_exact("buyer-1", "不要塑料材质")
        assert await store.delete_exact("buyer-1", "不要塑料")
        assert [
            item.statement for item in await store.list_by_buyer("buyer-1")
        ] == ["喜欢轻量", "偏好小众品牌"]

    asyncio.run(exercise())


def test_preference_tools_take_buyer_from_execution_context(tmp_path) -> None:
    store = SQLitePreferenceStore(tmp_path / "preferences.db")
    remember, forget = create_preference_tools(store)
    token = set_context(
        AgentExecutionContext(
            thread_id="thread-1",
            shopping=ShoppingContextSnapshot("shopping-1", "trusted-buyer"),
        )
    )

    async def exercise() -> None:
        saved = json.loads(
            await remember.ainvoke(
                {"kind": "dislike", "statement": "不要塑料材质"}
            )
        )
        assert saved["preference"]["buyer_id"] == "trusted-buyer"
        missed = json.loads(
            await forget.ainvoke({"statement": "不要塑料"})
        )
        assert missed["deleted"] is False
        removed = json.loads(
            await forget.ainvoke({"statement": "不要塑料材质"})
        )
        assert removed["deleted"] is True

    try:
        asyncio.run(exercise())
    finally:
        reset_context(token)


def test_middleware_projects_preferences_into_dynamic_working_memory(
    tmp_path,
) -> None:
    store = SQLitePreferenceStore(tmp_path / "preferences.db")
    asyncio.run(store.save(_preference("喜欢轻量")))
    middleware = PreferenceMemoryMiddleware(store, PreferenceSelector(like_limit=8))
    runtime = SimpleNamespace(
        context=AgentExecutionContext(
            thread_id="thread-1",
            shopping=ShoppingContextSnapshot("shopping-1", "buyer-1"),
        )
    )

    update = asyncio.run(
        middleware.abefore_model(
            {"messages": [], "working_memory": {"fact-1": {"kind": "fact"}}},
            runtime,
        )
    )

    assert update is not None
    assert update["working_memory"]["fact-1"] == {"kind": "fact"}
    preference_items = [
        value
        for key, value in update["working_memory"].items()
        if key.startswith("buyer-preference:")
    ]
    assert len(preference_items) == 1
    assert preference_items[0]["content"] == "[like] 喜欢轻量"


def test_stored_profile_drives_user_tower_vector(tmp_path) -> None:
    store = SQLitePreferenceStore(tmp_path / "preferences.db")
    asyncio.run(store.save(_preference("喜欢轻量")))
    source = StoredPreferenceProfileSource(
        store,
        PreferenceSelector(like_limit=8),
    )
    provider = EmbeddingUserSignalProvider(_Encoder(), source)

    signal = provider.get("buyer-1", "evaluation-products")

    assert signal is not None
    assert signal.summary == "[like] 喜欢轻量"
    assert list(signal.vector) == [11.0, 1.0]
