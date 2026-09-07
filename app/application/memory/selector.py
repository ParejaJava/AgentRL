"""控制每轮进入 Agent 与商品 User Tower 的长期偏好集合。"""

from __future__ import annotations

from collections.abc import Sequence

from app.domain.buyer import BuyerPreference, PreferenceKind


class PreferenceSelector:
    """全量保留负向约束，并只选取最近的正向偏好。"""

    def __init__(self, *, like_limit: int = 8) -> None:
        if like_limit < 0:
            raise ValueError("like_limit 不能小于 0")
        self._like_limit = like_limit

    def select(
        self,
        preferences: Sequence[BuyerPreference],
    ) -> tuple[BuyerPreference, ...]:
        """返回稳定顺序的偏好；dislike 不受正向偏好上限影响。"""

        dislikes = sorted(
            (item for item in preferences if item.kind is PreferenceKind.DISLIKE),
            key=lambda item: (item.created_at, item.preference_id),
        )
        likes = sorted(
            (item for item in preferences if item.kind is PreferenceKind.LIKE),
            key=lambda item: (item.updated_at, item.preference_id),
            reverse=True,
        )[: self._like_limit]
        likes.reverse()
        return (*dislikes, *likes)


def render_preference_profile(
    preferences: Sequence[BuyerPreference],
) -> str:
    """生成 Agent 和 User Tower 共用的可解释买家画像文本。"""

    return "\n".join(
        f"[{item.kind.value}] {item.statement}" for item in preferences
    )
