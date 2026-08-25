"""根据配置选择 provider-specific Prompt Cache 适配器。"""

from __future__ import annotations

from .base import CacheProviderAdapter, ImplicitCacheAdapter
from .qwen import QwenExplicitCacheAdapter


def create_cache_adapter(
    *,
    provider: str,
    explicit_cache: bool,
    model_name: str,
) -> CacheProviderAdapter:
    """仅在明确启用且识别为 Qwen 时添加显式 cache marker。"""

    normalized = provider.lower()
    is_qwen = normalized == "qwen" or (
        normalized == "auto" and "qwen" in model_name.lower()
    )
    if explicit_cache and is_qwen:
        return QwenExplicitCacheAdapter()
    return ImplicitCacheAdapter()


__all__ = [
    "CacheProviderAdapter",
    "ImplicitCacheAdapter",
    "QwenExplicitCacheAdapter",
    "create_cache_adapter",
]
