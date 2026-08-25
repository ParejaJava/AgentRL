"""不依赖 NumPy 的请求向量融合规则。"""

import math
from collections.abc import Sequence

from .models import FloatVector


def normalize_vector(vector: Sequence[float]) -> tuple[float, ...]:
    """校验并返回 L2 归一化向量。"""

    values = tuple(float(value) for value in vector)
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("vector must contain finite values")
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return tuple(value / norm for value in values)


def fuse_request_vector(
    query_vector: FloatVector,
    user_vector: FloatVector | None,
    *,
    query_weight: float,
) -> tuple[float, ...]:
    """在共享向量空间融合查询和买家偏好。"""

    if not 0.0 <= query_weight <= 1.0:
        raise ValueError("query_weight must be between 0 and 1")
    query = normalize_vector(query_vector)
    if user_vector is None:
        return query
    user = normalize_vector(user_vector)
    if len(query) != len(user):
        raise ValueError("query and user vectors must have the same dimension")
    fused = tuple(
        query_weight * query_value + (1.0 - query_weight) * user_value
        for query_value, user_value in zip(query, user, strict=True)
    )
    return normalize_vector(fused)
