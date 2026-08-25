"""Request-vector normalization and query/user fusion."""

import numpy as np

from app.application.catalog.models import FloatVector


def normalize_vector(vector: FloatVector) -> FloatVector:
    """Return a finite, L2-normalized float32 vector."""

    normalized = np.asarray(vector, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(normalized)):
        raise ValueError("vector contains non-finite values")
    norm = float(np.linalg.norm(normalized))
    if norm == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return np.ascontiguousarray(normalized / norm, dtype=np.float32)


def normalize_matrix(vectors: FloatVector) -> FloatVector:
    """L2-normalize a batch of vectors row by row."""

    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("vectors must be a two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("vectors contain non-finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("cannot normalize a zero vector")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def fuse_request_vector(
    query_vector: FloatVector,
    user_vector: FloatVector | None,
    *,
    query_weight: float,
) -> FloatVector:
    """Fuse query and user vectors in their shared BGE embedding space."""

    if not 0.0 <= query_weight <= 1.0:
        raise ValueError("query_weight must be between 0 and 1")

    query = normalize_vector(query_vector)
    if user_vector is None:
        return query

    user = normalize_vector(user_vector)
    if query.shape != user.shape:
        raise ValueError("query and user vectors must have the same dimension")

    fused = query_weight * query + (1.0 - query_weight) * user
    return normalize_vector(fused)
