"""FAISS HNSW+IP index and fork-selected index-domain registry."""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np

from app.application.catalog.config import ItemSearchConfig
from app.application.catalog.models import FloatVector, RecallHit
from app.application.catalog.ports import ItemVectorIndex
from app.domain.catalog import Product

from .fusion import normalize_matrix, normalize_vector
from .lexical import keyword_2gram_score

_INDEX_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def validate_index_id(index_id: str) -> str:
    """Validate a logical retrieval-domain name before resolving filesystem paths."""

    if _INDEX_ID_PATTERN.fullmatch(index_id) is None:
        raise ValueError(
            "index_id must contain only letters, numbers, dot, underscore, or hyphen"
        )
    return index_id


def _import_faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "FAISS support is not installed; run `uv sync --extra search`"
        ) from exc
    return faiss


@dataclass(frozen=True, slots=True)
class IndexManifest:
    """Compatibility metadata stored with every domain index."""

    schema_version: int
    index_id: str
    embedding_model: str
    embedding_dimension: int
    item_count: int
    index_type: str = "IndexHNSWFlat"
    metric: str = "inner_product"
    normalized: bool = True


class FaissItemIndex:
    """One metadata-filter-free FAISS index for a child-Agent retrieval domain."""

    def __init__(
        self,
        *,
        index: Any,
        manifest: IndexManifest,
        products: Mapping[str, Product],
        faiss_ids: Mapping[int, str],
    ) -> None:
        self._index = index
        self._manifest = manifest
        self._products = dict(products)
        self._faiss_ids = dict(faiss_ids)

    @property
    def index_id(self) -> str:
        return self._manifest.index_id

    @property
    def dimension(self) -> int:
        return self._manifest.embedding_dimension

    @property
    def size(self) -> int:
        return self._manifest.item_count

    @property
    def manifest(self) -> IndexManifest:
        return self._manifest

    @classmethod
    def build(
        cls,
        *,
        index_id: str,
        products: Sequence[Product],
        vectors: FloatVector,
        embedding_model: str,
        config: ItemSearchConfig,
    ) -> "FaissItemIndex":
        """Build a normalized HNSW inner-product index for one retrieval domain."""

        validate_index_id(index_id)
        if not products:
            raise ValueError("cannot build an index without products")
        item_ids = [product.item_id for product in products]
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("product item_id values must be unique")

        matrix = normalize_matrix(vectors)
        expected_shape = (len(products), config.embedding_dimension)
        if matrix.shape != expected_shape:
            raise ValueError(f"item vector shape must be {expected_shape}")

        faiss = _import_faiss()
        base_index = faiss.IndexHNSWFlat(
            config.embedding_dimension,
            config.hnsw_m,
            faiss.METRIC_INNER_PRODUCT,
        )
        base_index.hnsw.efConstruction = config.hnsw_ef_construction
        base_index.hnsw.efSearch = max(config.hnsw_ef_search, config.recall_k)
        index = faiss.IndexIDMap2(base_index)
        numeric_ids = np.arange(len(products), dtype=np.int64)
        index.add_with_ids(matrix, numeric_ids)

        manifest = IndexManifest(
            schema_version=2,
            index_id=index_id,
            embedding_model=embedding_model,
            embedding_dimension=config.embedding_dimension,
            item_count=len(products),
        )
        return cls(
            index=index,
            manifest=manifest,
            products={product.item_id: product for product in products},
            faiss_ids={position: item_id for position, item_id in enumerate(item_ids)},
        )

    def search(self, vector: FloatVector, top_k: int) -> Sequence[RecallHit]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if self.size == 0:
            return []

        request = normalize_vector(vector)
        if request.shape != (self.dimension,):
            raise ValueError(
                f"request vector dimension must be {self.dimension}, "
                f"got {request.shape[0]}"
            )
        scores, numeric_ids = self._index.search(
            request.reshape(1, -1),
            min(top_k, self.size),
        )
        hits: list[RecallHit] = []
        for numeric_id, score in zip(numeric_ids[0], scores[0], strict=True):
            if int(numeric_id) < 0:
                continue
            item_id = self._faiss_ids.get(int(numeric_id))
            if item_id is not None:
                hits.append(RecallHit(item_id=item_id, score=float(score)))
        return hits

    def keyword_search(self, query: str, top_k: int) -> Sequence[RecallHit]:
        """在同一索引商品快照上执行无模型的关键词二元组召回。"""

        if top_k < 1:
            raise ValueError("top_k must be positive")
        scored = [
            RecallHit(
                item_id=product.item_id,
                score=keyword_2gram_score(query, product.to_search_text()),
            )
            for product in self._products.values()
        ]
        matched = [hit for hit in scored if hit.score > 0]
        matched.sort(key=lambda hit: (hit.score, hit.item_id), reverse=True)
        return tuple(matched[:top_k])

    def get_product(self, item_id: str) -> Product:
        try:
            return self._products[item_id]
        except KeyError as exc:
            raise KeyError(f"item {item_id!r} is missing from index metadata") from exc

    def save(self, directory: Path) -> None:
        """Persist the FAISS index, product records, and compatibility manifest."""

        faiss = _import_faiss()
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / "index.faiss"))
        (directory / "manifest.json").write_text(
            json.dumps(asdict(self._manifest), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        product_rows = [
            {"faiss_id": faiss_id, "product": self._products[item_id].to_dict()}
            for faiss_id, item_id in sorted(self._faiss_ids.items())
        ]
        (directory / "products.json").write_text(
            json.dumps(product_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> "FaissItemIndex":
        """Load one previously built retrieval-domain index."""

        faiss = _import_faiss()
        manifest_data = json.loads(
            (directory / "manifest.json").read_text(encoding="utf-8")
        )
        manifest = IndexManifest(**manifest_data)
        if manifest.schema_version not in {1, 2}:
            raise ValueError("unsupported item index schema version")
        if manifest.index_type != "IndexHNSWFlat":
            raise ValueError("unsupported item index type")
        if manifest.metric != "inner_product" or not manifest.normalized:
            raise ValueError("item index must use normalized inner-product vectors")
        rows = json.loads((directory / "products.json").read_text(encoding="utf-8"))
        products: dict[str, Product] = {}
        faiss_ids: dict[int, str] = {}
        for row in rows:
            product = Product.from_dict(row["product"])
            products[product.item_id] = product
            faiss_ids[int(row["faiss_id"])] = product.item_id

        if len(products) != manifest.item_count:
            raise ValueError(
                "index manifest item_count does not match product metadata"
            )
        index = faiss.read_index(str(directory / "index.faiss"))
        if index.d != manifest.embedding_dimension:
            raise ValueError("FAISS index dimension does not match its manifest")
        if index.ntotal != manifest.item_count:
            raise ValueError("FAISS index item count does not match its manifest")
        return cls(
            index=index,
            manifest=manifest,
            products=products,
            faiss_ids=faiss_ids,
        )


class InMemoryIndexRegistry:
    """Registry used by tests and applications that construct indexes themselves."""

    def __init__(self, indexes: Mapping[str, ItemVectorIndex]) -> None:
        self._indexes = dict(indexes)

    def get(self, index_id: str) -> ItemVectorIndex:
        validate_index_id(index_id)
        try:
            return self._indexes[index_id]
        except KeyError as exc:
            raise KeyError(f"unknown item index domain: {index_id}") from exc


class FaissDirectoryIndexRegistry:
    """Lazily load and cache the FAISS domain selected by a child Agent."""

    def __init__(
        self,
        root: Path,
        *,
        expected_embedding_model: str,
        expected_dimension: int,
    ) -> None:
        self._root = root.resolve()
        self._expected_embedding_model = expected_embedding_model
        self._expected_dimension = expected_dimension
        self._cache: dict[str, FaissItemIndex] = {}
        self._lock = RLock()

    def _directory_for(self, index_id: str) -> Path:
        validate_index_id(index_id)
        directory = (self._root / index_id).resolve()
        if self._root not in directory.parents:
            raise ValueError("index domain resolves outside the configured root")
        return directory

    def get(self, index_id: str) -> FaissItemIndex:
        with self._lock:
            cached = self._cache.get(index_id)
            if cached is not None:
                return cached

            index = FaissItemIndex.load(self._directory_for(index_id))
            manifest = index.manifest
            if manifest.embedding_model != self._expected_embedding_model:
                raise ValueError(
                    "index embedding model does not match the online encoder"
                )
            if manifest.embedding_dimension != self._expected_dimension:
                raise ValueError("index dimension does not match the online encoder")
            self._cache[index_id] = index
            return index
