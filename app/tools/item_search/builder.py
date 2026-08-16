"""Offline item-tower index construction."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .config import ItemSearchConfig
from .faiss_index import FaissItemIndex, validate_index_id
from .ports import EmbeddingEncoder
from .schemas import Product


class ItemIndexBuilder:
    """Encode products and persist one fork-addressable FAISS domain index."""

    def __init__(
        self,
        encoder: EmbeddingEncoder,
        *,
        config: ItemSearchConfig | None = None,
    ) -> None:
        self._encoder = encoder
        self._config = config or ItemSearchConfig()
        if self._encoder.dimension != self._config.embedding_dimension:
            raise ValueError("encoder dimension does not match ItemSearch configuration")

    def build(
        self,
        *,
        index_id: str,
        products: Sequence[Product],
        output_root: Path,
    ) -> Path:
        """Build a new domain index without overwriting existing artifacts."""

        validate_index_id(index_id)
        target = output_root.resolve() / index_id
        if target.exists() and (not target.is_dir() or any(target.iterdir())):
            raise FileExistsError(f"index domain already exists: {target}")

        texts = [product.to_search_text() for product in products]
        vectors = np.asarray(
            self._encoder.embed_documents(texts),
            dtype=np.float32,
        )
        index = FaissItemIndex.build(
            index_id=index_id,
            products=products,
            vectors=vectors,
            embedding_model=self._config.embedding_model,
            config=self._config,
        )
        index.save(target)
        return target
