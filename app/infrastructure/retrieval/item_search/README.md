# ItemSearch architecture

ItemSearch only performs vector retrieval inside a domain already selected by a
forked child Agent. It intentionally has no platform, category, price, or other
metadata-filter API.

## Online path

1. The child Agent passes `index_id`, `query`, and an optional `user_id`.
2. BGE-M3 encodes the query; the user tower supplies a vector in the same space.
3. The normalized vectors are fused with a configurable weighted sum.
4. FAISS HNSW with inner product recalls up to 100 products from that index.
5. BGE reranker v2 M3 scores all recalled query-product pairs.
6. The service returns the reranked Top-K, defaulting to 10.

## Offline path

`ItemIndexBuilder` turns stable product text into normalized BGE-M3 vectors and
writes one directory per retrieval domain:

```text
data/indexes/<index_id>/
├── index.faiss
├── manifest.json
└── products.json
```

The manifest pins the embedding model and dimension so an incompatible online
encoder cannot silently query an old index.

## Model loading

The BGE and FAISS adapters import their optional dependencies lazily. Importing
the package, running unit tests with fakes, or constructing the service does not
download a model. A model is loaded only on the first real encode or rerank call.
