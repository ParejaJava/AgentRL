"""验证召回模型的设备与显存配置能够传递到基础设施适配器。"""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import numpy as np
import pytest

from app.infrastructure.retrieval.item_search import bge as bge_module
from app.infrastructure.retrieval.item_search.bge import (
    BGEEmbeddingEncoder,
    BGEReranker,
)
from app.infrastructure.retrieval.item_search.factory import (
    create_item_index_builder,
    create_item_search_service,
)
from app.infrastructure.settings import Settings


def test_bge_adapters_reject_invalid_batch_size() -> None:
    """批大小必须在加载大模型之前完成校验。"""

    with pytest.raises(ValueError, match="embedding batch_size"):
        BGEEmbeddingEncoder(batch_size=0)
    with pytest.raises(ValueError, match="reranker batch_size"):
        BGEReranker(batch_size=0)


class _ConcurrentProbe:
    """记录同一个模型实例是否被多个线程同时调用。"""

    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.lock = Lock()

    def _enter(self) -> None:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

    def _exit(self) -> None:
        with self.lock:
            self.active -= 1


class _FakeEmbeddingModel(_ConcurrentProbe):
    def encode(self, texts: list[str], **_kwargs: object) -> dict[str, np.ndarray]:
        self._enter()
        try:
            time.sleep(0.03)
            return {"dense_vecs": np.ones((len(texts), 2), dtype=np.float32)}
        finally:
            self._exit()


class _FakeRerankerModel(_ConcurrentProbe):
    def compute_score(
        self,
        pairs: list[list[str]],
        **_kwargs: object,
    ) -> list[float]:
        self._enter()
        try:
            time.sleep(0.03)
            return [0.5] * len(pairs)
        finally:
            self._exit()


def test_bge_model_instances_serialize_concurrent_inference() -> None:
    """并行子 Agent 共享 BGE 实例时不得并发借用 tokenizer/GPU 模型。"""

    encoder = BGEEmbeddingEncoder(dimension=2)
    embedding_model = _FakeEmbeddingModel()
    encoder._model = embedding_model
    reranker = BGEReranker()
    reranker_model = _FakeRerankerModel()
    reranker._model = reranker_model

    with ThreadPoolExecutor(max_workers=4) as executor:
        embedding_futures = [
            executor.submit(encoder.embed_queries, [f"query-{index}"])
            for index in range(4)
        ]
        reranker_futures = [
            executor.submit(reranker.score, [(f"query-{index}", "document")])
            for index in range(4)
        ]
        for future in (*embedding_futures, *reranker_futures):
            future.result()

    assert embedding_model.max_active == 1
    assert reranker_model.max_active == 1


def test_bge_same_configuration_reuses_process_model() -> None:
    """品类与商品检索使用同配置时只保留一份进程级模型实例。"""

    cache_key = (
        "reranker",
        "test-reranker",
        False,
        "cpu",
        8,
        512,
    )
    shared_model = _FakeRerankerModel()
    bge_module._MODEL_CACHE[cache_key] = shared_model
    try:
        first = BGEReranker("test-reranker", device="cpu")
        second = BGEReranker("test-reranker", device="cpu")

        assert first._get_model() is shared_model
        assert second._get_model() is shared_model
    finally:
        bge_module._MODEL_CACHE.pop(cache_key, None)


def test_item_search_factory_passes_gpu_memory_configuration() -> None:
    """商品检索与离线建索引共用相同的设备、精度和 embedding 批大小。"""

    service = create_item_search_service(
        # 工厂只保存索引根目录，不会在装配阶段访问磁盘。
        Path("data/indexes"),
        device="cuda:0",
        use_fp16=True,
        embedding_batch_size=16,
        reranker_batch_size=8,
    )
    builder = create_item_index_builder(
        device="cuda:0",
        use_fp16=True,
        embedding_batch_size=16,
    )

    assert service._encoder._device == "cuda:0"
    assert service._encoder._use_fp16 is True
    assert service._encoder._batch_size == 16
    assert service._reranker._device == "cuda:0"
    assert service._reranker._use_fp16 is True
    assert service._reranker._batch_size == 8
    assert builder._encoder._device == "cuda:0"
    assert builder._encoder._use_fp16 is True


def test_settings_load_retrieval_gpu_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """环境变量是 GPU 显存策略的唯一部署时配置入口。"""

    monkeypatch.setenv("RETRIEVAL_DEVICE", "cuda:0")
    monkeypatch.setenv("RETRIEVAL_USE_FP16", "true")
    monkeypatch.setenv("RETRIEVAL_EMBEDDING_BATCH_SIZE", "16")
    monkeypatch.setenv("RETRIEVAL_RERANKER_BATCH_SIZE", "8")

    settings = Settings.from_env()

    assert settings.retrieval_device == "cuda:0"
    assert settings.retrieval_use_fp16 is True
    assert settings.retrieval_embedding_batch_size == 16
    assert settings.retrieval_reranker_batch_size == 8
