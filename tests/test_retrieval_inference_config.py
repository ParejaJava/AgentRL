"""验证召回模型的设备与显存配置能够传递到基础设施适配器。"""

from pathlib import Path

import pytest

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
