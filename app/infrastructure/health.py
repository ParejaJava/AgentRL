"""外部基础设施的真实就绪探针。"""

from __future__ import annotations

from dataclasses import dataclass

from app.infrastructure.retrieval.category_insight.opensearch import (
    create_opensearch_client,
)
from app.infrastructure.settings import Settings


@dataclass(frozen=True, slots=True)
class OpenSearchReadiness:
    """OpenSearch 连接、知识索引和 Search Pipeline 的检查结果。"""

    reachable: bool
    index_ready: bool
    pipeline_ready: bool
    cluster_green: bool
    cluster_status: str | None = None
    error: str | None = None

    @property
    def ready(self) -> bool:
        """只有三项基础能力全部可用时才视为就绪。"""

        return (
            self.reachable
            and self.index_ready
            and self.pipeline_ready
            and self.cluster_green
        )

    def to_dict(self) -> dict[str, object]:
        """转换为不包含凭据的健康检查响应。"""

        return {
            "reachable": self.reachable,
            "index_ready": self.index_ready,
            "pipeline_ready": self.pipeline_ready,
            "cluster_green": self.cluster_green,
            "cluster_status": self.cluster_status,
            "error": self.error,
        }


class OpenSearchReadinessProbe:
    """不加载 Embedding/Reranker，仅检查 OpenSearch 在线检索资源。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def check(self) -> OpenSearchReadiness:
        """执行一次真实连接、索引和 Pipeline 检查。"""

        client = None
        try:
            client = create_opensearch_client(
                self._settings.opensearch_url,
                username=self._settings.opensearch_username,
                password=self._settings.opensearch_password,
                verify_certs=self._settings.opensearch_verify_certs,
                timeout_seconds=self._settings.opensearch_timeout_seconds,
            )
            reachable = bool(client.ping())
            if not reachable:
                return OpenSearchReadiness(False, False, False, False, error="ping failed")
            index_ready = bool(
                client.indices.exists(index=self._settings.opensearch_category_index)
            )
            errors: list[str] = []
            try:
                pipeline = client.transport.perform_request(
                    "GET",
                    f"/_search/pipeline/{self._settings.opensearch_category_pipeline}",
                )
                pipeline_ready = (
                    self._settings.opensearch_category_pipeline in pipeline
                )
            except Exception as exc:  # noqa: BLE001 - 每项就绪状态独立报告。
                pipeline_ready = False
                errors.append(f"pipeline:{type(exc).__name__}")
            cluster_status = None
            if index_ready:
                try:
                    health = client.cluster.health(
                        index=self._settings.opensearch_category_index,
                    )
                    cluster_status = str(health.get("status", "unknown")).lower()
                except Exception as exc:  # noqa: BLE001 - 每项就绪状态独立报告。
                    errors.append(f"cluster:{type(exc).__name__}")
            return OpenSearchReadiness(
                True,
                index_ready,
                pipeline_ready,
                cluster_status == "green",
                cluster_status=cluster_status,
                error=";".join(errors) or None,
            )
        except Exception as exc:  # noqa: BLE001 - 健康探针必须返回降级状态。
            return OpenSearchReadiness(
                False,
                False,
                False,
                False,
                error=type(exc).__name__,
            )
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
