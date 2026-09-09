"""与业务运行包隔离的证据报告模型。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

EvidenceStatus = Literal["verified", "code_verified", "planned", "boundary"]


@dataclass(slots=True)
class EvidenceReport:
    """一个 Claim 对应的一次可复现实验报告。"""

    claim_id: str
    capability: str
    status: EvidenceStatus
    started_at: str
    duration_seconds: float
    command: str
    environment: dict[str, Any]
    dataset: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    slices: dict[str, Any] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    passed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """转换成稳定的 JSON 对象。"""

        return asdict(self)

