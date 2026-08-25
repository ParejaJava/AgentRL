"""纯 Python 商品领域对象。"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TypeAlias

JsonScalar: TypeAlias = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class Product:
    """可被多个检索基础设施共享的商品实体。"""

    item_id: str
    title: str
    category: str = ""
    brand: str = ""
    description: str = ""
    platform: str = ""
    attributes: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, JsonScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.item_id.strip():
            raise ValueError("item_id 不能为空")
        if not self.title.strip():
            raise ValueError("title 不能为空")

    def to_search_text(self) -> str:
        """生成稳定的商品语义检索文本。"""

        fields = [
            ("商品名", self.title),
            ("类目", self.category),
            ("品牌", self.brand),
            ("平台", self.platform),
            ("描述", self.description),
        ]
        lines = [f"{label}: {value}" for label, value in fields if value]
        if self.attributes:
            attributes = "；".join(
                f"{key}={value}" for key, value in sorted(self.attributes.items())
            )
            lines.append(f"属性: {attributes}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """输出供接口适配器序列化的稳定字典。"""

        return {
            "item_id": self.item_id,
            "title": self.title,
            "category": self.category,
            "brand": self.brand,
            "description": self.description,
            "platform": self.platform,
            "attributes": dict(self.attributes),
            "metadata": dict(self.metadata),
        }
