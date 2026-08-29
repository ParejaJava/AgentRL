"""纯 Python 商品领域对象。"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TypeAlias

from .money import Money

JsonScalar: TypeAlias = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class Sku:
    """商品的可购买 SKU；搜索预算使用有库存 SKU 的价格判断。"""

    sku_id: str
    spec: str
    price: Money
    stock: int = 0

    def __post_init__(self) -> None:
        """拒绝无法下单或无法稳定标识的 SKU 数据。"""

        if not self.sku_id.strip():
            raise ValueError("sku_id 不能为空")
        if self.stock < 0:
            raise ValueError("stock 不能小于 0")

    def to_dict(self) -> dict[str, object]:
        """输出可 JSON 序列化的 SKU。"""

        return {
            "sku_id": self.sku_id,
            "spec": self.spec,
            "price": self.price.to_dict(),
            "stock": self.stock,
        }


@dataclass(frozen=True, slots=True)
class Product:
    """可被多个检索基础设施共享的商品实体。"""

    item_id: str
    title: str
    category: str = ""
    brand: str = ""
    description: str = ""
    platform: str = ""
    origin_country: str = ""
    ships_to: tuple[str, ...] = ()
    skus: tuple[Sku, ...] = ()
    primary_sku_id: str | None = None
    attributes: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, JsonScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """校验身份，并规范化国家/地区代码。"""

        if not self.item_id.strip():
            raise ValueError("item_id 不能为空")
        if not self.title.strip():
            raise ValueError("title 不能为空")
        normalized_destinations = tuple(
            destination.strip().upper() for destination in self.ships_to
        )
        if any(
            len(destination) != 2 or not destination.isalpha()
            for destination in normalized_destinations
        ):
            raise ValueError("ships_to 必须使用两位字母国家/地区代码")
        object.__setattr__(self, "ships_to", normalized_destinations)
        if self.primary_sku_id is not None:
            primary_sku_id = self.primary_sku_id.strip()
            if not primary_sku_id:
                raise ValueError("primary_sku_id 不能为空字符串")
            if primary_sku_id not in {sku.sku_id for sku in self.skus}:
                raise ValueError("primary_sku_id 必须引用当前商品中的 SKU")
            object.__setattr__(self, "primary_sku_id", primary_sku_id)

    def primary_sku(self) -> Sku:
        """返回显式主 SKU；旧数据没有标记时回退到首个有库存 SKU。"""

        if self.primary_sku_id is not None:
            for sku in self.skus:
                if sku.sku_id == self.primary_sku_id:
                    if sku.stock < 1:
                        raise ValueError("商品主 SKU 当前无库存")
                    return sku
        for sku in self.skus:
            if sku.stock > 0:
                return sku
        raise ValueError("商品没有可售 SKU")

    def to_search_text(self) -> str:
        """生成稳定的商品语义检索文本。"""

        fields = [
            ("商品名", self.title),
            ("类目", self.category),
            ("品牌", self.brand),
            ("平台", self.platform),
            ("产地", self.origin_country),
            ("描述", self.description),
        ]
        lines = [f"{label}: {value}" for label, value in fields if value]
        if self.ships_to:
            lines.append(f"可配送: {', '.join(self.ships_to)}")
        if self.skus:
            sku_text = "；".join(
                f"{sku.spec} {sku.price.major:g} {sku.price.currency} 库存{sku.stock}"
                for sku in self.skus
            )
            lines.append(f"SKU: {sku_text}")
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
            "origin_country": self.origin_country,
            "ships_to": list(self.ships_to),
            "skus": [sku.to_dict() for sku in self.skus],
            "primary_sku_id": self.primary_sku_id,
            "attributes": dict(self.attributes),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "Product":
        """从索引元数据恢复商品及其嵌套值对象。"""

        sku_rows = raw.get("skus", ())
        skus = tuple(
            Sku(
                sku_id=str(row["sku_id"]),
                spec=str(row.get("spec", "")),
                price=(
                    Money(
                        amount_minor=int(row["price"]["amount_minor"]),
                        currency=str(row["price"]["currency"]),
                    )
                    if "amount_minor" in row["price"]
                    else Money.from_major_units(
                        row["price"]["major"],
                        str(row["price"]["currency"]),
                    )
                ),
                stock=int(row.get("stock", 0)),
            )
            for row in sku_rows
        )
        return cls(
            item_id=str(raw["item_id"]),
            title=str(raw["title"]),
            category=str(raw.get("category", "")),
            brand=str(raw.get("brand", "")),
            description=str(raw.get("description", "")),
            platform=str(raw.get("platform", "")),
            origin_country=str(raw.get("origin_country", "")),
            ships_to=tuple(str(item) for item in raw.get("ships_to", ())),
            skus=skus,
            primary_sku_id=(
                str(raw["primary_sku_id"])
                if raw.get("primary_sku_id") is not None
                else None
            ),
            attributes=dict(raw.get("attributes", {})),
            metadata=dict(raw.get("metadata", {})),
        )
