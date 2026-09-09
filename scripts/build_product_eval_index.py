"""使用评测种子商品构建 BGE-M3 + FAISS 索引。"""

from __future__ import annotations

import argparse
import json
import runpy
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

from app.application.catalog import ItemSearchConfig
from app.composition import build_item_index_builder
from app.domain.catalog import Product
from app.infrastructure.retrieval.item_search.faiss_index import FaissItemIndex
from app.infrastructure.settings import Settings


def _parse_args() -> argparse.Namespace:
    """解析种子文件、索引域和输出目录。"""

    parser = argparse.ArgumentParser(description="构建商品离线评测索引")
    parser.add_argument(
        "--seed",
        type=Path,
        default=Path("eval/seed_products.py"),
    )
    parser.add_argument("--index-id", default="evaluation-products")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="已有索引与当前模型、维度和商品数一致时复用。",
    )
    return parser.parse_args()


def _load_products(path: Path) -> tuple[Product, ...]:
    """执行受版本控制的本地种子模块并校验其返回对象。"""

    namespace: dict[str, Any] = runpy.run_path(str(path))
    raw_builder = namespace.get("build_seed_products")
    if not callable(raw_builder):
        raise TypeError(f"种子模块缺少 build_seed_products()：{path}")
    builder = cast(Callable[[], Sequence[Product]], raw_builder)
    products = tuple(builder())
    if not products or any(not isinstance(item, Product) for item in products):
        raise ValueError("build_seed_products() 必须返回非空 Product 序列")
    if len({item.item_id for item in products}) != len(products):
        raise ValueError("评测种子商品 item_id 不能重复")
    return products


def main() -> None:
    """构建一次性评测索引；已有非空索引不会被覆盖。"""

    args = _parse_args()
    settings = Settings.from_env()
    products = _load_products(args.seed)
    index_config = ItemSearchConfig()
    output_root = args.output_root or settings.item_index_root
    target = output_root.resolve() / args.index_id
    if args.reuse and target.is_dir():
        existing = FaissItemIndex.load(target)
        if existing.manifest.embedding_model != index_config.embedding_model:
            raise ValueError("已有评测索引的 embedding 模型与当前配置不一致")
        if existing.dimension != index_config.embedding_dimension:
            raise ValueError("已有评测索引维度与当前配置不一致")
        expected_ids = {item.item_id for item in products}
        actual_ids = {
            row["product"]["item_id"]
            for row in json.loads(
                (target / "products.json").read_text(encoding="utf-8")
            )
        }
        if actual_ids != expected_ids:
            raise ValueError("已有评测索引商品快照与当前种子数据不一致")
        print(f"products\t{len(products)}")
        print(f"index\t{target}")
        print("reused\ttrue")
        return
    builder = build_item_index_builder(settings)
    target = builder.build(
        index_id=args.index_id,
        products=products,
        output_root=output_root,
    )
    print(f"products\t{len(products)}")
    print(f"index\t{target}")


if __name__ == "__main__":
    main()
