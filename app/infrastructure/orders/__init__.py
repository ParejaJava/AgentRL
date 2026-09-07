"""订单意向单基础设施适配器。"""

from .product_reader import IndexedProductReader
from .sqlite_repository import SQLiteOrderRepository

__all__ = ["IndexedProductReader", "SQLiteOrderRepository"]
