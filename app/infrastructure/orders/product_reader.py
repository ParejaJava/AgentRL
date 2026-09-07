"""从商品检索索引读取订单校验所需快照。"""

from app.application.catalog.ports import IndexRegistry
from app.domain.catalog import Product


class IndexedProductReader:
    """把检索索引注册表适配为订单应用层的商品读取端口。"""

    def __init__(self, indexes: IndexRegistry) -> None:
        self._indexes = indexes

    def get(self, index_id: str, item_id: str) -> Product | None:
        """读取商品；索引存在但商品未命中时返回 ``None``。"""

        index = self._indexes.get(index_id)
        try:
            return index.get_product(item_id.strip())
        except KeyError:
            return None
