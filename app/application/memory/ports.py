"""长期偏好存储端口。"""

from typing import Protocol

from app.domain.buyer import BuyerPreference


class PreferenceStore(Protocol):
    """保存、读取和精确删除买家偏好的异步端口。"""

    async def save(self, preference: BuyerPreference) -> BuyerPreference: ...

    async def list_by_buyer(self, buyer_id: str) -> tuple[BuyerPreference, ...]: ...

    async def delete_exact(self, buyer_id: str, statement: str) -> bool: ...


class SyncPreferenceReader(Protocol):
    """供在线同步商品检索线程读取买家偏好的最小端口。"""

    def list_by_buyer_sync(self, buyer_id: str) -> tuple[BuyerPreference, ...]: ...
