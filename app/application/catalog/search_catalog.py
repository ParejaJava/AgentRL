"""商品召回、重排、硬约束过滤与到手价内联应用用例。"""

import logging
import math
from dataclasses import dataclass, replace

from app.domain.catalog import Money, Product, ProductSearchSpec
from app.domain.shipping import ShippingQuote

from .category_resolution import SearchClarificationError
from .config import ItemSearchConfig
from .fusion import fuse_request_vector
from .models import (
    FilteredItem,
    ItemSearchCommand,
    ItemSearchResponse,
    RankedItem,
    RecallHit,
    RecallStrategy,
)
from .ports import (
    EmbeddingEncoder,
    IndexRegistry,
    ItemVectorIndex,
    NullUserSignalProvider,
    Reranker,
    UserSignalProvider,
)
from .pricing_ports import PricingProvider

logger = logging.getLogger(__name__)

_FILTERED_OUT_LIMIT = 3


@dataclass(frozen=True, slots=True)
class _Candidate:
    """统一承载向量、词法和重排阶段的候选分数。"""

    item_id: str
    retrieval_score: float
    rerank_score: float


class ItemSearchService:
    """在选定商品域内召回、重排、过滤并组装可购买商品卡。"""

    def __init__(
        self,
        *,
        encoder: EmbeddingEncoder,
        reranker: Reranker,
        indexes: IndexRegistry,
        user_signals: UserSignalProvider | None = None,
        pricing: PricingProvider | None = None,
        config: ItemSearchConfig | None = None,
    ) -> None:
        self._encoder = encoder
        self._reranker = reranker
        self._indexes = indexes
        self._user_signals = user_signals or NullUserSignalProvider()
        self._pricing = pricing
        self._config = config or ItemSearchConfig()
        if self._encoder.dimension != self._config.embedding_dimension:
            raise ValueError(
                "encoder dimension does not match ItemSearch configuration"
            )

    def resolve_category(self, index_id: str, query: str) -> dict[str, object]:
        """Resolve only against the selected catalog; unavailable metadata is explicit."""
        index = self._indexes.get(index_id)
        catalog = getattr(index, "category_catalog", None)
        if not callable(catalog):
            return {
                "status": "unavailable",
                "code": "category_catalog_unavailable",
                "message": "当前索引未提供品类目录，不能自动推断别名。",
            }
        return catalog().resolve(query)

    def search(self, command: ItemSearchCommand) -> ItemSearchResponse:
        """执行三级召回链，并在返回商品卡中内联估算到手价。"""

        spec = command.spec
        if spec.top_k > self._config.max_result_k:
            raise ValueError(f"top_k must be between 1 and {self._config.max_result_k}")
        if (
            spec.ship_to
            and self._pricing is not None
            and spec.ship_to not in self._pricing.supported_destinations()
        ):
            supported = ", ".join(self._pricing.supported_destinations())
            raise ValueError(f"暂不支持目的地 {spec.ship_to}；当前支持：{supported}")

        index = self._indexes.get(command.index_id)
        if index.dimension != self._encoder.dimension:
            raise ValueError("selected index dimension does not match the encoder")
        if spec.category:
            resolution = self.resolve_category(command.index_id, spec.category)
            if resolution["status"] == "resolved":
                spec = replace(spec, category=str(resolution["category"]))
            elif resolution["status"] == "needs_clarification":
                raise SearchClarificationError(resolution)

        # 品类既作为语义提示参与召回，之后也会作为确定性硬约束检查。
        retrieval_query = (
            spec.normalized_query
            if not spec.category
            else f"{spec.category} {spec.normalized_query}"
        )
        user_signal = (
            self._user_signals.get(command.buyer_id, command.index_id)
            if command.buyer_id
            else None
        )
        recall_limit = min(self._config.recall_k, index.size)
        strategy: RecallStrategy = "embedding_rerank"
        if recall_limit < 1:
            return self._empty_response(command, "keyword_2gram")

        if self._config.retrieval_mode == "lexical":
            strategy = "keyword_2gram"
            hits = list(index.keyword_search(retrieval_query, recall_limit))
        else:
            try:
                encoded = self._encoder.embed_queries([retrieval_query])
                if len(encoded) != 1 or len(encoded[0]) != self._encoder.dimension:
                    raise RuntimeError(
                        "query encoder returned an unexpected vector shape"
                    )
                request_vector = fuse_request_vector(
                    encoded[0],
                    user_signal.vector if user_signal else None,
                    query_weight=self._config.query_weight,
                )
                hits = list(index.search(request_vector, recall_limit))
            except Exception:
                logger.warning(
                    "商品 embedding 召回失败，降级到 keyword_2gram",
                    exc_info=True,
                )
                strategy = "keyword_2gram"
                hits = list(index.keyword_search(retrieval_query, recall_limit))
            else:
                if not hits:
                    strategy = "keyword_2gram"
                    hits = list(index.keyword_search(retrieval_query, recall_limit))

        if not hits:
            return self._empty_response(command, strategy)

        ranked_candidates, strategy = self._rank_candidates(
            hits=hits,
            index=index,
            query=spec.normalized_query,
            user_summary=user_signal.summary if user_signal else "",
            strategy=strategy,
        )

        eligible: list[_Candidate] = []
        filtered_out: list[FilteredItem] = []
        quotes: dict[str, ShippingQuote] = {}
        seen_ids: set[str] = set()
        for candidate in ranked_candidates:
            if candidate.item_id in seen_ids:
                continue
            seen_ids.add(candidate.item_id)
            product = index.get_product(candidate.item_id)
            reason, converted_price = self._reject_reason(product, spec, quotes)
            if reason is None:
                eligible.append(candidate)
            elif len(filtered_out) < _FILTERED_OUT_LIMIT:
                filtered_out.append(
                    FilteredItem(
                        item_id=product.item_id,
                        title=product.title,
                        category=product.category,
                        reason=reason,
                        converted_price=converted_price,
                    )
                )

        items = [
            self._to_ranked_item(
                rank=rank,
                candidate=candidate,
                product=index.get_product(candidate.item_id),
                spec=spec,
                quote=quotes.get(candidate.item_id),
            )
            for rank, candidate in enumerate(eligible[: spec.top_k], start=1)
        ]
        return ItemSearchResponse(
            query=spec.normalized_query,
            index_id=command.index_id,
            recall_count=len(hits),
            items=items,
            recall_strategy=strategy,
            filtered_out=tuple(filtered_out),
        )

    def _rank_candidates(
        self,
        *,
        hits: list[RecallHit],
        index: ItemVectorIndex,
        query: str,
        user_summary: str,
        strategy: RecallStrategy,
    ) -> tuple[list[_Candidate], RecallStrategy]:
        """执行 reranker；失败时返回原召回顺序和 embedding_only 标记。"""

        if strategy == "keyword_2gram":
            return (
                [_Candidate(hit.item_id, hit.score, hit.score) for hit in hits],
                strategy,
            )

        if self._config.retrieval_mode == "embedding":
            return (
                [_Candidate(hit.item_id, hit.score, hit.score) for hit in hits],
                "embedding_only",
            )

        products = [index.get_product(hit.item_id) for hit in hits]
        rerank_query = self._build_rerank_query(query, user_summary)
        try:
            scores = list(
                self._reranker.score(
                    [(rerank_query, product.to_search_text()) for product in products]
                )
            )
            if len(scores) != len(hits):
                raise RuntimeError(
                    "reranker score count does not match recalled candidates"
                )
            if not all(math.isfinite(float(score)) for score in scores):
                raise RuntimeError("reranker returned a non-finite score")
            candidates = [
                _Candidate(hit.item_id, hit.score, float(score))
                for hit, score in zip(hits, scores, strict=True)
            ]
            candidates.sort(
                key=lambda candidate: (
                    candidate.rerank_score,
                    candidate.retrieval_score,
                ),
                reverse=True,
            )
            return candidates, "embedding_rerank"
        except Exception:
            logger.warning(
                "商品 reranker 失败，降级到 embedding_only",
                exc_info=True,
            )
            return (
                [_Candidate(hit.item_id, hit.score, hit.score) for hit in hits],
                "embedding_only",
            )

    def _reject_reason(
        self,
        product: Product,
        spec: ProductSearchSpec,
        quotes: dict[str, ShippingQuote] | None = None,
    ) -> tuple[str | None, Money | None]:
        """返回硬约束拒绝原因及可选的目标币种商品价格。"""

        if spec.category and product.category.casefold() != spec.category.casefold():
            return "category_mismatch", None
        if spec.ship_to and spec.ship_to not in product.ships_to:
            return "ship_to_unavailable", None
        if product.brand.casefold() in {b.casefold() for b in spec.excluded_brands}:
            return "excluded_brand", None
        if (
            spec.required_brand
            and product.brand.casefold() != spec.required_brand.casefold()
        ):
            return "brand_mismatch", None
        try:
            primary = product.primary_sku()
        except ValueError:
            return "no_available_sku", None
        if primary.stock < spec.quantity:
            return "insufficient_stock", None

        converted_price: Money | None = None
        if spec.price_cap is not None:
            try:
                converted_price = self._convert_price(
                    primary.price,
                    spec.target_currency,
                )
                if spec.price_basis == "subtotal":
                    converted_price = Money(
                        converted_price.amount_minor * spec.quantity,
                        converted_price.currency,
                    )
                elif spec.price_basis == "landed":
                    if self._pricing is None:
                        return "pricing_unavailable", None
                    quote = self._pricing.quote(
                        unit_price=primary.price,
                        category=product.category,
                        ship_to=spec.ship_to,
                        quantity=spec.quantity,
                        target_currency=spec.target_currency,
                    )
                    converted_price = quote.landed_total
                    if quotes is not None:
                        quotes[product.item_id] = quote
            except ValueError:
                return "pricing_unavailable", None
            if converted_price.amount_minor > spec.price_cap.amount_minor:
                return "over_price_cap", converted_price
        return None, converted_price

    def _convert_price(self, price: Money, target_currency: str) -> Money:
        """通过计价端口转换价格；未配置端口时只允许同币种。"""

        if self._pricing is not None:
            return self._pricing.convert(price, target_currency)
        if price.currency != target_currency:
            raise ValueError("未配置汇率提供器，不能执行跨币种预算过滤")
        return price

    def _to_ranked_item(
        self,
        *,
        rank: int,
        candidate: _Candidate,
        product: Product,
        spec: ProductSearchSpec,
        quote: ShippingQuote | None = None,
    ) -> RankedItem:
        """组装最终商品卡，并在指定收货地时内联估算到手价。"""

        pricing_unavailable_reason = None
        if spec.ship_to and quote is None:
            try:
                primary = product.primary_sku()
                if self._pricing is None:
                    raise ValueError("当前未配置到手价提供器")
                quote = self._pricing.quote(
                    unit_price=primary.price,
                    category=product.category,
                    ship_to=spec.ship_to,
                    quantity=spec.quantity,
                    target_currency=spec.target_currency,
                )
            except ValueError as exc:
                # 到手价不可用不应抹掉已经满足配送和商品价格约束的候选。
                pricing_unavailable_reason = str(exc)
        return RankedItem(
            rank=rank,
            product=product,
            retrieval_score=candidate.retrieval_score,
            rerank_score=candidate.rerank_score,
            landed_price=quote,
            pricing_unavailable_reason=pricing_unavailable_reason,
        )

    @staticmethod
    def _empty_response(
        command: ItemSearchCommand,
        strategy: RecallStrategy,
    ) -> ItemSearchResponse:
        """创建保持查询和执行策略信息的空搜索响应。"""

        return ItemSearchResponse(
            query=command.spec.normalized_query,
            index_id=command.index_id,
            recall_count=0,
            items=(),
            recall_strategy=strategy,
        )

    @staticmethod
    def _build_rerank_query(query: str, user_summary: str) -> str:
        """把可解释的买家偏好附加到重排查询。"""

        return (
            query
            if not user_summary
            else f"当前查询：{query}\n用户偏好：{user_summary}"
        )
