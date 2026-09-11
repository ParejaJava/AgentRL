"""Resolve catalog categories using explicit names/aliases, never fuzzy guesses."""

from collections.abc import Iterable
from dataclasses import dataclass

from app.domain.catalog import Product


@dataclass(frozen=True, slots=True)
class CategoryCatalog:
    names: tuple[str, ...]
    aliases: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_products(cls, products: Iterable[Product]) -> "CategoryCatalog":
        """Read canonical categories and curated pipe-separated metadata aliases."""
        names: set[str] = set()
        aliases: set[tuple[str, str]] = set()
        for product in products:
            if not product.category.strip():
                continue
            names.add(product.category.strip())
            raw = product.metadata.get("category_aliases", "")
            if isinstance(raw, str):
                aliases.update(
                    (alias.strip(), product.category.strip())
                    for alias in raw.split("|")
                    if alias.strip()
                )
        return cls(tuple(sorted(names)), tuple(sorted(aliases)))

    def resolve(self, query: str) -> dict[str, object]:
        """Return a unique authoritative mapping or bounded clarification candidates."""
        key = query.strip().casefold()
        exact = sorted({name for name in self.names if name.casefold() == key})
        aliased = sorted(
            {
                name
                for alias, name in self.aliases
                if alias.casefold() == key and name in self.names
            }
        )
        matches = exact or aliased
        if len(matches) == 1:
            return {
                "status": "resolved",
                "category": matches[0],
                "source": "canonical" if exact else "curated_alias",
            }
        suggestions = matches or sorted(
            {
                name
                for name in self.names
                if key and (key in name.casefold() or name.casefold() in key)
            }
        )
        return {
            "status": "needs_clarification",
            "code": "category_ambiguous"
            if len(suggestions) > 1
            else "category_unresolved",
            "candidates": suggestions[:10],
            "message": "请核对目录品类；候选只是建议，不能自动替用户选择或取消品类限制。",
        }


class SearchClarificationError(ValueError):
    """Carry a structured business clarification across the tool boundary."""

    def __init__(self, details: dict[str, object]) -> None:
        self.details = details
        super().__init__(str(details.get("message", "搜索条件需要确认")))
