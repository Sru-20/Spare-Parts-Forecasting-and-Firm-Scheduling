"""Catalogue text classification helpers."""

from typing import Any

__all__ = [
    "CategoryCandidate",
    "CategoryResolution",
    "PartCategoryPredictor",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from .part_category import (
        CategoryCandidate,
        CategoryResolution,
        PartCategoryPredictor,
    )

    exports = {
        "CategoryCandidate": CategoryCandidate,
        "CategoryResolution": CategoryResolution,
        "PartCategoryPredictor": PartCategoryPredictor,
    }
    return exports[name]
