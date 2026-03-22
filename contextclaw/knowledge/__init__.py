from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bridge import ContextGraphBridge as ContextGraphBridge


def __getattr__(name: str) -> object:
    if name == "ContextGraphBridge":
        from .bridge import ContextGraphBridge

        return ContextGraphBridge
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ContextGraphBridge"]
