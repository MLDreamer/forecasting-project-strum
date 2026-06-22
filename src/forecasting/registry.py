"""Model / feature / plugin registries.

The pipeline iterates MODEL_REGISTRY.candidates_for(segment) — it never
names a model directly. Each model registers itself with the segment types
it supports via @register_model(name, segments=[...]).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# ---------------------------------------------------------------------------
# Segment type constants (locked, matches SB classification in segment.py)
# ---------------------------------------------------------------------------
SEG_SMOOTH = "smooth"
SEG_ERRATIC = "erratic"
SEG_INTERMITTENT = "intermittent"
SEG_LUMPY = "lumpy"
SEG_COLD_START = "cold_start"
SEG_DISCONTINUED = "discontinued"

ALL_SEGMENTS: frozenset[str] = frozenset(
    {SEG_SMOOTH, SEG_ERRATIC, SEG_INTERMITTENT, SEG_LUMPY, SEG_COLD_START, SEG_DISCONTINUED}
)


# ---------------------------------------------------------------------------
# Registry containers
# ---------------------------------------------------------------------------

# Maps model_name → {"cls": ModelClass, "segments": frozenset[str]}
_MODEL_ENTRIES: dict[str, dict[str, Any]] = {}

FEATURE_REGISTRY: dict[str, Any] = {}
PLUGIN_REGISTRY: dict[str, Any] = {}

# Legacy flat alias kept for Phase 0 scaffold test compatibility
MODEL_REGISTRY: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------


def register_model(
    name: str,
    segments: list[str] | None = None,
) -> Callable[[Any], Any]:
    """Register a ForecastModel class.

    Parameters
    ----------
    name:
        Unique registry key.
    segments:
        List of segment types this model supports.  Defaults to ALL_SEGMENTS.
    """
    seg_set: frozenset[str] = frozenset(segments) if segments else ALL_SEGMENTS

    def decorator(cls: Any) -> Any:
        _MODEL_ENTRIES[name] = {"cls": cls, "segments": seg_set}
        MODEL_REGISTRY[name] = cls  # flat alias for backwards compat
        return cls

    return decorator


def register_feature(name: str) -> Callable[[Any], Any]:
    def decorator(fn: Any) -> Any:
        FEATURE_REGISTRY[name] = fn
        return fn

    return decorator


def register_plugin(name: str) -> Callable[[Any], Any]:
    def decorator(cls: Any) -> Any:
        PLUGIN_REGISTRY[name] = cls
        return cls

    return decorator


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def candidates_for(segment: str) -> list[Any]:
    """Return model classes registered for the given segment type."""
    return [entry["cls"] for entry in _MODEL_ENTRIES.values() if segment in entry["segments"]]


def registered_model_names(segment: str | None = None) -> list[str]:
    """Return registered model names, optionally filtered by segment."""
    if segment is None:
        return list(_MODEL_ENTRIES)
    return [name for name, entry in _MODEL_ENTRIES.items() if segment in entry["segments"]]
