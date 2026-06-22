"""ConfigBuilder — the notebook layer for the pipeline config.

Provides business-term aliases so the client can write:
    cfg = (ConfigBuilder.from_yaml("configs/fontana.yaml")
           .set("weeks of history required", 26)
           .set("scope statuses", ["active", "draft"])
           .build())

The ConfigBuilder always funnels through the pydantic PipelineConfig validation
as the safety gate — it is impossible to bypass schema checks.

Rules:
- Every alias maps to a canonical config path (dot-notation).
- The canonical path can also be used directly (no lookup needed).
- Chaining: every .set() returns self so calls can be chained.
- .build() calls PipelineConfig.model_validate() — any invalid value raises.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

# Business-term aliases → canonical pydantic config paths
ALIASES: dict[str, str] = {
    # Data / scope
    "weeks of history required": "data.min_history_weeks",
    "scope statuses": "data.scope_statuses",
    "sales file": "data.sales_file",
    "master file": "data.master_file",
    # Lifecycle
    "weeks to avoid forecasting": "lifecycle.dormancy_threshold_weeks",
    "dormancy threshold": "lifecycle.dormancy_threshold_weeks",
    # Densify
    "week relabel shift": "densify.week_relabel_shift_days",
    # Forecast / model
    "forecast weeks": "model.horizon_weeks",
    "quantiles": "model.quantiles",
    "cv folds": "model.cv_folds",
    "selection folds": "model.selection_folds",
    "calibration guardrail": "model.calibration_guardrail",
    "use heavy models": "model.use_heavy_models",
    "target week features": "features.target_week_features",
    # Selection
    "selection bar": "model.calibration_guardrail",  # alias for the cov guardrail
    # Segment / cluster
    "fallback k": "segment.fallback_k",
    "stability threshold": "segment.stability_ari_threshold",
}


class ConfigBuilder:
    """Fluent builder for PipelineConfig.

    Usage
    -----
    >>> cfg = (ConfigBuilder.from_yaml("configs/fontana.yaml")
    ...        .set("weeks of history required", 26)
    ...        .set("scope statuses", ["active", "draft"])
    ...        .build())

    The ``set`` method accepts either a business-term alias (see ALIASES above)
    or a canonical dot-path like ``"lifecycle.dormancy_threshold_weeks"``.
    """

    def __init__(self, base: dict) -> None:
        self._d: dict = copy.deepcopy(base)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ConfigBuilder:
        """Load from a YAML file."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(raw)

    @classmethod
    def from_dict(cls, data: dict) -> ConfigBuilder:
        """Load from a plain dict."""
        return cls(data)

    def set(self, path_or_alias: str, value: Any) -> ConfigBuilder:
        """Set a config value by alias or canonical dot-path.

        Returns self for chaining.
        """
        canonical = ALIASES.get(path_or_alias, path_or_alias)
        parts = canonical.split(".")
        node = self._d
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
        return self

    def get(self, path_or_alias: str) -> Any:
        """Read a config value by alias or canonical dot-path."""
        canonical = ALIASES.get(path_or_alias, path_or_alias)
        parts = canonical.split(".")
        node = self._d
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node

    def build(self):
        """Validate and return a PipelineConfig.

        Raises pydantic.ValidationError on any invalid value — this is the
        safety gate that ensures the core stays client-value-free.
        """
        from configs._schema import PipelineConfig

        return PipelineConfig.model_validate(self._d)

    def to_dict(self) -> dict:
        """Return the raw dict (useful for debugging)."""
        return copy.deepcopy(self._d)

    def __repr__(self) -> str:
        return f"ConfigBuilder(client={self._d.get('client', '?')})"
