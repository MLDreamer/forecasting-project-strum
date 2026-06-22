"""Pydantic v2 schema for the pipeline YAML config.

All client-specific values live in a YAML validated by this schema.
The core in src/forecasting/ must never contain client-specific literals.

Usage:
    from configs._schema import PipelineConfig
    cfg = PipelineConfig.from_yaml(path)
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class KeepActiveOverride(BaseModel):
    sku_id: int
    reason: str = Field(min_length=1)


class DataConfig(BaseModel):
    raw_dir: Path = Path("data/raw")
    sales_file: str = "processed_data_filtered.csv"
    master_file: str = "product_item_master.csv"
    variants_file: str = "variants_export.csv"
    schema_map: dict[str, str] = Field(default_factory=dict)


class LifecycleConfig(BaseModel):
    dormancy_threshold_weeks: Annotated[int, Field(ge=1)] = 26
    keep_active_overrides: list[KeepActiveOverride] = Field(default_factory=list)


class DensifyConfig(BaseModel):
    week_relabel_shift_days: Annotated[int, Field(ge=0, le=6)] = 6


class FeaturesConfig(BaseModel):
    target_week_features: bool = False


class SegmentConfig(BaseModel):
    fallback_k: Annotated[int, Field(ge=2, le=32)] = 8
    stability_ari_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    revenue_tier_anchor: str = "deployment_time"


class ModelConfig(BaseModel):
    quantiles: list[Annotated[float, Field(ge=0.0, le=1.0)]] = Field(
        default=[
            0.05,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95,
        ]
    )
    horizon_weeks: Annotated[int, Field(ge=1)] = 26
    cv_folds: Annotated[int, Field(ge=2)] = 4
    selection_folds: list[Annotated[int, Field(ge=1)]] = Field(default=[2, 3, 4])
    calibration_guardrail: tuple[float, float] = (0.75, 0.85)
    use_heavy_models: bool = False

    @field_validator("quantiles")
    @classmethod
    def quantiles_sorted_unique(cls, v: list[float]) -> list[float]:
        if sorted(set(v)) != sorted(v):
            raise ValueError("quantiles must be sorted and unique")
        return v

    @field_validator("calibration_guardrail")
    @classmethod
    def guardrail_ordered(cls, v: tuple[float, float]) -> tuple[float, float]:
        if v[0] >= v[1]:
            raise ValueError("calibration_guardrail[0] must be < [1]")
        return v

    @model_validator(mode="after")
    def selection_folds_within_cv(self) -> ModelConfig:
        for f in self.selection_folds:
            if f > self.cv_folds:
                raise ValueError(f"selection_fold {f} exceeds cv_folds {self.cv_folds}")
        return self


class OutputConfig(BaseModel):
    forecast_horizon_start: str = "2026-05-24"
    forecast_horizon_end: str = "2026-11-15"


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class PipelineConfig(BaseModel):
    client: str = Field(min_length=1)
    data: DataConfig = Field(default_factory=DataConfig)
    lifecycle: LifecycleConfig = Field(default_factory=LifecycleConfig)
    densify: DensifyConfig = Field(default_factory=DensifyConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    segment: SegmentConfig = Field(default_factory=SegmentConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        """Load and validate a YAML config file."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(raw)

    @classmethod
    def from_dict(cls, data: dict) -> PipelineConfig:  # type: ignore[type-arg]
        """Validate from a plain dict (useful for tests and notebook overrides)."""
        return cls.model_validate(data)

    # ------------------------------------------------------------------
    # ConfigBuilder integration point (Phase 18)
    # ------------------------------------------------------------------

    def override(self, **kwargs: object) -> PipelineConfig:
        """Return a new config with the given top-level fields replaced.

        For deep overrides use ConfigBuilder (Phase 18).
        """
        return self.model_copy(update=kwargs)
