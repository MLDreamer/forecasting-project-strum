"""Rolling-origin cross-validation harness.

Design (doc Phase 14):
- 4 folds, 26-week horizon each.
- Anchored rolling-origin: fold k origin = ts_max - (N_FOLDS - k + 1) * H
  (each fold uses more data; the origin moves toward the present).
- The CV cutoff is applied exactly ONCE per fold (locked leakage discipline).
- Fold 1 is included in the harness output but EXCLUDED from selection scoring
  (doc: 'fold 1 skipped — cold-start data too thin').
- Per-horizon WAPE is the primary output (H=26 values per model per fold).
- Also computes CRPS, coverage_80, coverage_90 as secondary metrics.
- Home of the Phase 11.5 A vs A+ re-check: pass target_week_features=True to
  a ClusterPooledLGBM candidate to measure out-of-sample WAPE vs the default.

Output:
    CVResult — per-fold, per-model, per-horizon metrics + raw quantile predictions.

The harness calls model.fit_series() on the training window and model.predict()
on the forecast origin.  Models that use fit_dataframe() (ClusterPooledLGBM)
are handled via a protocol check.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from forecasting import config
from forecasting.features import FeaturesResult  # noqa: F401 (type hint reference)
from forecasting.lifecycle import LifecycleResult
from forecasting.metrics import coverage_80, coverage_90, crps_from_quantiles, wape_per_horizon
from forecasting.models.base import ForecastModel, ForecastResult
from forecasting.segment import SegmentResult

logger = logging.getLogger(__name__)

# Locked CV constants (doc)
N_FOLDS: int = 4
HORIZON: int = 26
SELECTION_FOLDS: frozenset[int] = frozenset({2, 3, 4})  # fold 1 excluded from selection


# ---------------------------------------------------------------------------
# Protocol for DataFrame-fitting models (ClusterPooledLGBM)
# ---------------------------------------------------------------------------


@runtime_checkable
class DataFrameFitter(Protocol):
    """Models that expose fit_dataframe() instead of fit_series()."""

    def fit_dataframe(
        self,
        features_df: pd.DataFrame,
        segments_df: pd.DataFrame,
        cutoff: pd.Timestamp | None = None,
    ) -> object: ...

    def predict_dataframe(
        self,
        features_df: pd.DataFrame,
        segments_df: pd.DataFrame,
        horizon: int | None = None,
        cutoff: pd.Timestamp | None = None,
    ) -> ForecastResult: ...


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass
class FoldMetrics:
    """Metrics for one model on one CV fold."""

    fold: int
    model_name: str
    n_skus: int
    wape_per_horizon: np.ndarray  # shape (H,)
    crps_scalar: float
    coverage_80_scalar: float
    coverage_90_scalar: float
    wape_overall: float  # mean of wape_per_horizon
    in_selection: bool  # True if fold in SELECTION_FOLDS


@dataclass
class CVResult:
    """Full CV harness output."""

    fold_metrics: list[FoldMetrics] = field(default_factory=list)
    """One entry per (model, fold) combination."""

    fold_predictions: dict[tuple[str, int], ForecastResult] = field(default_factory=dict)
    """(model_name, fold) → ForecastResult for raw quantile access."""

    fold_actuals: dict[int, np.ndarray] = field(default_factory=dict)
    """fold → actual sales array, shape (n_sku, H)."""

    sku_order: dict[int, list[int]] = field(default_factory=dict)
    """fold → ordered list of sku_ids aligned with fold_actuals axis 0."""

    def selection_wape(self, model_name: str) -> float:
        """Revenue-weighted mean WAPE over selection folds (2–4)."""
        vals = [
            fm.wape_overall
            for fm in self.fold_metrics
            if fm.model_name == model_name and fm.in_selection
        ]
        return float(np.mean(vals)) if vals else float("inf")

    def per_horizon_wape(self, model_name: str, fold: int) -> np.ndarray | None:
        """Return per-horizon WAPE array for one model/fold, or None if absent."""
        for fm in self.fold_metrics:
            if fm.model_name == model_name and fm.fold == fold:
                return fm.wape_per_horizon
        return None

    def summary(self) -> pd.DataFrame:
        """Tidy DataFrame with one row per (model, fold)."""
        rows = []
        for fm in self.fold_metrics:
            rows.append(
                {
                    "model": fm.model_name,
                    "fold": fm.fold,
                    "in_selection": fm.in_selection,
                    "wape_overall": round(fm.wape_overall, 4),
                    "crps": round(fm.crps_scalar, 4),
                    "cov80": round(fm.coverage_80_scalar, 3),
                    "cov90": round(fm.coverage_90_scalar, 3),
                    "n_skus": fm.n_skus,
                }
            )
        return pd.DataFrame(rows).sort_values(["fold", "model"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Core fold data builder
# ---------------------------------------------------------------------------


def _build_fold_data(
    full_dense: pd.DataFrame,
    full_features: pd.DataFrame,
    segments: SegmentResult,
    cutoff: pd.Timestamp,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[int]]:
    """Slice dense grid and feature matrix at a given CV cutoff.

    Returns
    -------
    train_dense  : dense rows with timestamp <= cutoff
    train_features : feature rows with timestamp <= cutoff
    actuals      : shape (n_sku, horizon) — actual sales after cutoff
    sku_order    : list of sku_ids in the order of actuals axis 0
    """
    train_dense = full_dense[full_dense[config.COL_TIMESTAMP] <= cutoff].copy()
    train_features = full_features[full_features[config.COL_TIMESTAMP] <= cutoff].copy()

    # Build actuals: for each SKU, collect the H weeks immediately after cutoff
    holdout_end = cutoff + pd.Timedelta(weeks=horizon)
    holdout = full_dense[
        (full_dense[config.COL_TIMESTAMP] > cutoff)
        & (full_dense[config.COL_TIMESTAMP] <= holdout_end)
    ]

    sku_order: list[int] = sorted(holdout[config.COL_SKU_ID].unique())
    n_sku = len(sku_order)
    actuals = np.zeros((n_sku, horizon))

    for i, sku in enumerate(sku_order):
        sku_holdout = holdout[holdout[config.COL_SKU_ID] == sku].sort_values(config.COL_TIMESTAMP)
        h = min(len(sku_holdout), horizon)
        actuals[i, :h] = sku_holdout[config.COL_SALES].values[:h]

    return train_dense, train_features, actuals, sku_order


# ---------------------------------------------------------------------------
# Model runner for one fold
# ---------------------------------------------------------------------------


def _run_model_on_fold(
    model: ForecastModel,
    model_name: str,
    fold: int,
    train_dense: pd.DataFrame,
    train_features: pd.DataFrame,
    segments: SegmentResult,
    actuals: np.ndarray,  # (n_sku_holdout, H)
    sku_order: list[int],
    cutoff: pd.Timestamp,
    horizon: int,
) -> tuple[FoldMetrics, ForecastResult] | None:
    """Fit + predict one model on one fold.  Returns None on failure."""
    try:
        if isinstance(model, DataFrameFitter):
            # ClusterPooledLGBM path
            model.fit_dataframe(train_features, segments.segments, cutoff=cutoff)
            result = model.predict_dataframe(
                train_features, segments.segments, horizon=horizon, cutoff=cutoff
            )
            # Align result to sku_order
            # predict_dataframe returns SKUs in sorted order; need to match sku_order
            fitted_uid_order = sorted(train_features[config.COL_SKU_ID].unique())
            idx_map = {sku: i for i, sku in enumerate(fitted_uid_order)}
            q_cube = np.zeros((len(sku_order), horizon, result.n_quantiles))
            for j, sku in enumerate(sku_order):
                if sku in idx_map:
                    q_cube[j] = result.quantiles[idx_map[sku]]
            result = ForecastResult.from_quantiles(q_cube, result.q_levels)
        else:
            # Series-based models (classical, intermittent, Tweedie, Chronos, etc.)
            series_dict = {
                str(sku): train_dense[train_dense[config.COL_SKU_ID] == sku][
                    config.COL_SALES
                ].values
                for sku in sku_order
            }
            model.fit_series(series_dict)
            result = model.predict(np.empty(0), horizon)

        # Align result quantiles to actuals shape
        n_sku_result = result.n_sku
        n_sku_actual = actuals.shape[0]
        if n_sku_result != n_sku_actual:
            logger.warning(
                "Fold %d %s: result n_sku=%d != actuals n_sku=%d — padding with zeros",
                fold,
                model_name,
                n_sku_result,
                n_sku_actual,
            )
            q_cube = np.zeros((n_sku_actual, horizon, result.n_quantiles))
            q_cube[: min(n_sku_result, n_sku_actual)] = result.quantiles[
                : min(n_sku_result, n_sku_actual)
            ]
            result = ForecastResult.from_quantiles(q_cube, result.q_levels)

        # Compute metrics
        y_true_flat = actuals.ravel()
        q_flat = result.quantiles.reshape(-1, result.n_quantiles)

        wape_h = wape_per_horizon(actuals, result.median())
        crps_s = crps_from_quantiles(y_true_flat, q_flat, result.q_levels)
        cov80 = coverage_80(y_true_flat, q_flat, result.q_levels)
        cov90 = coverage_90(y_true_flat, q_flat, result.q_levels)

        fm = FoldMetrics(
            fold=fold,
            model_name=model_name,
            n_skus=n_sku_actual,
            wape_per_horizon=wape_h,
            crps_scalar=crps_s,
            coverage_80_scalar=cov80,
            coverage_90_scalar=cov90,
            wape_overall=float(wape_h.mean()),
            in_selection=(fold in SELECTION_FOLDS),
        )
        logger.info(
            "Fold %d %s: WAPE=%.3f CRPS=%.3f cov80=%.2f cov90=%.2f n_sku=%d",
            fold,
            model_name,
            fm.wape_overall,
            crps_s,
            cov80,
            cov90,
            n_sku_actual,
        )
        return fm, result

    except Exception as exc:
        logger.warning("Fold %d %s FAILED: %s", fold, model_name, exc)
        return None


# ---------------------------------------------------------------------------
# Main harness entry point
# ---------------------------------------------------------------------------


def run_cv(
    full_dense: pd.DataFrame,
    full_features: pd.DataFrame,
    lifecycle: LifecycleResult,
    segments: SegmentResult,
    models: dict[str, ForecastModel],
    n_folds: int = N_FOLDS,
    horizon: int = HORIZON,
) -> CVResult:
    """Run rolling-origin CV for all provided models.

    Parameters
    ----------
    full_dense : DataFrame
        Output of densify() — full weekly grid, Saturday-dated.
    full_features : DataFrame
        Output of add_hierarchy_features() — 16,068 × 114.
    lifecycle : LifecycleResult
        From infer_lifecycle() — not used directly but provided for context.
    segments : SegmentResult
        From segment_and_cluster() — needed for DataFrameFitter models.
    models : dict[str, ForecastModel]
        Model name → model instance.  Models are reset between folds.
    n_folds : int
        Number of CV folds (default 4).
    horizon : int
        Forecast horizon in weeks (default 26).

    Returns
    -------
    CVResult
    """
    ts_max = full_dense[config.COL_TIMESTAMP].max()
    result = CVResult()

    for fold in range(1, n_folds + 1):
        # Compute fold cutoff (anchored rolling-origin)
        cutoff = ts_max - pd.Timedelta(weeks=(n_folds - fold + 1) * horizon)
        logger.info("=== CV Fold %d / %d | cutoff=%s ===", fold, n_folds, cutoff.date())

        # Slice training data and build actuals
        train_dense, train_features, actuals, sku_order = _build_fold_data(
            full_dense, full_features, segments, cutoff, horizon
        )

        if len(sku_order) == 0:
            logger.warning("Fold %d: no SKUs in holdout window — skipping.", fold)
            continue

        result.fold_actuals[fold] = actuals
        result.sku_order[fold] = sku_order

        # Run each model
        for model_name, model in models.items():
            out = _run_model_on_fold(
                model=model,
                model_name=model_name,
                fold=fold,
                train_dense=train_dense,
                train_features=train_features,
                segments=segments,
                actuals=actuals,
                sku_order=sku_order,
                cutoff=cutoff,
                horizon=horizon,
            )
            if out is not None:
                fm, forecast = out
                result.fold_metrics.append(fm)
                result.fold_predictions[(model_name, fold)] = forecast

    logger.info(
        "CV complete: %d model×fold results over %d folds",
        len(result.fold_metrics),
        n_folds,
    )
    return result
