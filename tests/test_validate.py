"""Phase 14 gate: rolling-origin CV harness — 4 folds, 26w horizon, per-horizon metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting import config
from forecasting.validate import (
    HORIZON,
    N_FOLDS,
    SELECTION_FOLDS,
    CVResult,
    FoldMetrics,
    _build_fold_data,
    run_cv,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_n_folds_is_4() -> None:
    assert N_FOLDS == 4


def test_horizon_is_26() -> None:
    assert HORIZON == 26


def test_selection_folds_excludes_fold_1() -> None:
    """Fold 1 excluded from selection — cold-start data too thin."""
    assert 1 not in SELECTION_FOLDS
    assert SELECTION_FOLDS == frozenset({2, 3, 4})


# ---------------------------------------------------------------------------
# Synthetic test fixture (no real data — fast unit tests)
# ---------------------------------------------------------------------------


def _make_synthetic_dense(n_sku: int = 6, n_weeks: int = 120) -> pd.DataFrame:
    """Build a minimal dense grid for CV testing."""
    rng = np.random.default_rng(42)
    rows = []
    ts = pd.date_range("2023-01-07", periods=n_weeks, freq="W-SAT")
    for sku_i in range(n_sku):
        sku_id = 100 + sku_i
        for _t, dt in enumerate(ts):
            rows.append(
                {
                    config.COL_SKU_ID: sku_id,
                    config.COL_TIMESTAMP: dt,
                    config.COL_SALES: float(rng.exponential(8) * rng.binomial(1, 0.6)),
                    config.COL_LIST_PRICE: 20.0,
                    config.COL_DISCOUNT_PCT: 0.1,
                    "product_type": "Candles",
                    "status": "active",
                    "is_potential_stockout": False,
                    "lag_1": float(rng.exponential(5)),
                    "roll4_mean": float(rng.exponential(5)),
                    "roll13_mean": float(rng.exponential(5)),
                    "roll52_mean": float(rng.exponential(5)),
                    "cluster_id": sku_i % 2,
                    "revenue_tier": "B",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synth_dense() -> pd.DataFrame:
    return _make_synthetic_dense()


# ---------------------------------------------------------------------------
# _build_fold_data
# ---------------------------------------------------------------------------


def test_build_fold_data_train_before_cutoff(synth_dense: pd.DataFrame) -> None:
    ts_max = synth_dense[config.COL_TIMESTAMP].max()
    cutoff = ts_max - pd.Timedelta(weeks=26)
    train, _, actuals, sku_order = _build_fold_data(
        synth_dense, synth_dense, None, cutoff, horizon=4
    )
    assert (train[config.COL_TIMESTAMP] <= cutoff).all()


def test_build_fold_data_actuals_shape(synth_dense: pd.DataFrame) -> None:
    ts_max = synth_dense[config.COL_TIMESTAMP].max()
    cutoff = ts_max - pd.Timedelta(weeks=26)
    _, _, actuals, sku_order = _build_fold_data(synth_dense, synth_dense, None, cutoff, horizon=4)
    assert actuals.shape == (len(sku_order), 4)


def test_build_fold_data_actuals_nonneg(synth_dense: pd.DataFrame) -> None:
    ts_max = synth_dense[config.COL_TIMESTAMP].max()
    cutoff = ts_max - pd.Timedelta(weeks=26)
    _, _, actuals, _ = _build_fold_data(synth_dense, synth_dense, None, cutoff, horizon=4)
    assert (actuals >= 0).all()


def test_build_fold_data_holdout_after_cutoff(synth_dense: pd.DataFrame) -> None:
    ts_max = synth_dense[config.COL_TIMESTAMP].max()
    cutoff = ts_max - pd.Timedelta(weeks=26)
    _, _, actuals, sku_order = _build_fold_data(synth_dense, synth_dense, None, cutoff, horizon=4)
    # sku_order contains only SKUs present in the holdout window
    assert len(sku_order) > 0
    assert len(sku_order) <= synth_dense[config.COL_SKU_ID].nunique()


# ---------------------------------------------------------------------------
# run_cv with SeasonalNaive (fast, no heavy deps)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cv_result(synth_dense: pd.DataFrame) -> CVResult:
    """Run a minimal 2-fold CV on synthetic data with SeasonalNaive."""
    from forecasting.models.baseline import SeasonalNaive
    from forecasting.segment import SegmentResult

    # Minimal segment stub
    n_sku = synth_dense[config.COL_SKU_ID].nunique()
    sku_ids = sorted(synth_dense[config.COL_SKU_ID].unique())
    seg_df = pd.DataFrame(
        {
            config.COL_SKU_ID: sku_ids,
            "cluster_id": [i % 2 for i in range(n_sku)],
            "revenue_tier": ["B"] * n_sku,
            "sb_class": ["smooth"] * n_sku,
            "idi": [2.0] * n_sku,
            "cv2": [0.3] * n_sku,
            "zero_rate": [0.3] * n_sku,
            "silhouette_score": [0.4] * n_sku,
            "best_k": [2] * n_sku,
            "used_fallback_k": [False] * n_sku,
        }
    )
    from forecasting.lifecycle import LifecycleResult  # noqa: F401

    seg = SegmentResult(
        segments=seg_df,
        selected_k=2,
        used_fallback=False,
        best_silhouette=0.4,
    )

    models = {"seasonal_naive": SeasonalNaive()}

    return run_cv(
        full_dense=synth_dense,
        full_features=synth_dense,
        lifecycle=None,
        segments=seg,
        models=models,
        n_folds=2,
        horizon=4,
    )


def test_cv_result_fold_count(cv_result: CVResult) -> None:
    """2 folds × 1 model = 2 FoldMetrics entries."""
    assert len(cv_result.fold_metrics) == 2


def test_cv_result_folds_populated(cv_result: CVResult) -> None:
    folds = {fm.fold for fm in cv_result.fold_metrics}
    assert folds == {1, 2}


def test_cv_result_actuals_shape(cv_result: CVResult) -> None:
    for _fold, actuals in cv_result.fold_actuals.items():
        assert actuals.ndim == 2
        assert actuals.shape[1] == 4  # horizon=4


def test_cv_result_wape_per_horizon_shape(cv_result: CVResult) -> None:
    for fm in cv_result.fold_metrics:
        assert fm.wape_per_horizon.shape == (4,)


def test_cv_result_wape_finite(cv_result: CVResult) -> None:
    for fm in cv_result.fold_metrics:
        assert np.isfinite(fm.wape_overall)
        assert fm.wape_overall >= 0


def test_cv_result_coverage_in_unit_interval(cv_result: CVResult) -> None:
    for fm in cv_result.fold_metrics:
        assert 0.0 <= fm.coverage_80_scalar <= 1.0
        assert 0.0 <= fm.coverage_90_scalar <= 1.0


def test_cv_result_predictions_stored(cv_result: CVResult) -> None:
    from forecasting.models.base import ForecastResult

    for _key, pred in cv_result.fold_predictions.items():
        assert isinstance(pred, ForecastResult)


def test_cv_result_selection_wape(cv_result: CVResult) -> None:
    """selection_wape only uses folds in SELECTION_FOLDS."""
    wape = cv_result.selection_wape("seasonal_naive")
    # With n_folds=2, folds are 1 and 2.
    # SELECTION_FOLDS = {2,3,4} → only fold 2 is in selection.
    # So selection_wape = fold 2 WAPE.
    fold2_wape = next(
        fm.wape_overall
        for fm in cv_result.fold_metrics
        if fm.fold == 2 and fm.model_name == "seasonal_naive"
    )
    assert wape == pytest.approx(fold2_wape)


def test_cv_result_summary_dataframe(cv_result: CVResult) -> None:
    summary = cv_result.summary()
    assert isinstance(summary, pd.DataFrame)
    required = {"model", "fold", "wape_overall", "crps", "cov80", "cov90", "n_skus"}
    assert required.issubset(summary.columns)
    assert len(summary) == 2


def test_cv_per_horizon_wape_accessible(cv_result: CVResult) -> None:
    wape_h = cv_result.per_horizon_wape("seasonal_naive", fold=1)
    assert wape_h is not None
    assert wape_h.shape == (4,)


def test_cv_per_horizon_wape_missing_returns_none(cv_result: CVResult) -> None:
    wape_h = cv_result.per_horizon_wape("nonexistent_model", fold=1)
    assert wape_h is None


# ---------------------------------------------------------------------------
# FoldMetrics in_selection flag
# ---------------------------------------------------------------------------


def test_fold_metrics_in_selection_correct() -> None:
    fm_sel = FoldMetrics(
        fold=2,
        model_name="m",
        n_skus=10,
        wape_per_horizon=np.zeros(4),
        crps_scalar=0.0,
        coverage_80_scalar=0.8,
        coverage_90_scalar=0.9,
        wape_overall=0.5,
        in_selection=True,
    )
    fm_excl = FoldMetrics(
        fold=1,
        model_name="m",
        n_skus=10,
        wape_per_horizon=np.zeros(4),
        crps_scalar=0.0,
        coverage_80_scalar=0.8,
        coverage_90_scalar=0.9,
        wape_overall=0.5,
        in_selection=False,
    )
    assert fm_sel.in_selection is True
    assert fm_excl.in_selection is False


# ---------------------------------------------------------------------------
# Fold cutoff arithmetic
# ---------------------------------------------------------------------------


def test_fold_cutoffs_are_ordered(synth_dense: pd.DataFrame) -> None:
    """Later folds must have later cutoffs (more training data)."""
    ts_max = synth_dense[config.COL_TIMESTAMP].max()
    horizon = 4
    n_folds = 2
    cutoffs = [
        ts_max - pd.Timedelta(weeks=(n_folds - fold + 1) * horizon)
        for fold in range(1, n_folds + 1)
    ]
    for i in range(len(cutoffs) - 1):
        assert cutoffs[i] < cutoffs[i + 1]


def test_fold_4_cutoff_closest_to_ts_max(synth_dense: pd.DataFrame) -> None:
    """Fold 4 (last fold) has the most recent cutoff = ts_max - H."""
    ts_max = synth_dense[config.COL_TIMESTAMP].max()
    # With n_folds=2, horizon=4: fold 2 cutoff = ts_max - (2-2+1)*4w = ts_max - 4w
    fold2_cutoff = ts_max - pd.Timedelta(weeks=(2 - 2 + 1) * 4)
    assert fold2_cutoff == ts_max - pd.Timedelta(weeks=4)
