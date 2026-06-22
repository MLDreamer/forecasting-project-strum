"""Phase 11 gate: ClusterPooledLGBM — 152 boosters, importance, floor+sort."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting import config
from forecasting.models.base import ForecastResult
from forecasting.models.ml_global import (
    CAT_FEATURES,
    N_BOOSTERS,
    ClusterPooledLGBM,
    LGBMFitResult,
    build_training_rows,
)
from forecasting.registry import candidates_for

# ---------------------------------------------------------------------------
# Minimal synthetic fixture (no real data load — unit tests only)
# ---------------------------------------------------------------------------

Q_LEVELS = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
N_Q = len(Q_LEVELS)
HORIZON = 4
N_SKU = 12
N_WEEKS = 60
N_CLUSTERS = 4  # small for speed


def _make_features_df(n_sku: int = N_SKU, n_weeks: int = N_WEEKS, seed: int = 0) -> pd.DataFrame:
    """Synthetic feature grid with the minimum required columns."""
    rng = np.random.default_rng(seed)
    rows = []
    ts = pd.date_range("2023-01-07", periods=n_weeks, freq="W-SAT")
    product_types = ["Candles", "Wax Melts", "Accessories"]
    statuses = ["active", "draft"]

    for sku_i in range(n_sku):
        sku_id = 1000 + sku_i
        for t, dt in enumerate(ts):
            rows.append(
                {
                    config.COL_SKU_ID: sku_id,
                    config.COL_TIMESTAMP: dt,
                    config.COL_SALES: float(rng.exponential(10) * rng.binomial(1, 0.6)),
                    config.COL_LIST_PRICE: 20.0 + rng.normal(0, 2),
                    config.COL_DISCOUNT_PCT: rng.random() * 0.3,
                    "product_type": product_types[sku_i % 3],
                    "status": statuses[sku_i % 2],
                    "is_potential_stockout": False,
                    # Minimal feature cols
                    "lag_1": float(rng.exponential(8) * rng.binomial(1, 0.6)),
                    "lag_4": float(rng.exponential(8) * rng.binomial(1, 0.6)),
                    "roll4_mean": float(rng.exponential(8)),
                    "roll13_mean": float(rng.exponential(8)),
                    "roll52_mean": float(rng.exponential(8)),
                    "fourier_52w_sin_k1": float(np.sin(2 * np.pi * t / 52)),
                    "week_of_year": float(dt.isocalendar().week),
                    "idi": float(rng.uniform(1, 5)),
                    "cv2": float(rng.uniform(0, 2)),
                    "cluster_id": int(sku_i % N_CLUSTERS),
                    "revenue_tier": ["A", "B", "C"][sku_i % 3],
                }
            )
    return pd.DataFrame(rows)


def _make_segments_df(n_sku: int = N_SKU) -> pd.DataFrame:
    return pd.DataFrame(
        {
            config.COL_SKU_ID: [1000 + i for i in range(n_sku)],
            "cluster_id": [i % N_CLUSTERS for i in range(n_sku)],
            "revenue_tier": [["A", "B", "C"][i % 3] for i in range(n_sku)],
            "sb_class": ["smooth"] * n_sku,
        }
    )


@pytest.fixture(scope="module")
def fitted_model() -> ClusterPooledLGBM:
    feat_df = _make_features_df()
    seg_df = _make_segments_df()
    m = ClusterPooledLGBM(
        q_levels=Q_LEVELS,
        horizon=HORIZON,
        lgbm_params={
            "objective": "quantile",
            "metric": "quantile",
            "n_estimators": 20,  # fast for tests
            "learning_rate": 0.1,
            "num_leaves": 8,
            "verbose": -1,
            "random_state": 42,
        },
        early_stopping_rounds=5,
    )
    m.fit_dataframe(feat_df, seg_df)
    return m


# ---------------------------------------------------------------------------
# Architecture constants
# ---------------------------------------------------------------------------


def test_n_boosters_constant() -> None:
    assert N_BOOSTERS == 152


def test_cat_features_list() -> None:
    assert "sku_id" in CAT_FEATURES
    assert "cluster_id" in CAT_FEATURES
    assert "revenue_tier" in CAT_FEATURES


# ---------------------------------------------------------------------------
# build_training_rows
# ---------------------------------------------------------------------------


def test_build_training_rows_has_target() -> None:
    feat_df = _make_features_df(n_sku=4, n_weeks=30)
    seg_df = _make_segments_df(n_sku=4)
    train = build_training_rows(feat_df, seg_df, horizon=4)
    assert "target" in train.columns
    assert "horizon_step" in train.columns


def test_build_training_rows_horizon_steps() -> None:
    """horizon_step must range from 1..horizon."""
    feat_df = _make_features_df(n_sku=4, n_weeks=30)
    seg_df = _make_segments_df(n_sku=4)
    train = build_training_rows(feat_df, seg_df, horizon=4)
    steps = sorted(train["horizon_step"].unique())
    assert steps == [1.0, 2.0, 3.0, 4.0]


def test_build_training_rows_nonneg_target() -> None:
    feat_df = _make_features_df(n_sku=4, n_weeks=30)
    seg_df = _make_segments_df(n_sku=4)
    train = build_training_rows(feat_df, seg_df, horizon=4)
    assert (train["target"] >= 0).all()


def test_build_training_rows_cluster_id_present() -> None:
    feat_df = _make_features_df(n_sku=4, n_weeks=30)
    seg_df = _make_segments_df(n_sku=4)
    train = build_training_rows(feat_df, seg_df, horizon=4)
    assert "cluster_id" in train.columns


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------


def test_model_is_fitted(fitted_model: ClusterPooledLGBM) -> None:
    assert fitted_model.is_fitted


def test_n_boosters_fitted(fitted_model: ClusterPooledLGBM) -> None:
    """With 4 clusters × 7 quantiles = 28 boosters for our small fixture."""
    assert fitted_model.n_boosters_fitted == N_CLUSTERS * N_Q


def test_fit_result_not_none(fitted_model: ClusterPooledLGBM) -> None:
    assert fitted_model._fit_result is not None
    assert isinstance(fitted_model._fit_result, LGBMFitResult)


def test_boosters_dict_keys(fitted_model: ClusterPooledLGBM) -> None:
    """Keys must be (cluster_id, quantile) tuples."""
    for key in fitted_model._fit_result.boosters:
        assert isinstance(key, tuple)
        assert len(key) == 2
        cluster_id, q = key
        assert isinstance(cluster_id, int | np.integer)
        assert 0.0 <= q <= 1.0


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


def test_importance_not_none(fitted_model: ClusterPooledLGBM) -> None:
    imp = fitted_model.feature_importance
    assert imp is not None
    assert len(imp) > 0


def test_importance_columns(fitted_model: ClusterPooledLGBM) -> None:
    imp = fitted_model.feature_importance
    assert isinstance(imp, pd.DataFrame)
    required = {"feature", "cluster_id", "q", "importance"}
    assert required.issubset(imp.columns)


def test_importance_nonneg(fitted_model: ClusterPooledLGBM) -> None:
    assert (fitted_model.feature_importance["importance"] >= 0).all()


def test_importance_covers_all_boosters(fitted_model: ClusterPooledLGBM) -> None:
    """Importance DataFrame must have entries for every (cluster, q) fitted."""
    imp = fitted_model.feature_importance
    combos = set(zip(imp["cluster_id"], imp["q"], strict=False))
    expected = set(fitted_model._fit_result.boosters.keys())
    assert combos == expected


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


def test_predict_dataframe_returns_forecast_result(fitted_model: ClusterPooledLGBM) -> None:
    feat_df = _make_features_df()
    seg_df = _make_segments_df()
    result = fitted_model.predict_dataframe(feat_df, seg_df, horizon=HORIZON)
    assert isinstance(result, ForecastResult)


def test_predict_shape(fitted_model: ClusterPooledLGBM) -> None:
    feat_df = _make_features_df()
    seg_df = _make_segments_df()
    result = fitted_model.predict_dataframe(feat_df, seg_df, horizon=HORIZON)
    assert result.quantiles.shape == (N_SKU, HORIZON, N_Q)


def test_predict_nonneg(fitted_model: ClusterPooledLGBM) -> None:
    feat_df = _make_features_df()
    seg_df = _make_segments_df()
    result = fitted_model.predict_dataframe(feat_df, seg_df, horizon=HORIZON)
    assert (result.quantiles >= 0).all()


def test_predict_sorted(fitted_model: ClusterPooledLGBM) -> None:
    """After ForecastResult._finalize, quantiles must be non-decreasing."""
    feat_df = _make_features_df()
    seg_df = _make_segments_df()
    result = fitted_model.predict_dataframe(feat_df, seg_df, horizon=HORIZON)
    diffs = np.diff(result.quantiles, axis=2)
    assert (diffs >= 0).all()


# ---------------------------------------------------------------------------
# Floor + sort invariants (the 29% crossing finding)
# ---------------------------------------------------------------------------


def test_crossing_repaired_by_finalize() -> None:
    """Even if raw boosters cross, ForecastResult.from_quantiles must repair it."""
    # Simulate raw crossed output
    raw = np.array([[[10.0, 8.0, 12.0, 9.0, 15.0, 11.0, 20.0]]])  # (1,1,7)
    q_levels = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    result = ForecastResult.from_quantiles(raw, q_levels)
    diffs = np.diff(result.quantiles, axis=2)
    assert (diffs >= 0).all(), "Crossing not repaired"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_lgbm_registered_for_smooth() -> None:
    assert ClusterPooledLGBM in candidates_for("smooth")


def test_lgbm_registered_for_cold_start() -> None:
    assert ClusterPooledLGBM in candidates_for("cold_start")


def test_lgbm_registered_for_lumpy() -> None:
    assert ClusterPooledLGBM in candidates_for("lumpy")


# ---------------------------------------------------------------------------
# predict() raises NotImplementedError (use predict_dataframe instead)
# ---------------------------------------------------------------------------


def test_predict_raises_not_implemented(fitted_model: ClusterPooledLGBM) -> None:
    with pytest.raises(NotImplementedError):
        fitted_model.predict(np.empty(0), horizon=4)
