"""Phase 10 gate: intermittent / lumpy demand models."""

from __future__ import annotations

import numpy as np
import pytest

from forecasting.models.base import ForecastResult
from forecasting.models.intermittent import (
    CompoundBernoulliModel,
    CrostonSBAModel,
    TSBModel,
    _fit_compound_bernoulli,
)
from forecasting.registry import candidates_for

Q_LEVELS = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
HORIZON = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lumpy_series(n: int = 52, p: float = 0.3, seed: int = 0) -> np.ndarray:
    """Lumpy: sparse + high-variance demand."""
    rng = np.random.default_rng(seed)
    return np.where(rng.random(n) < p, rng.exponential(20, n), 0.0)


def _intermittent_series(n: int = 52, p: float = 0.4, seed: int = 0) -> np.ndarray:
    """Intermittent: sparse + low-variance demand size."""
    rng = np.random.default_rng(seed)
    return np.where(rng.random(n) < p, rng.normal(5, 1, n).clip(0), 0.0)


# ---------------------------------------------------------------------------
# _fit_compound_bernoulli unit tests
# ---------------------------------------------------------------------------


def test_fit_cb_returns_valid_p() -> None:
    y = _lumpy_series(60, p=0.35, seed=1)
    p, shape, scale = _fit_compound_bernoulli(y)
    assert 0 <= p <= 1


def test_fit_cb_shape_scale_positive() -> None:
    y = _lumpy_series(60, p=0.4, seed=2)
    p, shape, scale = _fit_compound_bernoulli(y)
    assert shape > 0
    assert scale > 0


def test_fit_cb_all_zeros() -> None:
    y = np.zeros(50)
    p, shape, scale = _fit_compound_bernoulli(y)
    assert p == pytest.approx(0.0)


def test_fit_cb_degenerate_single_nonzero() -> None:
    y = np.zeros(50)
    y[10] = 5.0
    p, shape, scale = _fit_compound_bernoulli(y)
    # shape=1.0 (degenerate), scale=5.0
    assert shape == pytest.approx(1.0)
    assert scale == pytest.approx(5.0)


def test_fit_cb_p_reflects_sparsity() -> None:
    rng = np.random.default_rng(99)
    y_sparse = np.where(rng.random(100) < 0.2, rng.exponential(10, 100), 0.0)
    y_dense = np.where(rng.random(100) < 0.8, rng.exponential(10, 100), 0.0)
    p_sparse, _, _ = _fit_compound_bernoulli(y_sparse)
    p_dense, _, _ = _fit_compound_bernoulli(y_dense)
    assert p_sparse < p_dense


# ---------------------------------------------------------------------------
# CompoundBernoulli model
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cb_model() -> CompoundBernoulliModel:
    series = {
        "lumpy_a": _lumpy_series(52, seed=3),
        "lumpy_b": _lumpy_series(60, seed=4),
        "tiny": np.array([1.0, 0.0, 2.0]),  # too short / too few nonzero → skipped
    }
    m = CompoundBernoulliModel(q_levels=Q_LEVELS, n_samples=200, random_seed=7)
    m.fit_series(series)
    return m


def test_cb_fitted(cb_model: CompoundBernoulliModel) -> None:
    assert cb_model.is_fitted


def test_cb_skips_tiny(cb_model: CompoundBernoulliModel) -> None:
    assert "tiny" in cb_model._skipped_skus


def test_cb_predict_shape(cb_model: CompoundBernoulliModel) -> None:
    result = cb_model.predict(np.empty(0), HORIZON)
    assert result.quantiles.shape == (3, HORIZON, len(Q_LEVELS))


def test_cb_predict_non_negative(cb_model: CompoundBernoulliModel) -> None:
    result = cb_model.predict(np.empty(0), HORIZON)
    assert (result.quantiles >= 0).all()


def test_cb_predict_sorted(cb_model: CompoundBernoulliModel) -> None:
    result = cb_model.predict(np.empty(0), HORIZON)
    diffs = np.diff(result.quantiles, axis=2)
    assert (diffs >= 0).all()


def test_cb_skipped_sku_zero_forecast(cb_model: CompoundBernoulliModel) -> None:
    result = cb_model.predict(np.empty(0), HORIZON)
    uid_order = sorted(cb_model._sku_series.keys())
    idx = uid_order.index("tiny")
    np.testing.assert_array_equal(result.quantiles[idx], 0.0)


def test_cb_higher_p_raises_p90() -> None:
    """SKU with higher demand probability should have higher P90."""
    m = CompoundBernoulliModel(q_levels=Q_LEVELS, n_samples=1000, random_seed=42)
    rng = np.random.default_rng(5)
    y_sparse = np.where(rng.random(60) < 0.15, rng.exponential(10, 60), 0.0)
    y_dense = np.where(rng.random(60) < 0.80, rng.exponential(10, 60), 0.0)
    m.fit_series({"sparse": y_sparse, "dense": y_dense})
    result = m.predict(np.empty(0), horizon=8)
    uid_order = sorted(m._sku_series.keys())
    dense_idx = uid_order.index("dense")
    sparse_idx = uid_order.index("sparse")
    p90_dense = result.quantile_at(0.90)[dense_idx].mean()
    p90_sparse = result.quantile_at(0.90)[sparse_idx].mean()
    assert p90_dense > p90_sparse


def test_cb_returns_forecast_result(cb_model: CompoundBernoulliModel) -> None:
    result = cb_model.predict(np.empty(0), HORIZON)
    assert isinstance(result, ForecastResult)


# ---------------------------------------------------------------------------
# CrostonSBA model
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def croston_model() -> CrostonSBAModel:
    series = {
        "int_a": _intermittent_series(52, seed=10),
        "int_b": _intermittent_series(60, seed=11),
        "tiny": np.array([0.0, 1.0, 0.0]),
    }
    m = CrostonSBAModel(q_levels=Q_LEVELS)
    m.fit_series(series)
    return m


def test_croston_fitted(croston_model: CrostonSBAModel) -> None:
    assert croston_model.is_fitted


def test_croston_skips_tiny(croston_model: CrostonSBAModel) -> None:
    assert "tiny" in croston_model._skipped_skus


def test_croston_predict_shape(croston_model: CrostonSBAModel) -> None:
    result = croston_model.predict(np.empty(0), HORIZON)
    assert result.quantiles.shape == (3, HORIZON, len(Q_LEVELS))


def test_croston_predict_non_negative(croston_model: CrostonSBAModel) -> None:
    result = croston_model.predict(np.empty(0), HORIZON)
    assert (result.quantiles >= 0).all()


def test_croston_predict_sorted(croston_model: CrostonSBAModel) -> None:
    result = croston_model.predict(np.empty(0), HORIZON)
    diffs = np.diff(result.quantiles, axis=2)
    assert (diffs >= 0).all()


def test_croston_returns_forecast_result(croston_model: CrostonSBAModel) -> None:
    result = croston_model.predict(np.empty(0), HORIZON)
    assert isinstance(result, ForecastResult)


# ---------------------------------------------------------------------------
# TSB model
# ---------------------------------------------------------------------------


def test_tsb_fits_and_predicts() -> None:
    series = {
        "lump_a": _lumpy_series(52, seed=20),
        "lump_b": _lumpy_series(52, seed=21),
    }
    m = TSBModel(q_levels=Q_LEVELS, alpha_d=0.1, alpha_p=0.1)
    m.fit_series(series)
    result = m.predict(np.empty(0), HORIZON)
    assert isinstance(result, ForecastResult)
    assert result.quantiles.shape == (2, HORIZON, len(Q_LEVELS))
    assert (result.quantiles >= 0).all()


def test_tsb_predict_sorted() -> None:
    series = {"sku1": _lumpy_series(52, seed=30)}
    m = TSBModel(q_levels=Q_LEVELS)
    m.fit_series(series)
    result = m.predict(np.empty(0), HORIZON)
    diffs = np.diff(result.quantiles, axis=2)
    assert (diffs >= 0).all()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_croston_registered_for_intermittent() -> None:
    assert CrostonSBAModel in candidates_for("intermittent")


def test_tsb_registered_for_intermittent() -> None:
    assert TSBModel in candidates_for("intermittent")


def test_tsb_registered_for_lumpy() -> None:
    assert TSBModel in candidates_for("lumpy")


def test_cb_registered_for_lumpy() -> None:
    assert CompoundBernoulliModel in candidates_for("lumpy")


def test_cb_registered_for_intermittent() -> None:
    assert CompoundBernoulliModel in candidates_for("intermittent")


def test_croston_not_registered_for_smooth() -> None:
    assert CrostonSBAModel not in candidates_for("smooth")
