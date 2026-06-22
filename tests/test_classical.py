"""Phase 9 gate: classical models (AutoETS / AutoARIMA / Theta) + conformal intervals."""

from __future__ import annotations

import numpy as np
import pytest

from forecasting.models.base import ForecastResult
from forecasting.models.classical import (
    AutoARIMAModel,
    AutoETSModel,
    ThetaModel,
    _conformal_quantiles,
)
from forecasting.registry import candidates_for

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

Q_LEVELS = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
HORIZON = 4


def _make_series(n: int = 52, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    trend = np.linspace(5.0, 15.0, n)
    seasonal = 3.0 * np.sin(2 * np.pi * np.arange(n) / 12)
    noise = rng.normal(0, 1.5, n)
    return np.maximum(0.0, trend + seasonal + noise)


@pytest.fixture(scope="module")
def ets_model() -> AutoETSModel:
    series = {
        "sku_a": _make_series(52, seed=1),
        "sku_b": _make_series(60, seed=2),
        "sku_c": _make_series(10, seed=3),  # too short → skipped
    }
    m = AutoETSModel(q_levels=Q_LEVELS, season_length=12)
    m.fit_series(series)
    return m


# ---------------------------------------------------------------------------
# _conformal_quantiles unit tests
# ---------------------------------------------------------------------------


def test_conformal_median_equals_point() -> None:
    """q=0.5 must exactly equal the point forecast."""
    residuals = np.array([1.0, 2.0, 3.0])
    point = np.array([10.0, 12.0, 11.0, 9.0])
    q = np.array([0.1, 0.5, 0.9])
    result = _conformal_quantiles(residuals, point, q)
    np.testing.assert_array_equal(result[:, 1], point)


def test_conformal_upper_above_median() -> None:
    residuals = np.array([2.0, 4.0, 6.0])
    point = np.array([10.0, 10.0])
    q = np.array([0.1, 0.5, 0.9])
    result = _conformal_quantiles(residuals, point, q)
    assert (result[:, 2] >= result[:, 1]).all()  # q90 >= q50


def test_conformal_lower_below_median() -> None:
    residuals = np.array([2.0, 4.0, 6.0])
    point = np.array([10.0, 10.0])
    q = np.array([0.1, 0.5, 0.9])
    result = _conformal_quantiles(residuals, point, q)
    assert (result[:, 0] <= result[:, 1]).all()  # q10 <= q50


def test_conformal_lower_floored_at_zero() -> None:
    residuals = np.array([50.0, 60.0, 70.0])  # very wide intervals
    point = np.array([5.0])  # point near 0
    q = np.array([0.1, 0.5, 0.9])
    result = _conformal_quantiles(residuals, point, q)
    assert result[:, 0] >= 0.0  # lower cannot go negative


def test_conformal_no_residuals_returns_point() -> None:
    residuals = np.array([])
    point = np.array([7.0, 8.0])
    q = np.array([0.1, 0.5, 0.9])
    result = _conformal_quantiles(residuals, point, q)
    # All quantiles equal point forecast when no calibration data
    for qi in range(3):
        np.testing.assert_array_equal(result[:, qi], point)


def test_conformal_output_shape() -> None:
    H, n_q = 6, len(Q_LEVELS)
    residuals = np.abs(np.random.default_rng(0).normal(0, 2, 20))
    point = np.ones(H) * 10
    result = _conformal_quantiles(residuals, point, Q_LEVELS)
    assert result.shape == (H, n_q)


def test_conformal_sorted_quantiles() -> None:
    residuals = np.abs(np.random.default_rng(1).normal(0, 3, 30))
    point = np.random.default_rng(2).random(5) * 20
    result = _conformal_quantiles(residuals, point, Q_LEVELS)
    diffs = np.diff(result, axis=1)
    assert (diffs >= 0).all(), "Quantiles should be non-decreasing"


# ---------------------------------------------------------------------------
# AutoETS model tests
# ---------------------------------------------------------------------------


def test_ets_is_fitted(ets_model: AutoETSModel) -> None:
    assert ets_model.is_fitted


def test_ets_skips_short_series(ets_model: AutoETSModel) -> None:
    assert "sku_c" in ets_model._skipped_skus


def test_ets_predict_returns_forecast_result(ets_model: AutoETSModel) -> None:
    result = ets_model.predict(np.empty(0), HORIZON)
    assert isinstance(result, ForecastResult)


def test_ets_predict_shape(ets_model: AutoETSModel) -> None:
    result = ets_model.predict(np.empty(0), HORIZON)
    assert result.quantiles.shape == (3, HORIZON, len(Q_LEVELS))


def test_ets_predict_non_negative(ets_model: AutoETSModel) -> None:
    result = ets_model.predict(np.empty(0), HORIZON)
    assert (result.quantiles >= 0).all()


def test_ets_predict_sorted(ets_model: AutoETSModel) -> None:
    result = ets_model.predict(np.empty(0), HORIZON)
    diffs = np.diff(result.quantiles, axis=2)
    assert (diffs >= 0).all()


def test_ets_skipped_sku_zero_forecast(ets_model: AutoETSModel) -> None:
    result = ets_model.predict(np.empty(0), HORIZON)
    uid_order = sorted(ets_model._sku_series.keys())
    c_idx = uid_order.index("sku_c")
    np.testing.assert_array_equal(result.quantiles[c_idx], 0.0)


def test_ets_conformal_intervals_wider_for_noisy_series() -> None:
    """SKU with higher noise should get wider conformal intervals."""
    rng = np.random.default_rng(77)
    trend = np.linspace(5.0, 10.0, 52)

    y_low_noise = np.maximum(0, trend + rng.normal(0, 0.5, 52))
    y_high_noise = np.maximum(0, trend + rng.normal(0, 5.0, 52))

    m = AutoETSModel(q_levels=Q_LEVELS, season_length=12)
    m.fit_series({"low": y_low_noise, "high": y_high_noise})
    result = m.predict(np.empty(0), horizon=4)

    uid_order = sorted(m._sku_series.keys())
    high_idx = uid_order.index("high")
    low_idx = uid_order.index("low")

    width_high = (result.quantile_at(0.90) - result.quantile_at(0.10))[high_idx].mean()
    width_low = (result.quantile_at(0.90) - result.quantile_at(0.10))[low_idx].mean()
    assert width_high > width_low


# ---------------------------------------------------------------------------
# AutoARIMA model tests (lightweight — reuse same logic as ETS)
# ---------------------------------------------------------------------------


def test_arima_fits_and_predicts() -> None:
    series = {"sku1": _make_series(40, seed=5), "sku2": _make_series(30, seed=6)}
    m = AutoARIMAModel(q_levels=Q_LEVELS, season_length=12)
    m.fit_series(series)
    result = m.predict(np.empty(0), horizon=4)
    assert isinstance(result, ForecastResult)
    assert result.quantiles.shape == (2, 4, len(Q_LEVELS))
    assert (result.quantiles >= 0).all()


# ---------------------------------------------------------------------------
# Theta model tests
# ---------------------------------------------------------------------------


def test_theta_fits_and_predicts() -> None:
    series = {"sku1": _make_series(52, seed=9), "sku2": _make_series(52, seed=10)}
    m = ThetaModel(q_levels=Q_LEVELS, season_length=12)
    m.fit_series(series)
    result = m.predict(np.empty(0), horizon=4)
    assert isinstance(result, ForecastResult)
    assert result.quantiles.shape == (2, 4, len(Q_LEVELS))


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_ets_registered_for_smooth() -> None:
    assert AutoETSModel in candidates_for("smooth")


def test_ets_registered_for_erratic() -> None:
    assert AutoETSModel in candidates_for("erratic")


def test_arima_registered_for_smooth() -> None:
    assert AutoARIMAModel in candidates_for("smooth")


def test_theta_registered_for_intermittent() -> None:
    assert ThetaModel in candidates_for("intermittent")


def test_classical_models_not_registered_for_lumpy() -> None:
    # Lumpy demand → intermittent/tweedie models, not classical ETS/ARIMA
    lumpy_candidates = candidates_for("lumpy")
    assert AutoETSModel not in lumpy_candidates
    assert AutoARIMAModel not in lumpy_candidates


# ---------------------------------------------------------------------------
# Skip rule: < 26 weeks
# ---------------------------------------------------------------------------


def test_short_series_skipped() -> None:
    m = AutoETSModel(q_levels=Q_LEVELS, season_length=12)
    m.fit_series(
        {
            "long": _make_series(52),
            "short": _make_series(10),  # < 26 weeks
        }
    )
    assert "short" in m._skipped_skus
    assert "long" not in m._skipped_skus


def test_all_short_series_still_returns_zeros() -> None:
    m = AutoETSModel(q_levels=Q_LEVELS, season_length=12)
    m.fit_series({"tiny": np.array([1.0, 2.0, 3.0])})
    result = m.predict(np.empty(0), horizon=4)
    assert result.quantiles.shape == (1, 4, len(Q_LEVELS))
    np.testing.assert_array_equal(result.quantiles, 0.0)
