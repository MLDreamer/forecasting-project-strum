"""Baseline models gate: SeasonalNaive + ZeroForecast + TrendSeasonal + RecentLevel."""

from __future__ import annotations

import numpy as np
import pytest

from forecasting.models.base import ForecastResult
from forecasting.models.baseline import (
    RecentLevelModel,
    SeasonalNaive,
    TrendSeasonalModel,
    ZeroForecast,
)
from forecasting.registry import ALL_SEGMENTS, candidates_for

Q_LEVELS = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
HORIZON = 4


def _smooth(n: int = 60, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    trend = np.linspace(5.0, 12.0, n)
    return np.maximum(0.0, trend + rng.normal(0, 1.5, n))


def _short(n: int = 10, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.maximum(0.0, rng.normal(8.0, 2.0, n))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_seasonal_naive_registered_for_all_segments() -> None:
    """SeasonalNaive must cover every segment — it is the universal fallback."""
    import forecasting.models.baseline  # trigger registration  # noqa: F401

    for seg in ALL_SEGMENTS:
        assert SeasonalNaive in candidates_for(seg), f"SeasonalNaive missing for {seg}"


def test_zero_forecast_registered_for_discontinued() -> None:
    import forecasting.models.baseline  # noqa: F401

    assert ZeroForecast in candidates_for("discontinued")


def test_zero_forecast_not_registered_for_smooth() -> None:
    import forecasting.models.baseline  # noqa: F401

    assert ZeroForecast not in candidates_for("smooth")


# ---------------------------------------------------------------------------
# ZeroForecast
# ---------------------------------------------------------------------------


def test_zero_forecast_predict_all_zeros() -> None:
    m = ZeroForecast(q_levels=Q_LEVELS)
    m.fit_series({"sku1": np.ones(30), "sku2": np.ones(30)})
    result = m.predict(np.empty(0), HORIZON)
    np.testing.assert_array_equal(result.quantiles, 0.0)


def test_zero_forecast_shape() -> None:
    m = ZeroForecast(q_levels=Q_LEVELS)
    m.fit_series({"a": np.ones(10), "b": np.ones(10), "c": np.ones(10)})
    result = m.predict(np.empty(0), HORIZON)
    assert result.quantiles.shape == (3, HORIZON, len(Q_LEVELS))


def test_zero_forecast_returns_forecast_result() -> None:
    m = ZeroForecast(q_levels=Q_LEVELS)
    m.fit_series({"x": np.ones(5)})
    result = m.predict(np.empty(0), HORIZON)
    assert isinstance(result, ForecastResult)


# ---------------------------------------------------------------------------
# SeasonalNaive — shape + non-negativity + sort
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sn_model() -> SeasonalNaive:
    series = {
        "long_sku": _smooth(n=70, seed=1),  # > 52 weeks → uses last-year template
        "short_sku": _short(n=20, seed=2),  # < 52 weeks → uses mean template
    }
    m = SeasonalNaive(q_levels=Q_LEVELS)
    m.fit_series(series)
    return m


def test_sn_is_fitted(sn_model: SeasonalNaive) -> None:
    assert sn_model.is_fitted


def test_sn_predict_shape(sn_model: SeasonalNaive) -> None:
    result = sn_model.predict(np.empty(0), HORIZON)
    assert result.quantiles.shape == (2, HORIZON, len(Q_LEVELS))


def test_sn_predict_non_negative(sn_model: SeasonalNaive) -> None:
    result = sn_model.predict(np.empty(0), HORIZON)
    assert (result.quantiles >= 0).all()


def test_sn_predict_sorted(sn_model: SeasonalNaive) -> None:
    result = sn_model.predict(np.empty(0), HORIZON)
    diffs = np.diff(result.quantiles, axis=2)
    assert (diffs >= 0).all()


def test_sn_returns_forecast_result(sn_model: SeasonalNaive) -> None:
    result = sn_model.predict(np.empty(0), HORIZON)
    assert isinstance(result, ForecastResult)


# ---------------------------------------------------------------------------
# SeasonalNaive — P50 equals point forecast
# ---------------------------------------------------------------------------


def test_sn_p50_equals_point_forecast() -> None:
    """q=0.5 must be the point forecast (no interval shift)."""
    y = _smooth(70, seed=5)
    m = SeasonalNaive(q_levels=Q_LEVELS)
    m.fit_series({"sku": y})
    result = m.predict(np.empty(0), HORIZON)

    # Reconstruct expected point forecast from last-year template
    template = y[-52:]
    expected_point = np.array([template[(h - 1) % 52] for h in range(1, HORIZON + 1)])
    expected_point = np.maximum(0.0, expected_point)

    p50_idx = int(np.argmin(np.abs(Q_LEVELS - 0.5)))
    actual_p50 = result.quantiles[0, :, p50_idx]
    np.testing.assert_allclose(actual_p50, expected_point, rtol=1e-6)


# ---------------------------------------------------------------------------
# SeasonalNaive — upper quantile >= P50, lower <= P50
# ---------------------------------------------------------------------------


def test_sn_upper_above_p50(sn_model: SeasonalNaive) -> None:
    result = sn_model.predict(np.empty(0), HORIZON)
    p50_idx = int(np.argmin(np.abs(Q_LEVELS - 0.5)))
    p90_idx = int(np.argmin(np.abs(Q_LEVELS - 0.90)))
    assert (result.quantiles[:, :, p90_idx] >= result.quantiles[:, :, p50_idx]).all()


def test_sn_lower_below_p50(sn_model: SeasonalNaive) -> None:
    result = sn_model.predict(np.empty(0), HORIZON)
    p50_idx = int(np.argmin(np.abs(Q_LEVELS - 0.5)))
    p10_idx = int(np.argmin(np.abs(Q_LEVELS - 0.10)))
    assert (result.quantiles[:, :, p10_idx] <= result.quantiles[:, :, p50_idx]).all()


# ---------------------------------------------------------------------------
# SeasonalNaive — short-history fallback uses mean
# ---------------------------------------------------------------------------


def test_sn_short_history_uses_constant_mean() -> None:
    """Short series (<52w): all point forecasts are the series mean."""
    y = np.full(10, 7.0)  # constant series, mean=7
    m = SeasonalNaive(q_levels=Q_LEVELS)
    m.fit_series({"sku": y})
    result = m.predict(np.empty(0), 4)

    p50_idx = int(np.argmin(np.abs(Q_LEVELS - 0.5)))
    p50 = result.quantiles[0, :, p50_idx]
    np.testing.assert_allclose(p50, 7.0, rtol=1e-6)


# ---------------------------------------------------------------------------
# SeasonalNaive — interval width > 0 for noisy series
# ---------------------------------------------------------------------------


def test_sn_noisy_series_has_nonzero_intervals() -> None:
    rng = np.random.default_rng(77)
    y = np.maximum(0.0, np.linspace(5.0, 15.0, 60) + rng.normal(0, 4.0, 60))
    m = SeasonalNaive(q_levels=Q_LEVELS)
    m.fit_series({"noisy": y})
    result = m.predict(np.empty(0), HORIZON)
    p10_idx = int(np.argmin(np.abs(Q_LEVELS - 0.10)))
    p90_idx = int(np.argmin(np.abs(Q_LEVELS - 0.90)))
    width = result.quantiles[0, :, p90_idx] - result.quantiles[0, :, p10_idx]
    assert (width > 0).all(), "Noisy series should have non-zero PI width"


# ---------------------------------------------------------------------------
# SeasonalNaive — empty series skipped
# ---------------------------------------------------------------------------


def test_sn_empty_series_skipped() -> None:
    m = SeasonalNaive(q_levels=Q_LEVELS)
    m.fit_series({"empty": np.array([]), "ok": _smooth(60)})
    assert "empty" in m._skipped_skus


# ---------------------------------------------------------------------------
# TrendSeasonalModel (erratic segment — S1.3)
# ---------------------------------------------------------------------------


def test_trend_seasonal_shape() -> None:
    y = _smooth(70, seed=10)
    m = TrendSeasonalModel(q_levels=Q_LEVELS)
    m.fit_series({"sku": y})
    result = m.predict(np.empty(0), HORIZON)
    assert result.quantiles.shape == (1, HORIZON, len(Q_LEVELS))


def test_trend_seasonal_nonneg() -> None:
    y = _smooth(70, seed=11)
    m = TrendSeasonalModel(q_levels=Q_LEVELS)
    m.fit_series({"sku": y})
    result = m.predict(np.empty(0), HORIZON)
    assert (result.quantiles >= 0).all()


def test_trend_seasonal_sorted() -> None:
    y = _smooth(70, seed=12)
    m = TrendSeasonalModel(q_levels=Q_LEVELS)
    m.fit_series({"sku": y})
    result = m.predict(np.empty(0), HORIZON)
    assert (np.diff(result.quantiles, axis=2) >= 0).all()


def test_trend_seasonal_stockout_gate() -> None:
    """Dead SKU (last 8 weeks zero) → near-zero forecast."""
    rng = np.random.default_rng(13)
    y = np.concatenate([rng.exponential(10, 60), np.zeros(10)])  # dies at end
    m = TrendSeasonalModel(q_levels=Q_LEVELS)
    m.fit_series({"dead": y})
    result = m.predict(np.empty(0), HORIZON)
    p50_idx = len(Q_LEVELS) // 2
    # Near-zero: P50 should be very small
    assert result.quantiles[0, :, p50_idx].mean() < 5.0


def test_trend_seasonal_growth_clip() -> None:
    """YoY growth multiplier is clipped to [0.5, 3.0]."""
    # Series with explosive 10× growth in last 13 weeks
    y = np.concatenate([np.ones(52) * 5.0, np.ones(13) * 50.0])
    m = TrendSeasonalModel(q_levels=Q_LEVELS)
    m.fit_series({"explosive": y})
    # Forecast P50 should be capped: base * 3.0, not base * 10
    result = m.predict(np.empty(0), 4)
    p50 = result.quantiles[0, :, len(Q_LEVELS) // 2]
    # Base is ~5, growth clip at 3.0 → P50 ≤ 5 * 3.0 + some residual
    assert p50.mean() <= 20.0, f"P50 too high: {p50.mean():.2f} — growth clip may be broken"


def test_trend_seasonal_registered_for_erratic() -> None:
    import forecasting.models.baseline  # noqa: F401

    assert TrendSeasonalModel in candidates_for("erratic")


# ---------------------------------------------------------------------------
# RecentLevelModel (smooth segment — S1.3)
# ---------------------------------------------------------------------------


def test_recent_level_shape() -> None:
    y = _smooth(60, seed=20)
    m = RecentLevelModel(q_levels=Q_LEVELS)
    m.fit_series({"sku": y})
    result = m.predict(np.empty(0), HORIZON)
    assert result.quantiles.shape == (1, HORIZON, len(Q_LEVELS))


def test_recent_level_constant_p50() -> None:
    """P50 should be the 8-week mean (constant across horizon)."""
    y = np.full(60, 7.0)
    m = RecentLevelModel(q_levels=Q_LEVELS)
    m.fit_series({"sku": y})
    result = m.predict(np.empty(0), HORIZON)
    p50_idx = len(Q_LEVELS) // 2
    p50 = result.quantiles[0, :, p50_idx]
    np.testing.assert_allclose(p50, 7.0, rtol=1e-5)


def test_recent_level_dead_sku_near_zero() -> None:
    """Series that ended with 8 zeros → 8-week mean ≈ 0 → near-zero forecast."""
    y = np.concatenate([np.ones(52) * 10.0, np.zeros(8)])
    m = RecentLevelModel(q_levels=Q_LEVELS)
    m.fit_series({"dead": y})
    result = m.predict(np.empty(0), HORIZON)
    p50_idx = len(Q_LEVELS) // 2
    assert result.quantiles[0, :, p50_idx].mean() == pytest.approx(0.0, abs=1e-6)


def test_recent_level_nonneg() -> None:
    rng = np.random.default_rng(21)
    y = rng.exponential(10, 50)
    m = RecentLevelModel(q_levels=Q_LEVELS)
    m.fit_series({"sku": y})
    result = m.predict(np.empty(0), HORIZON)
    assert (result.quantiles >= 0).all()


def test_recent_level_sorted() -> None:
    y = _smooth(60, seed=22)
    m = RecentLevelModel(q_levels=Q_LEVELS)
    m.fit_series({"sku": y})
    result = m.predict(np.empty(0), HORIZON)
    assert (np.diff(result.quantiles, axis=2) >= 0).all()


def test_recent_level_registered_for_smooth() -> None:
    import forecasting.models.baseline  # noqa: F401

    assert RecentLevelModel in candidates_for("smooth")
