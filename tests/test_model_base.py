"""Phase 8 gate: ForecastResult dual constructor — from_quantiles and from_samples
both floor at 0 and sort along the quantile axis."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from forecasting.models.base import (
    ForecastModel,
    ForecastResult,
    _finalize,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

Q_LEVELS = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
N_SKU, H, N_Q = 4, 6, len(Q_LEVELS)


@pytest.fixture
def sorted_quantiles() -> np.ndarray:
    """Clean sorted non-negative quantile cube (n_sku, H, n_q)."""
    rng = np.random.default_rng(42)
    raw = rng.random((N_SKU, H, N_Q)) * 100
    return np.sort(raw, axis=2)


@pytest.fixture
def unsorted_quantiles() -> np.ndarray:
    """Deliberately crossed quantile cube."""
    rng = np.random.default_rng(7)
    raw = rng.random((N_SKU, H, N_Q)) * 100
    # Reverse sort to maximise crossings
    return raw[:, :, ::-1].copy()


@pytest.fixture
def negative_quantiles() -> np.ndarray:
    """Quantile cube with some negative values."""
    rng = np.random.default_rng(13)
    return rng.random((N_SKU, H, N_Q)) * 20 - 10  # range [-10, 10]


# ---------------------------------------------------------------------------
# _finalize: floor + sort invariants
# ---------------------------------------------------------------------------


def test_finalize_floors_at_zero(negative_quantiles: np.ndarray) -> None:
    result = _finalize(negative_quantiles, Q_LEVELS)
    assert (result >= 0).all(), "All quantiles must be >= 0 after _finalize"


def test_finalize_sorts_quantiles(unsorted_quantiles: np.ndarray) -> None:
    result = _finalize(unsorted_quantiles, Q_LEVELS)
    diffs = np.diff(result, axis=2)
    assert (diffs >= 0).all(), "Quantiles must be non-decreasing along axis 2 after _finalize"


def test_finalize_preserves_shape(sorted_quantiles: np.ndarray) -> None:
    result = _finalize(sorted_quantiles, Q_LEVELS)
    assert result.shape == sorted_quantiles.shape


def test_finalize_already_sorted_no_change(sorted_quantiles: np.ndarray) -> None:
    """Non-negative, already-sorted input should be returned unchanged."""
    result = _finalize(sorted_quantiles, Q_LEVELS)
    np.testing.assert_array_equal(result, sorted_quantiles)


def test_finalize_emits_warning_on_high_crossing() -> None:
    """Crossing fraction > 5% must trigger a UserWarning."""
    rng = np.random.default_rng(99)
    # Reversed sort guarantees ~100% crossing
    bad = rng.random((10, 10, N_Q))
    bad = bad[:, :, ::-1].copy() + 1.0  # positive but reversed
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _finalize(bad, Q_LEVELS)
    assert any(issubclass(warning.category, UserWarning) for warning in w), (
        "Expected UserWarning for high crossing fraction"
    )


# ---------------------------------------------------------------------------
# from_quantiles
# ---------------------------------------------------------------------------


def test_from_quantiles_shape(sorted_quantiles: np.ndarray) -> None:
    result = ForecastResult.from_quantiles(sorted_quantiles, Q_LEVELS)
    assert result.quantiles.shape == (N_SKU, H, N_Q)


def test_from_quantiles_floors_negatives(negative_quantiles: np.ndarray) -> None:
    result = ForecastResult.from_quantiles(negative_quantiles, Q_LEVELS)
    assert (result.quantiles >= 0).all()


def test_from_quantiles_sorts_crossings(unsorted_quantiles: np.ndarray) -> None:
    result = ForecastResult.from_quantiles(unsorted_quantiles, Q_LEVELS)
    diffs = np.diff(result.quantiles, axis=2)
    assert (diffs >= 0).all()


def test_from_quantiles_stores_q_levels() -> None:
    q = np.sort(np.random.default_rng(0).random((2, 3, N_Q)), axis=2)
    result = ForecastResult.from_quantiles(q, Q_LEVELS)
    np.testing.assert_array_equal(result.q_levels, Q_LEVELS)


def test_from_quantiles_rejects_wrong_ndim() -> None:
    with pytest.raises(ValueError, match="3-D"):
        ForecastResult.from_quantiles(np.ones((5, 6)), Q_LEVELS)


def test_from_quantiles_rejects_mismatched_n_q() -> None:
    with pytest.raises(ValueError, match="q_levels"):
        ForecastResult.from_quantiles(np.ones((2, 3, 5)), Q_LEVELS)  # 5 != 7


def test_from_quantiles_stores_sku_ids() -> None:
    q = np.ones((3, 4, N_Q))
    ids = np.array([101, 202, 303])
    result = ForecastResult.from_quantiles(q, Q_LEVELS, sku_ids=ids)
    np.testing.assert_array_equal(result.sku_ids, ids)


def test_from_quantiles_sku_ids_optional() -> None:
    q = np.ones((2, 3, N_Q))
    result = ForecastResult.from_quantiles(q, Q_LEVELS)
    assert result.sku_ids is None


# ---------------------------------------------------------------------------
# from_samples
# ---------------------------------------------------------------------------


def test_from_samples_shape() -> None:
    rng = np.random.default_rng(5)
    samples = rng.random((N_SKU, H, 500)) * 50
    result = ForecastResult.from_samples(samples, Q_LEVELS)
    assert result.quantiles.shape == (N_SKU, H, N_Q)


def test_from_samples_floors_negatives() -> None:
    rng = np.random.default_rng(6)
    samples = rng.random((N_SKU, H, 200)) * 20 - 10  # some negative
    result = ForecastResult.from_samples(samples, Q_LEVELS)
    assert (result.quantiles >= 0).all()


def test_from_samples_sorts_quantiles() -> None:
    rng = np.random.default_rng(8)
    samples = rng.random((N_SKU, H, 200)) * 50
    result = ForecastResult.from_samples(samples, Q_LEVELS)
    diffs = np.diff(result.quantiles, axis=2)
    assert (diffs >= 0).all()


def test_from_samples_rejects_wrong_ndim() -> None:
    with pytest.raises(ValueError, match="3-D"):
        ForecastResult.from_samples(np.ones((4, 100)), Q_LEVELS)


def test_from_samples_median_near_empirical() -> None:
    """P50 from from_samples must be close to np.median of samples."""
    rng = np.random.default_rng(17)
    samples = rng.random((3, 5, 1000)) * 100
    result = ForecastResult.from_samples(samples, Q_LEVELS)
    empirical_p50 = np.median(samples, axis=2)
    # Allow small tolerance due to quantile interpolation
    np.testing.assert_allclose(result.median(), empirical_p50, rtol=0.02)


def test_from_samples_wider_than_from_pointforecast() -> None:
    """Samples with variance should produce wider intervals than all-same samples."""
    rng = np.random.default_rng(21)
    n_s = 500
    point = np.full((2, 4, n_s), 10.0)  # no variance
    varied = rng.random((2, 4, n_s)) * 20
    result_point = ForecastResult.from_samples(point, Q_LEVELS)
    result_varied = ForecastResult.from_samples(varied, Q_LEVELS)
    # Width = q95 - q05
    width_point = result_point.quantile_at(0.95) - result_point.quantile_at(0.05)
    width_varied = result_varied.quantile_at(0.95) - result_varied.quantile_at(0.05)
    assert (width_varied >= width_point).all()


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


def test_n_sku_accessor() -> None:
    result = ForecastResult.from_quantiles(np.ones((5, 3, N_Q)), Q_LEVELS)
    assert result.n_sku == 5


def test_horizon_accessor() -> None:
    result = ForecastResult.from_quantiles(np.ones((5, 26, N_Q)), Q_LEVELS)
    assert result.horizon == 26


def test_n_quantiles_accessor() -> None:
    result = ForecastResult.from_quantiles(np.ones((5, 3, N_Q)), Q_LEVELS)
    assert result.n_quantiles == N_Q


def test_median_shape() -> None:
    result = ForecastResult.from_quantiles(np.ones((4, 6, N_Q)), Q_LEVELS)
    assert result.median().shape == (4, 6)


def test_quantile_at_p10() -> None:
    rng = np.random.default_rng(33)
    q = np.sort(rng.random((3, 5, N_Q)), axis=2) * 100
    result = ForecastResult.from_quantiles(q, Q_LEVELS)
    expected_idx = int(np.argmin(np.abs(Q_LEVELS - 0.10)))
    np.testing.assert_array_equal(result.quantile_at(0.10), result.quantiles[:, :, expected_idx])


# ---------------------------------------------------------------------------
# ForecastModel ABC
# ---------------------------------------------------------------------------


class _DummyModel(ForecastModel):
    """Minimal concrete implementation for testing the ABC."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> _DummyModel:
        self._fitted = True
        return self

    def predict(self, X: np.ndarray, horizon: int) -> ForecastResult:
        n_sku = X.shape[0]
        q_cube = np.ones((n_sku, horizon, len(self.q_levels)))
        return ForecastResult.from_quantiles(q_cube, self.q_levels)


def test_model_abc_cannot_instantiate_directly() -> None:
    with pytest.raises(TypeError):
        ForecastModel()  # type: ignore[abstract]


def test_model_concrete_instantiates() -> None:
    m = _DummyModel()
    assert not m.is_fitted


def test_model_fit_sets_fitted_flag() -> None:
    m = _DummyModel()
    m.fit(np.ones((5, 10)), np.ones(5))
    assert m.is_fitted


def test_model_predict_returns_forecast_result() -> None:
    m = _DummyModel()
    m.fit(np.ones((3, 10)), np.ones(3))
    result = m.predict(np.ones((3, 10)), horizon=26)
    assert isinstance(result, ForecastResult)
    assert result.quantiles.shape == (3, 26, len(_DummyModel().q_levels))


def test_model_default_q_levels() -> None:
    m = _DummyModel()
    assert len(m.q_levels) == 19  # 19 quantiles from config


def test_model_custom_q_levels() -> None:
    q = np.array([0.1, 0.5, 0.9])
    m = _DummyModel(q_levels=q)
    np.testing.assert_array_equal(m.q_levels, q)


def test_model_repr() -> None:
    m = _DummyModel()
    s = repr(m)
    assert "_DummyModel" in s
    assert "fitted=False" in s
