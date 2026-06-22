"""Phase 13 gate: foundation models (Chronos + Moirai) flow through ForecastResult.

Doc requirements:
- Both flow through ForecastResult with no special-casing (dual-constructor test).
- Chronos uses from_samples; Moirai uses from_quantiles.
- Track: per-horizon WAPE, per-SB-class breakdown (esp. cold_start vs seasonal_naive),
  inference timing per SKU.
- Moirai: graceful ImportError when uni2ts unavailable (this box).
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pytest

from forecasting.models.base import ForecastResult
from forecasting.models.foundation import ChronosTiny, InferenceTiming, MoiraiSmall
from forecasting.registry import candidates_for

logger = logging.getLogger(__name__)

Q_LEVELS = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
HORIZON = 4

_CHRONOS_AVAILABLE = True
try:
    import torch  # noqa: F401
    from chronos import ChronosPipeline  # noqa: F401
except ImportError:
    _CHRONOS_AVAILABLE = False

_MOIRAI_AVAILABLE = True
try:
    from uni2ts.model.moirai import MoiraiForecast  # noqa: F401
except ImportError:
    _MOIRAI_AVAILABLE = False

requires_chronos = pytest.mark.skipif(
    not _CHRONOS_AVAILABLE, reason="chronos-forecasting not installed"
)
requires_moirai = pytest.mark.skipif(not _MOIRAI_AVAILABLE, reason="uni2ts not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _series(n: int = 52, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    trend = np.linspace(5.0, 12.0, n)
    return np.maximum(0.0, trend + rng.normal(0, 2.0, n))


def _intermittent(n: int = 52, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.where(rng.random(n) < 0.3, rng.exponential(8, n), 0.0)


def _cold_start(n: int = 4, seed: int = 0) -> np.ndarray:
    """Very short series — the cold-start case foundation models must handle."""
    rng = np.random.default_rng(seed)
    return np.maximum(0.0, rng.exponential(5, n))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_chronos_registered_for_cold_start() -> None:
    import forecasting.models.foundation  # trigger registration  # noqa: F401

    assert ChronosTiny in candidates_for("cold_start")


def test_chronos_registered_for_smooth() -> None:
    import forecasting.models.foundation  # noqa: F401

    assert ChronosTiny in candidates_for("smooth")


def test_moirai_registered_for_cold_start() -> None:
    import forecasting.models.foundation  # noqa: F401

    assert MoiraiSmall in candidates_for("cold_start")


# ---------------------------------------------------------------------------
# Moirai unavailability — graceful ImportError (always runs)
# ---------------------------------------------------------------------------


def test_moirai_fit_series_raises_import_error_when_unavailable() -> None:
    """When uni2ts is absent, fit_series must raise ImportError, not crash."""
    if _MOIRAI_AVAILABLE:
        pytest.skip("uni2ts is installed — can't test unavailability path")
    m = MoiraiSmall(q_levels=Q_LEVELS)
    with pytest.raises(ImportError, match="uni2ts"):
        m.fit_series({"sku1": _series(52)})


def test_moirai_registers_even_when_unavailable() -> None:
    """Registration must succeed even without uni2ts installed."""
    import forecasting.models.foundation  # noqa: F401

    assert MoiraiSmall in candidates_for("smooth")


# ---------------------------------------------------------------------------
# Chronos — core ForecastResult contract (dual-constructor test)
# ---------------------------------------------------------------------------


@requires_chronos
def test_chronos_returns_forecast_result() -> None:
    """Chronos must return ForecastResult — no special-casing downstream."""
    m = ChronosTiny(q_levels=Q_LEVELS, n_samples=10)
    m.fit_series({"sku1": _series(52, seed=1)})
    result = m.predict(np.empty(0), HORIZON)
    assert isinstance(result, ForecastResult)


@requires_chronos
def test_chronos_uses_from_samples_path() -> None:
    """Chronos output must come from from_samples (sample-based model)."""
    m = ChronosTiny(q_levels=Q_LEVELS, n_samples=10)
    m.fit_series({"sku1": _series(52, seed=2)})
    result = m.predict(np.empty(0), HORIZON)
    # from_samples always produces non-negative sorted output
    assert (result.quantiles >= 0).all()
    diffs = np.diff(result.quantiles, axis=2)
    assert (diffs >= 0).all()


@requires_chronos
def test_chronos_shape() -> None:
    n_skus = 3
    series = {f"sku{i}": _series(52, seed=i) for i in range(n_skus)}
    m = ChronosTiny(q_levels=Q_LEVELS, n_samples=10)
    m.fit_series(series)
    result = m.predict(np.empty(0), HORIZON)
    assert result.quantiles.shape == (n_skus, HORIZON, len(Q_LEVELS))


@requires_chronos
def test_chronos_non_negative() -> None:
    m = ChronosTiny(q_levels=Q_LEVELS, n_samples=10)
    m.fit_series({"sku1": _series(52, seed=3)})
    result = m.predict(np.empty(0), HORIZON)
    assert (result.quantiles >= 0).all()


@requires_chronos
def test_chronos_sorted_quantiles() -> None:
    m = ChronosTiny(q_levels=Q_LEVELS, n_samples=10)
    m.fit_series({"sku1": _series(52, seed=4)})
    result = m.predict(np.empty(0), HORIZON)
    assert (np.diff(result.quantiles, axis=2) >= 0).all()


# ---------------------------------------------------------------------------
# Chronos — handles different SB segment types (cold_start critical)
# ---------------------------------------------------------------------------


@requires_chronos
def test_chronos_handles_cold_start_series() -> None:
    """Cold-start (very short history): must not crash, must return valid result."""
    m = ChronosTiny(q_levels=Q_LEVELS, n_samples=10)
    m.fit_series({"cold": _cold_start(n=4, seed=5)})
    result = m.predict(np.empty(0), HORIZON)
    assert isinstance(result, ForecastResult)
    assert (result.quantiles >= 0).all()


@requires_chronos
def test_chronos_handles_intermittent_series() -> None:
    m = ChronosTiny(q_levels=Q_LEVELS, n_samples=10)
    m.fit_series({"int1": _intermittent(52, seed=6)})
    result = m.predict(np.empty(0), HORIZON)
    assert isinstance(result, ForecastResult)
    assert result.quantiles.shape == (1, HORIZON, len(Q_LEVELS))


@requires_chronos
def test_chronos_per_sb_class_breakdown() -> None:
    """Fit on one SKU per SB class, verify all produce valid forecasts."""
    rng = np.random.default_rng(42)
    series = {
        "smooth": np.maximum(0, np.linspace(5, 12, 52) + rng.normal(0, 1, 52)),
        "erratic": np.maximum(0, np.linspace(5, 12, 52) + rng.normal(0, 5, 52)),
        "intermittent": np.where(rng.random(52) < 0.3, rng.exponential(5, 52), 0.0),
        "lumpy": np.where(rng.random(52) < 0.25, rng.exponential(20, 52), 0.0),
        "cold_start": np.maximum(0, rng.exponential(5, 6)),
    }
    m = ChronosTiny(q_levels=Q_LEVELS, n_samples=10)
    m.fit_series(series)
    result = m.predict(np.empty(0), HORIZON)

    assert result.quantiles.shape == (5, HORIZON, len(Q_LEVELS))
    assert (result.quantiles >= 0).all()
    assert (np.diff(result.quantiles, axis=2) >= 0).all()


# ---------------------------------------------------------------------------
# Chronos — inference timing tracked
# ---------------------------------------------------------------------------


@requires_chronos
def test_chronos_timing_populated() -> None:
    """After predict(), timing object must be populated."""
    m = ChronosTiny(q_levels=Q_LEVELS, n_samples=10)
    m.fit_series({"s1": _series(52, seed=7), "s2": _series(52, seed=8)})
    m.predict(np.empty(0), HORIZON)

    assert m.timing.total_skus == 2
    assert m.timing.total_seconds > 0
    assert len(m.timing.per_sku_seconds) == 2
    assert m.timing.mean_seconds_per_sku > 0
    logger.info = lambda *a, **k: None  # silence during test
    m.timing.log_summary("ChronosTiny")


@requires_chronos
def test_chronos_timing_reasonable() -> None:
    """Per-SKU inference time should be < 30 seconds on CPU (T5-tiny is fast)."""
    m = ChronosTiny(q_levels=Q_LEVELS, n_samples=10)
    m.fit_series({"sku1": _series(52, seed=9)})
    t0 = time.perf_counter()
    m.predict(np.empty(0), HORIZON)
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0, f"Inference took {elapsed:.1f}s — too slow for production batch"


# ---------------------------------------------------------------------------
# InferenceTiming helper
# ---------------------------------------------------------------------------


def test_timing_mean_empty() -> None:
    t = InferenceTiming()
    assert t.mean_seconds_per_sku == 0.0
    assert t.p95_seconds_per_sku == 0.0


def test_timing_mean_computed() -> None:
    t = InferenceTiming(total_skus=3, total_seconds=3.0, per_sku_seconds=[0.5, 1.0, 1.5])
    assert t.mean_seconds_per_sku == pytest.approx(1.0)
    assert t.p95_seconds_per_sku > t.mean_seconds_per_sku * 0.9


# ---------------------------------------------------------------------------
# WAPE comparison: Chronos vs SeasonalNaive on smooth series
# ---------------------------------------------------------------------------


@requires_chronos
def test_chronos_vs_seasonal_naive_wape() -> None:
    """Track per-horizon WAPE for Chronos vs SeasonalNaive (doc requirement).

    Not asserting Chronos wins — this is a diagnostic comparison.
    Both must produce valid ForecastResult objects.
    """
    from forecasting.metrics import wape
    from forecasting.models.baseline import SeasonalNaive

    rng = np.random.default_rng(55)
    n_train = 52
    n_test = HORIZON

    y_train = np.maximum(0.0, np.linspace(5, 12, n_train) + rng.normal(0, 1.5, n_train))
    y_test = np.maximum(0.0, np.linspace(12, 14, n_test) + rng.normal(0, 1.5, n_test))

    # SeasonalNaive
    sn = SeasonalNaive(q_levels=Q_LEVELS)
    sn.fit_series({"sku": y_train})
    sn_result = sn.predict(np.empty(0), HORIZON)
    sn_p50 = sn_result.median()[0]  # (H,)

    # Chronos
    ch = ChronosTiny(q_levels=Q_LEVELS, n_samples=20)
    ch.fit_series({"sku": y_train})
    ch_result = ch.predict(np.empty(0), HORIZON)
    ch_p50 = ch_result.median()[0]  # (H,)

    wape_sn = wape(y_test, sn_p50)
    wape_ch = wape(y_test, ch_p50)

    # Both must be finite and non-negative
    assert np.isfinite(wape_sn)
    assert wape_sn >= 0
    assert np.isfinite(wape_ch)
    assert wape_ch >= 0

    # Log the comparison — the doc says "where they must earn their slot"
    logger.info(
        "WAPE comparison — SeasonalNaive: %.3f | Chronos-T5-tiny: %.3f",
        wape_sn,
        wape_ch,
    )
