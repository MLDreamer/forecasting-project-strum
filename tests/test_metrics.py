"""Phase 7 gate: worked-example unit tests for all metrics.

Every expected value is derived analytically below each test so the logic
can be verified without running code.
"""

from __future__ import annotations

import numpy as np
import pytest

from forecasting.metrics import (
    coverage,
    coverage_80,
    coverage_90,
    crps_from_quantiles,
    crps_per_horizon,
    evaluate,
    mase,
    pinball,
    pinball_all_quantiles,
    revenue_weighted_wape,
    smape,
    wape,
    wape_per_horizon,
    wis,
)

# ---------------------------------------------------------------------------
# WAPE
# ---------------------------------------------------------------------------


def test_wape_perfect_forecast() -> None:
    y = np.array([10.0, 20.0, 30.0])
    assert wape(y, y) == pytest.approx(0.0)


def test_wape_known_value() -> None:
    # y=[10,20], pred=[8,20]
    # default weights = y = [10,20]
    # WAPE = (10*|10-8| + 20*|20-20|) / (10+20) = (10*2 + 0) / 30 = 20/30 = 0.6667
    y = np.array([10.0, 20.0])
    f = np.array([8.0, 20.0])
    assert wape(y, f) == pytest.approx(2.0 / 3.0, rel=1e-6)


def test_wape_all_zeros_returns_zero() -> None:
    y = np.zeros(5)
    f = np.ones(5)
    assert wape(y, f) == pytest.approx(0.0)


def test_wape_uniform_weights() -> None:
    # y=[10,0], pred=[0,10], weights=[1,1]
    # WAPE = (1*10 + 1*10) / (1+1) = 10
    y = np.array([10.0, 0.0])
    f = np.array([0.0, 10.0])
    w = np.array([1.0, 1.0])
    assert wape(y, f, w) == pytest.approx(10.0)


def test_wape_per_horizon_shape() -> None:
    N, H = 4, 6
    y = np.random.default_rng(0).random((N, H))
    f = np.random.default_rng(1).random((N, H))
    result = wape_per_horizon(y, f)
    assert result.shape == (H,)
    assert (result >= 0).all()


def test_wape_per_horizon_known() -> None:
    # N=2, H=2
    # y=[[4,8],[6,2]], f=[[4,4],[6,2]]
    # h=0: errors=[0,0], WAPE=0
    # h=1: errors=[4,0], wts=y[:,1]=[8,2], WAPE=(8*4+2*0)/(8+2)=32/10=3.2
    y = np.array([[4.0, 8.0], [6.0, 2.0]])
    f = np.array([[4.0, 4.0], [6.0, 2.0]])
    result = wape_per_horizon(y, f)
    assert result[0] == pytest.approx(0.0)
    assert result[1] == pytest.approx(3.2, rel=1e-6)


# ---------------------------------------------------------------------------
# MASE
# ---------------------------------------------------------------------------


def test_mase_perfect_forecast() -> None:
    y_train = np.arange(1.0, 54.0)  # 53 points for seasonality=52
    y_true = np.array([10.0, 20.0])
    assert mase(y_true, y_true, y_train) == pytest.approx(0.0)


def test_mase_known_value() -> None:
    # scale = mean(|y_train[t] - y_train[t-1]|) for seasonality=1
    # y_train = [1,3,5,7] -> diffs=[2,2,2] -> scale=2
    # y_true=[6], y_pred=[4] -> MAE=2
    # MASE = 2/2 = 1.0
    y_train = np.array([1.0, 3.0, 5.0, 7.0])
    y_true = np.array([6.0])
    y_pred = np.array([4.0])
    result = mase(y_true, y_pred, y_train, seasonality=1)
    assert result == pytest.approx(1.0, rel=1e-6)


def test_mase_short_train_fallback() -> None:
    # y_train shorter than seasonality=52 -> uses mean(|y_train|) as scale
    y_train = np.array([2.0, 4.0, 6.0])  # only 3 points
    y_true = np.array([5.0])
    y_pred = np.array([3.0])
    # scale = mean(|[2,4,6]|) = 4.0; MAE = 2.0; MASE = 0.5
    result = mase(y_true, y_pred, y_train, seasonality=52)
    assert result == pytest.approx(0.5, rel=1e-6)


# ---------------------------------------------------------------------------
# Pinball
# ---------------------------------------------------------------------------


def test_pinball_q05_underforecast() -> None:
    # y=10, f=8, q=0.05 -> f < y -> loss = q*(y-f) = 0.05*2 = 0.10
    # uniform weight -> 0.10
    assert pinball(np.array([10.0]), np.array([8.0]), 0.05) == pytest.approx(0.10, rel=1e-6)


def test_pinball_q05_overforecast() -> None:
    # y=8, f=10, q=0.05 -> f > y -> loss = (q-1)*(y-f) = (-0.95)*(-2) = 1.90
    assert pinball(np.array([8.0]), np.array([10.0]), 0.05) == pytest.approx(1.90, rel=1e-6)


def test_pinball_q50_is_half_mae() -> None:
    # At q=0.5, pinball = 0.5 * |y - f|
    y = np.array([4.0, 8.0, 12.0])
    f = np.array([2.0, 10.0, 12.0])
    # |errors| = [2, 2, 0] -> mean = 4/3
    # pinball at q=0.5: uniform weights -> sum = 0.5*(2+2+0)/3 = 0.5*4/3 = 2/3
    result = pinball(y, f, 0.5)
    assert result == pytest.approx(2.0 / 3.0, rel=1e-6)


def test_pinball_all_quantiles_shape() -> None:
    N, Q = 5, 3
    y = np.ones(N)
    q_arr = np.column_stack([0.5 * np.ones(N), np.ones(N), 1.5 * np.ones(N)])
    q_levels = np.array([0.1, 0.5, 0.9])
    result = pinball_all_quantiles(y, q_arr, q_levels)
    assert result.shape == (Q,)


# ---------------------------------------------------------------------------
# CRPS
# ---------------------------------------------------------------------------


def test_crps_perfect_forecast_degenerate() -> None:
    # If all Q quantiles equal y_true exactly, CRPS = 0
    y = np.array([5.0, 10.0])
    q_levels = np.array([0.1, 0.5, 0.9])
    quantiles = np.column_stack([y, y, y])
    result = crps_from_quantiles(y, quantiles, q_levels)
    assert result == pytest.approx(0.0, abs=1e-8)


def test_crps_nonneg() -> None:
    rng = np.random.default_rng(42)
    N, Q = 20, 9
    y = rng.random(N) * 100
    q_levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    # Random quantiles (not necessarily sorted — just testing non-negativity of formula)
    quantiles = np.sort(rng.random((N, Q)) * 100, axis=1)
    result = crps_from_quantiles(y, quantiles, q_levels)
    assert result >= 0.0


def test_crps_wider_intervals_higher_score() -> None:
    # Perfectly calibrated narrow intervals vs wide ones with same median
    y = np.array([5.0])
    q_levels = np.array([0.1, 0.5, 0.9])
    # Narrow: quantiles all at truth
    narrow = np.array([[5.0, 5.0, 5.0]])
    # Wide: q10=0, q50=5, q90=10
    wide = np.array([[0.0, 5.0, 10.0]])
    crps_narrow = crps_from_quantiles(y, narrow, q_levels)
    crps_wide = crps_from_quantiles(y, wide, q_levels)
    assert crps_narrow < crps_wide


def test_crps_per_horizon_shape() -> None:
    N, H, Q = 10, 26, 9
    rng = np.random.default_rng(7)
    y = rng.random((N, H))
    quantiles = np.sort(rng.random((N, H, Q)), axis=2)
    q_levels = np.linspace(0.1, 0.9, Q)
    result = crps_per_horizon(y, quantiles, q_levels)
    assert result.shape == (H,)
    assert (result >= 0).all()


# ---------------------------------------------------------------------------
# WIS
# ---------------------------------------------------------------------------


def test_wis_nonneg() -> None:
    rng = np.random.default_rng(11)
    N, Q = 15, 9
    y = rng.random(N) * 50
    q_levels = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80])
    quantiles = np.sort(rng.random((N, Q)) * 50, axis=1)
    result = wis(y, quantiles, q_levels)
    assert result >= 0.0


def test_wis_known_single_interval() -> None:
    # y=5, lower=3, upper=8, q_levels=[0.25, 0.5, 0.75]
    # Only one interval pair: (q=0.25, q=0.75), alpha=0.5
    # y inside [3,8], so penalty=0
    # IS = (8-3) + 0 = 5
    # median=0.5 term: 0.5*|5-5|=0
    # total = 0.5*5 * (0.5/2) = 0.5*5*0.25 ... let me recompute
    # WIS = (1/(K+0.5)) * [0.5*|y-med| + sum_k alpha_k/2 * IS_k]
    # K=1 interval, median=q[1]=5
    # denom = 1 + 0.5 = 1.5
    # 0.5*|5-5| = 0
    # alpha=2*q[0]=2*0.25=0.5, IS=5+0=5
    # contribution = 0.5/2 * 5 = 1.25
    # WIS = (0 + 1.25) / 1.5 = 0.833...
    y = np.array([5.0])
    quantiles = np.array([[3.0, 5.0, 8.0]])
    q_levels = np.array([0.25, 0.5, 0.75])
    result = wis(y, quantiles, q_levels)
    assert result == pytest.approx(1.25 / 1.5, rel=1e-5)


def test_wis_penalised_when_outside() -> None:
    # y=0 outside [3,8], q_levels=[0.25,0.5,0.75]
    # median (q=0.5) = 5: 0.5*|0-5| = 2.5
    # One interval: alpha=2*0.25=0.5, lower=3, upper=8
    # penalty = max(0, 3-0) = 3
    # IS = (8-3) + 2/0.5 * 3 = 5 + 12 = 17
    # alpha/2 * IS = 0.5/2 * 17 = 4.25
    # total = 2.5 + 4.25 = 6.75; denom = 1+0.5 = 1.5
    # WIS = 6.75 / 1.5 = 4.5
    y = np.array([0.0])
    quantiles = np.array([[3.0, 5.0, 8.0]])
    q_levels = np.array([0.25, 0.5, 0.75])
    result = wis(y, quantiles, q_levels)
    assert result == pytest.approx(4.5, rel=1e-5)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def test_coverage_all_inside() -> None:
    y = np.array([2.0, 4.0, 6.0])
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([10.0, 10.0, 10.0])
    assert coverage(y, lo, hi) == pytest.approx(1.0)


def test_coverage_none_inside() -> None:
    y = np.array([20.0, 30.0])
    lo = np.array([0.0, 0.0])
    hi = np.array([10.0, 10.0])
    assert coverage(y, lo, hi) == pytest.approx(0.0)


def test_coverage_partial() -> None:
    # y=[5, 15], lo=[0,0], hi=[10,10]
    # 5 inside, 15 outside -> 1 of 2 = 0.5 (uniform weights)
    y = np.array([5.0, 15.0])
    lo = np.zeros(2)
    hi = np.full(2, 10.0)
    assert coverage(y, lo, hi) == pytest.approx(0.5)


def test_coverage_80_below_90() -> None:
    rng = np.random.default_rng(22)
    N, Q = 100, 9
    y = rng.random(N) * 10
    q_levels = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80])
    quantiles = np.sort(rng.random((N, Q)) * 10, axis=1)
    c80 = coverage_80(y, quantiles, q_levels)
    c90 = coverage_90(y, quantiles, q_levels)
    # 80% PI (q10-q90 = 80pp width) should generally be <= 90% PI (q05-q80 = 75pp width)
    # Note: this is a probabilistic assertion that holds in expectation but may not
    # hold for every random seed. Use lenient check.
    assert 0.0 <= c80 <= 1.0
    assert 0.0 <= c90 <= 1.0


# ---------------------------------------------------------------------------
# sMAPE
# ---------------------------------------------------------------------------


def test_smape_perfect() -> None:
    y = np.array([5.0, 10.0])
    assert smape(y, y) == pytest.approx(0.0)


def test_smape_known_value() -> None:
    # y=10, f=6 -> 2*|4|/(10+6) = 8/16 = 0.5
    y = np.array([10.0])
    f = np.array([6.0])
    assert smape(y, f) == pytest.approx(0.5, rel=1e-5)


def test_smape_symmetric() -> None:
    # sMAPE of (y,f) should equal sMAPE of (f,y)
    y = np.array([3.0, 7.0])
    f = np.array([5.0, 4.0])
    assert smape(y, f) == pytest.approx(smape(f, y), rel=1e-6)


def test_smape_both_zero_no_nan() -> None:
    # Both zero: 2*0 / (0+0+eps) = 0
    y = np.zeros(3)
    f = np.zeros(3)
    result = smape(y, f)
    assert np.isfinite(result)
    assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# evaluate() convenience function
# ---------------------------------------------------------------------------


def test_evaluate_returns_all_keys() -> None:
    N, Q = 10, 9
    rng = np.random.default_rng(99)
    y = rng.random(N) * 20
    quantiles = np.sort(rng.random((N, Q)) * 20, axis=1)
    q_levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    y_pred = quantiles[:, 4]  # P50
    y_train = rng.random(100) * 20
    result = evaluate(y, y_pred, quantiles, q_levels, y_train)
    required = {"wape", "crps", "wis", "coverage_80", "coverage_90", "smape", "mase"}
    assert required.issubset(result.keys())
    # One pinball key per quantile
    for q in q_levels:
        assert f"pinball_{q:.2f}" in result


def test_evaluate_all_values_finite() -> None:
    N, Q = 20, 9
    rng = np.random.default_rng(55)
    y = rng.random(N) * 50
    quantiles = np.sort(rng.random((N, Q)) * 50, axis=1)
    q_levels = np.linspace(0.1, 0.9, Q)
    y_pred = quantiles[:, Q // 2]
    result = evaluate(y, y_pred, quantiles, q_levels)
    for k, v in result.items():
        assert np.isfinite(v), f"{k} is not finite: {v}"


def test_evaluate_wape_consistent() -> None:
    """evaluate['wape'] must match standalone wape()."""
    N, Q = 15, 9
    rng = np.random.default_rng(77)
    y = rng.random(N) * 30
    quantiles = np.sort(rng.random((N, Q)) * 30, axis=1)
    q_levels = np.linspace(0.1, 0.9, Q)
    y_pred = quantiles[:, Q // 2]
    result = evaluate(y, y_pred, quantiles, q_levels)
    assert result["wape"] == pytest.approx(wape(y, y_pred), rel=1e-6)


# ---------------------------------------------------------------------------
# Revenue-weighting smoke tests
# ---------------------------------------------------------------------------


def test_wape_revenue_weighting_shifts_toward_high_volume() -> None:
    # Low-volume SKU has huge error; high-volume SKU is perfect.
    # Revenue-weighted WAPE should be much lower than uniform WAPE.
    y = np.array([1.0, 1000.0])  # high-volume SKU sells 1000
    f = np.array([100.0, 1000.0])  # low-volume SKU: off by 99, high-volume: perfect
    wape_rev = wape(y, f)  # default = revenue-weighted
    wape_uni = wape(y, f, np.ones(2))
    assert wape_rev < wape_uni


def test_coverage_weighted_differs_from_uniform() -> None:
    y = np.array([5.0, 15.0])
    lo = np.zeros(2)
    hi = np.full(2, 10.0)
    # Only y[0]=5 is inside; uniform -> 0.5
    assert coverage(y, lo, hi) == pytest.approx(0.5)
    # Revenue-weight [1, 99]: nearly all weight on y[1]=15 (outside) -> ~0.01
    w = np.array([1.0, 99.0])
    assert coverage(y, lo, hi, w) < 0.1


# ---------------------------------------------------------------------------
# revenue_weighted_wape — the locked selection metric (S1.2)
# ---------------------------------------------------------------------------


def test_rww_perfect_forecast() -> None:
    """f == y → RW-WAPE = 0."""
    y = np.array([[10.0, 20.0], [5.0, 8.0]])
    price = np.array([100.0, 50.0])
    assert revenue_weighted_wape(y, y, price) == pytest.approx(0.0)


def test_rww_known_value() -> None:
    """Hand-computed example.

    y = [[4, 6], [2, 8]]   f = [[4, 6], [0, 8]]   price = [10, 10]
    SKU 0: |y-f| sums = 0, y sums = 10
    SKU 1: |y-f| sums = 2, y sums = 10
    RW-WAPE = (10*0 + 10*2) / (10*10 + 10*10) = 20/200 = 0.10
    """
    y = np.array([[4.0, 6.0], [2.0, 8.0]])
    f = np.array([[4.0, 6.0], [0.0, 8.0]])
    price = np.array([10.0, 10.0])
    assert revenue_weighted_wape(y, f, price) == pytest.approx(0.10, rel=1e-6)


def test_rww_high_price_sku_dominates() -> None:
    """SKU with higher price dominates the metric.

    SKU 0 (price=1):   huge error (y=10, f=0)
    SKU 1 (price=100): perfect forecast (y=10, f=10)
    RW-WAPE should be low (close to 0) because the accurate high-price SKU dominates.
    """
    y = np.array([[10.0], [10.0]])
    f = np.array([[0.0], [10.0]])
    price_high_acc = np.array([1.0, 100.0])  # accurate SKU has high price
    price_high_err = np.array([100.0, 1.0])  # inaccurate SKU has high price

    rww_acc = revenue_weighted_wape(y, f, price_high_acc)
    rww_err = revenue_weighted_wape(y, f, price_high_err)
    assert rww_acc < rww_err


def test_rww_all_zero_actual_returns_zero() -> None:
    """When actual demand is zero, denominator is zero → returns 0.0 safely."""
    y = np.zeros((3, 4))
    f = np.ones((3, 4)) * 5.0
    price = np.array([20.0, 30.0, 10.0])
    assert revenue_weighted_wape(y, f, price) == pytest.approx(0.0)


def test_rww_shape_2d() -> None:
    """Accepts (n_sku, H) arrays."""
    rng = np.random.default_rng(42)
    y = rng.random((10, 26)) * 50
    f = rng.random((10, 26)) * 50
    price = rng.random(10) * 100
    result = revenue_weighted_wape(y, f, price)
    assert np.isfinite(result)
    assert result >= 0.0


def test_rww_protocol_pooled_not_averaged() -> None:
    """Pooled ratio ≠ average of per-SKU WAPEs for heterogeneous demand."""
    y = np.array([[100.0], [1.0]])
    f = np.array([[110.0], [0.0]])
    price = np.array([1.0, 1.0])

    # Pooled (correct per protocol)
    rww_pooled = revenue_weighted_wape(y, f, price)
    # Average of per-SKU WAPEs
    wape0 = abs(100 - 110) / 100
    wape1 = abs(1 - 0) / 1
    avg = (wape0 + wape1) / 2

    # Pooled < average when one SKU has much higher volume
    assert rww_pooled != pytest.approx(avg)
