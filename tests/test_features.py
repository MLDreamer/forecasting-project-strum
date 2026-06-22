"""Phase 4 gate: feature shape, lag-1 leakage check, no-future-data discipline."""

from __future__ import annotations

import numpy as np
import pytest

from forecasting import config
from forecasting.densify import densify
from forecasting.features import FeaturesResult, build_features
from forecasting.io import load_all
from forecasting.lifecycle import infer_lifecycle


@pytest.fixture(scope="module")
def result() -> FeaturesResult:
    d = load_all()
    lc = infer_lifecycle(d.sales, d.master)
    dense = densify(d.sales, lc, d.joined, week_relabel_shift_days=6)
    return build_features(dense.dense, lc)


# ---------------------------------------------------------------------------
# Gate: shape
# ---------------------------------------------------------------------------


def test_row_count(result: FeaturesResult) -> None:
    assert len(result.features) == 16_068


def test_column_count(result: FeaturesResult) -> None:
    assert len(result.features.columns) == 98, (
        f"Expected 98 cols, got {len(result.features.columns)}: {list(result.features.columns)}"
    )


def test_feature_col_count(result: FeaturesResult) -> None:
    assert len(result.feature_cols) == 90


# ---------------------------------------------------------------------------
# Leakage discipline: lag_1[t] == sales[t-1]
# ---------------------------------------------------------------------------


def test_lag1_equals_previous_sales(result: FeaturesResult) -> None:
    """lag_1 at time t must equal sales at t-1 for every SKU."""
    df = result.features.sort_values([config.COL_SKU_ID, config.COL_TIMESTAMP])
    for sku, grp in df.groupby(config.COL_SKU_ID):
        grp = grp.reset_index(drop=True)
        if len(grp) < 2:
            continue
        expected = grp[config.COL_SALES].shift(1).iloc[1:].values
        actual = grp["lag_1"].iloc[1:].values
        assert np.allclose(actual, expected, equal_nan=True), (
            f"SKU {sku}: lag_1 does not equal sales[t-1]"
        )


def test_lag4_equals_sales_shifted4(result: FeaturesResult) -> None:
    df = result.features.sort_values([config.COL_SKU_ID, config.COL_TIMESTAMP])
    for sku, grp in df.groupby(config.COL_SKU_ID):
        grp = grp.reset_index(drop=True)
        if len(grp) < 5:
            continue
        expected = grp[config.COL_SALES].shift(4).iloc[4:].values
        actual = grp["lag_4"].iloc[4:].values
        assert np.allclose(actual, expected, equal_nan=True), f"SKU {sku}: lag_4 mismatch"


def test_discount_is_never_contemporaneous(result: FeaturesResult) -> None:
    """discount_pct itself must NOT appear in feature_cols — only its lag."""
    assert config.COL_DISCOUNT_PCT not in result.feature_cols
    assert "discount_pct_lag1" in result.feature_cols


# ---------------------------------------------------------------------------
# Required feature groups present
# ---------------------------------------------------------------------------


def test_all_lag_cols_present(result: FeaturesResult) -> None:
    for lag in [1, 2, 3, 4, 5, 6, 8, 13, 26, 52]:
        assert f"lag_{lag}" in result.feature_cols, f"Missing lag_{lag}"


def test_rolling_mean_cols_present(result: FeaturesResult) -> None:
    for w in [4, 8, 13, 26, 52]:
        assert f"roll{w}_mean" in result.feature_cols


def test_fourier_cols_present(result: FeaturesResult) -> None:
    # Annual (52w) k=1..4
    for k in [1, 2, 3, 4]:
        assert f"fourier_52w_sin_k{k}" in result.feature_cols
        assert f"fourier_52w_cos_k{k}" in result.feature_cols
    # Semi-annual (26w) k=1..2
    for k in [1, 2]:
        assert f"fourier_26w_sin_k{k}" in result.feature_cols
    # Quarterly (13w) k=1..4
    for k in [1, 2, 3, 4]:
        assert f"fourier_13w_sin_k{k}" in result.feature_cols


def test_holiday_cols_present(result: FeaturesResult) -> None:
    for hol in ["hol_christmas", "hol_thanksgiving", "hol_new_year", "hol_black_friday"]:
        assert hol in result.feature_cols


def test_static_cols_present(result: FeaturesResult) -> None:
    for col in ["idi", "cv2", "zero_rate", "gini", "hurst", "abc_tier_enc"]:
        assert col in result.feature_cols


def test_calendar_cols_present(result: FeaturesResult) -> None:
    for col in ["week_of_year", "month", "quarter", "weeks_since_first_sale", "sku_age_weeks"]:
        assert col in result.feature_cols


# ---------------------------------------------------------------------------
# Value plausibility
# ---------------------------------------------------------------------------


def test_fourier_range(result: FeaturesResult) -> None:
    """Fourier values must be in [-1, 1]."""
    fourier_cols = [c for c in result.feature_cols if c.startswith("fourier_")]
    for col in fourier_cols:
        vals = result.features[col]
        assert vals.between(-1.0, 1.0).all(), f"{col} out of [-1, 1]"


def test_holiday_flags_binary(result: FeaturesResult) -> None:
    hol_cols = [c for c in result.feature_cols if c.startswith("hol_")]
    for col in hol_cols:
        unique = result.features[col].unique()
        assert set(unique).issubset({0.0, 1.0}), f"{col} has non-binary values"


def test_is_q4_binary(result: FeaturesResult) -> None:
    assert set(result.features["is_q4"].unique()).issubset({0.0, 1.0})


def test_zero_rate_in_unit_interval(result: FeaturesResult) -> None:
    assert result.features["zero_rate"].between(0.0, 1.0).all()


def test_gini_in_unit_interval(result: FeaturesResult) -> None:
    assert result.features["gini"].between(0.0, 1.0).all()


def test_idi_positive(result: FeaturesResult) -> None:
    assert (result.features["idi"] > 0).all()


def test_no_nulls_in_feature_cols(result: FeaturesResult) -> None:
    """After filling, no feature column should have NaN."""
    null_cols = [c for c in result.feature_cols if result.features[c].isna().any()]
    assert null_cols == [], f"NaN found in: {null_cols}"


def test_base_columns_preserved(result: FeaturesResult) -> None:
    """The 8 base columns from densify must still be present."""
    required = {
        config.COL_SKU_ID,
        config.COL_TIMESTAMP,
        config.COL_SALES,
        config.COL_LIST_PRICE,
        config.COL_DISCOUNT_PCT,
        config.COL_PRODUCT_TYPE,
        config.COL_STATUS,
        "is_potential_stockout",
    }
    assert required.issubset(result.features.columns)


def test_statics_constant_per_sku(result: FeaturesResult) -> None:
    """Static features must be identical for all rows of a given SKU."""
    for col in ["idi", "cv2", "zero_rate", "gini", "hurst", "abc_tier_enc"]:
        nunique = result.features.groupby(config.COL_SKU_ID)[col].nunique()
        bad = nunique[nunique > 1]
        assert len(bad) == 0, f"Static column '{col}' varies within SKU: {bad.index.tolist()}"


def test_weeks_since_first_sale_nonneg(result: FeaturesResult) -> None:
    assert (result.features["weeks_since_first_sale"] >= 0).all()


def test_momentum_finite(result: FeaturesResult) -> None:
    mom_cols = [c for c in result.feature_cols if c.startswith("mom")]
    for col in mom_cols:
        assert np.isfinite(result.features[col].values).all(), f"{col} has non-finite values"
