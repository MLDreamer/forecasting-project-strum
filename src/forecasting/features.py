"""Registry-driven feature generators.

Leakage discipline (locked):
- All lags/rolling are shifted so feature at time t uses data up to t-1.
- discount_pct is only used lagged (never contemporaneous).
- Cluster/hierarchy LOO aggregates are added in Phases 5b/6b AFTER cutoff split.
- The CV cutoff is applied ONCE in the CV harness (Phase 14), not here.

Feature groups produced (8 base + 90 feature = 98 total columns):
  A. Sales lags:        lag_1..6, lag_8, lag_13, lag_26, lag_52   (10)
  B. Rolling mean:      roll4/8/13/26/52_mean                      (5)
  C. Rolling std:       roll4/8/13/26/52_std                       (5)
  D. Rolling max/min:   roll4/13_max, roll4/13_min                 (4)
  E. Rolling discount:  discount_roll4/13_mean                     (2)
  F. Log transforms:    log1p of roll4/8/13/26 means + lag1/4/13  (7)
  G. Log price:         log1p_list_price                           (1)
  H. Momentum:          mom4, mom13, mom26, mom52                  (4)
  I. Fourier 52w:       sin/cos k=1..4                             (8)
  J. Fourier 26w:       sin/cos k=1..2                             (4)
  K. Fourier 13w:       sin/cos k=1..4                             (8)
  L. US holidays:       15 binary flags                            (15)
  M. Promo/price:       discount_lag1, price_lag1, price_roll4/13, price_vs_roll13 (5)
  N. Calendar:          week_of_year, month, quarter,
                        weeks_since_first_sale, sku_age_weeks      (5)
  O. Seasonality:       is_q4                                      (1)
  P. Statics:           idi, cv2, zero_rate, abc_tier_enc, gini, hurst (6)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecasting import config
from forecasting.lifecycle import LifecycleResult

logger = logging.getLogger(__name__)

# Catalog epoch — used to compute sku_age_weeks
_CATALOG_EPOCH = pd.Timestamp("2020-12-27")

# US holiday approximate dates expressed as (month, day) tuples
# For floating holidays we use fixed-week approximations
_HOLIDAY_MMDD: dict[str, tuple[int, int]] = {
    "hol_new_year": (1, 1),
    "hol_mlk": (1, 15),  # 3rd Monday ≈ Jan 15
    "hol_valentines": (2, 14),
    "hol_presidents": (2, 19),  # 3rd Monday ≈ Feb 19
    "hol_mothers": (5, 12),  # 2nd Sunday ≈ May 12
    "hol_memorial": (5, 27),  # last Monday ≈ May 27
    "hol_fathers": (6, 16),  # 3rd Sunday ≈ Jun 16
    "hol_independence": (7, 4),
    "hol_labor": (9, 2),  # 1st Monday ≈ Sep 2
    "hol_halloween": (10, 31),
    "hol_columbus": (10, 14),  # 2nd Monday ≈ Oct 14
    "hol_veterans": (11, 11),
    "hol_thanksgiving": (11, 28),  # 4th Thursday ≈ Nov 28
    "hol_black_friday": (11, 29),
    "hol_christmas": (12, 25),
}
_HOLIDAY_WINDOW_WEEKS = 1  # flag is 1 if week contains the holiday ±3 days


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeaturesResult:
    """Output of build_features()."""

    features: pd.DataFrame
    """Dense grid enriched with all feature columns; 17,539 × 98."""

    feature_cols: list[str]
    """Ordered list of the 90 engineered feature column names."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shift_per_sku(df: pd.DataFrame, col: str, periods: int) -> pd.Series:
    """Lag *col* by *periods* weeks within each SKU group."""
    return df.groupby(config.COL_SKU_ID)[col].shift(periods)


def _rolling_per_sku(
    df: pd.DataFrame,
    col: str,
    window: int,
    func: str = "mean",
    min_periods: int = 1,
) -> pd.Series:
    """Rolling statistic on *col* within each SKU, already shifted by 1 (no leakage)."""
    rolled = (
        df.groupby(config.COL_SKU_ID)[col]
        .shift(1)
        .groupby(df[config.COL_SKU_ID])
        .transform(lambda s: s.rolling(window, min_periods=min_periods).agg(func))
    )
    return rolled


def _fourier_features(timestamps: pd.Series, period: int, max_k: int) -> dict[str, np.ndarray]:
    """Return sin/cos Fourier pairs for ISO-week index with given period and harmonics."""
    week = timestamps.dt.isocalendar().week.astype(float).values
    out: dict[str, np.ndarray] = {}
    for k in range(1, max_k + 1):
        angle = 2.0 * np.pi * k * week / period
        out[f"fourier_{period}w_sin_k{k}"] = np.sin(angle)
        out[f"fourier_{period}w_cos_k{k}"] = np.cos(angle)
    return out


def _holiday_flags(timestamps: pd.Series) -> dict[str, np.ndarray]:
    """Binary flag per week: 1 if week contains the holiday within ±3 days."""
    n = len(timestamps)
    flags: dict[str, np.ndarray] = {}
    for name, (mm, dd) in _HOLIDAY_MMDD.items():
        arr = np.zeros(n, dtype=np.float32)
        for i, ts in enumerate(timestamps):
            holiday_approx = pd.Timestamp(year=ts.year, month=mm, day=dd)
            if abs((ts - holiday_approx).days) <= 3 + 7:  # within ±10 days
                arr[i] = 1.0
        flags[name] = arr
    return flags


def _static_features(grp: pd.DataFrame) -> dict[str, float]:
    """Compute per-SKU static demand-pattern statistics."""
    sales = grp[config.COL_SALES].values
    n = len(sales)

    # Zero rate
    zero_rate = float((sales == 0).mean())

    # IDI (Inter-Demand Interval): mean gap between non-zero demands
    nz_idx = np.where(sales > 0)[0]
    if len(nz_idx) >= 2:
        idi = float(np.diff(nz_idx).mean())
    elif len(nz_idx) == 1:
        idi = float(n)
    else:
        idi = float(n)

    # CV2 (squared coefficient of variation of non-zero demands)
    nz_vals = sales[sales > 0]
    if len(nz_vals) >= 2:
        cv2 = float((nz_vals.std() / nz_vals.mean()) ** 2)
    else:
        cv2 = 0.0

    # Gini coefficient
    if n > 1 and sales.sum() > 0:
        s = np.sort(sales)
        idx = np.arange(1, n + 1)
        gini = float((2 * (idx * s).sum() / (n * s.sum())) - (n + 1) / n)
        gini = max(0.0, min(1.0, gini))
    else:
        gini = 0.0

    # Hurst exponent (R/S method, simplified)
    if n >= 20 and sales.std() > 0:
        mean = sales.mean()
        deviations = np.cumsum(sales - mean)
        r = deviations.max() - deviations.min()
        s_val = sales.std()
        hurst = float(np.log(r / s_val + 1e-9) / np.log(n))
        hurst = max(0.0, min(1.0, hurst))
    else:
        hurst = 0.5

    # ABC tier: A=top 20% revenue, B=next 30%, C=bottom 50%
    total_rev = float(sales.sum())

    return {
        "idi": idi,
        "cv2": cv2,
        "zero_rate": zero_rate,
        "gini": gini,
        "hurst": hurst,
        "_total_rev": total_rev,  # used for ABC assignment, dropped after
    }


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def build_features(
    dense: pd.DataFrame,
    lifecycle: LifecycleResult,
) -> FeaturesResult:
    """Build all feature columns on the dense weekly grid.

    Parameters
    ----------
    dense:
        Output of densify() — 17,539 rows, Saturday-dated (after relabelling).
    lifecycle:
        Output of infer_lifecycle() — provides is_active / is_dormant flags.

    Returns
    -------
    FeaturesResult with features DataFrame (17,539 × 98) and feature_cols list.
    """
    df = dense.copy().sort_values([config.COL_SKU_ID, config.COL_TIMESTAMP]).reset_index(drop=True)
    feature_cols: list[str] = []

    # -------------------------------------------------------------------------
    # A. Sales lags
    # -------------------------------------------------------------------------
    for lag in [1, 2, 3, 4, 5, 6, 8, 13, 26, 52]:
        col = f"lag_{lag}"
        df[col] = _shift_per_sku(df, config.COL_SALES, lag)
        feature_cols.append(col)

    # -------------------------------------------------------------------------
    # B. Rolling means  (shift-1 inside helper → no leakage)
    # -------------------------------------------------------------------------
    for w in [4, 8, 13, 26, 52]:
        col = f"roll{w}_mean"
        df[col] = _rolling_per_sku(df, config.COL_SALES, w)
        feature_cols.append(col)

    # -------------------------------------------------------------------------
    # C. Rolling std
    # -------------------------------------------------------------------------
    for w in [4, 8, 13, 26, 52]:
        col = f"roll{w}_std"
        df[col] = _rolling_per_sku(df, config.COL_SALES, w, func="std")
        feature_cols.append(col)

    # -------------------------------------------------------------------------
    # D. Rolling max / min
    # -------------------------------------------------------------------------
    for w in [4, 13]:
        for func in ("max", "min"):
            col = f"roll{w}_{func}"
            df[col] = _rolling_per_sku(df, config.COL_SALES, w, func=func)
            feature_cols.append(col)

    # -------------------------------------------------------------------------
    # E. Rolling discount (lagged)
    # -------------------------------------------------------------------------
    for w in [4, 13]:
        col = f"discount_roll{w}_mean"
        df[col] = _rolling_per_sku(df, config.COL_DISCOUNT_PCT, w)
        feature_cols.append(col)

    # -------------------------------------------------------------------------
    # F. Log1p transforms of rolling means + key lags
    # -------------------------------------------------------------------------
    for w in [4, 8, 13, 26]:
        col = f"log1p_roll{w}_mean"
        df[col] = np.log1p(df[f"roll{w}_mean"].fillna(0.0))
        feature_cols.append(col)
    for lag in [1, 4, 13]:
        col = f"log1p_lag{lag}"
        df[col] = np.log1p(df[f"lag_{lag}"].fillna(0.0))
        feature_cols.append(col)

    # -------------------------------------------------------------------------
    # G. Log price
    # -------------------------------------------------------------------------
    df["log1p_list_price"] = np.log1p(df[config.COL_LIST_PRICE].fillna(0.0))
    feature_cols.append("log1p_list_price")

    # -------------------------------------------------------------------------
    # H. Momentum: sales[t-1] / roll_mean[t-shift] - 1
    # -------------------------------------------------------------------------
    for w in [4, 13, 26, 52]:
        col = f"mom{w}"
        roll = df[f"roll{w}_mean"].replace(0, np.nan)
        df[col] = (df["lag_1"] / roll - 1.0).fillna(0.0)
        feature_cols.append(col)

    # -------------------------------------------------------------------------
    # I/J/K. Fourier features (ISO-week based)
    # -------------------------------------------------------------------------
    for period, max_k in [(52, 4), (26, 2), (13, 4)]:
        fourier = _fourier_features(df[config.COL_TIMESTAMP], period=period, max_k=max_k)
        for col, arr in fourier.items():
            df[col] = arr
            feature_cols.append(col)

    # -------------------------------------------------------------------------
    # L. US holiday flags
    # -------------------------------------------------------------------------
    hol_flags = _holiday_flags(df[config.COL_TIMESTAMP])
    for col, arr in hol_flags.items():
        df[col] = arr
        feature_cols.append(col)

    # -------------------------------------------------------------------------
    # M. Promo / price features (lagged → no leakage)
    # -------------------------------------------------------------------------
    df["discount_pct_lag1"] = _shift_per_sku(df, config.COL_DISCOUNT_PCT, 1)
    feature_cols.append("discount_pct_lag1")

    df["list_price_lag1"] = _shift_per_sku(df, config.COL_LIST_PRICE, 1)
    feature_cols.append("list_price_lag1")

    df["list_price_roll4"] = _rolling_per_sku(df, config.COL_LIST_PRICE, 4)
    feature_cols.append("list_price_roll4")

    df["list_price_roll13"] = _rolling_per_sku(df, config.COL_LIST_PRICE, 13)
    feature_cols.append("list_price_roll13")

    price_roll13 = df["list_price_roll13"].replace(0, np.nan)
    df["price_vs_roll13"] = (df["list_price_lag1"] / price_roll13 - 1.0).fillna(0.0)
    feature_cols.append("price_vs_roll13")

    # -------------------------------------------------------------------------
    # N. Calendar features
    # -------------------------------------------------------------------------
    df["week_of_year"] = df[config.COL_TIMESTAMP].dt.isocalendar().week.astype(float)
    feature_cols.append("week_of_year")

    df["month"] = df[config.COL_TIMESTAMP].dt.month.astype(float)
    feature_cols.append("month")

    df["quarter"] = df[config.COL_TIMESTAMP].dt.quarter.astype(float)
    feature_cols.append("quarter")

    # weeks_since_first_sale: computed from lifecycle
    lc_first = lifecycle.lifecycle.set_index(config.COL_SKU_ID)["first_sale"]
    df["weeks_since_first_sale"] = df.apply(
        lambda r: max(
            0.0,
            (
                r[config.COL_TIMESTAMP]
                - lc_first.get(r[config.COL_SKU_ID], r[config.COL_TIMESTAMP])
            ).days
            / 7.0,
        ),
        axis=1,
    )
    feature_cols.append("weeks_since_first_sale")

    df["sku_age_weeks"] = ((df[config.COL_TIMESTAMP] - _CATALOG_EPOCH).dt.days / 7.0).clip(
        lower=0.0
    )
    feature_cols.append("sku_age_weeks")

    # -------------------------------------------------------------------------
    # O. is_q4
    # -------------------------------------------------------------------------
    df["is_q4"] = (df["quarter"] == 4).astype(float)
    feature_cols.append("is_q4")

    # -------------------------------------------------------------------------
    # P. Per-SKU static features (computed over full history, leakage-safe for statics)
    # -------------------------------------------------------------------------
    static_rows = []
    for sku, grp in df.groupby(config.COL_SKU_ID):
        stats = _static_features(grp)
        stats[config.COL_SKU_ID] = sku
        static_rows.append(stats)

    statics_df = pd.DataFrame(static_rows)

    # ABC tier based on total revenue
    rev_sorted = statics_df["_total_rev"].rank(pct=True)
    statics_df["abc_tier_enc"] = pd.cut(
        rev_sorted,
        bins=[0, 0.5, 0.8, 1.0],
        labels=[0, 1, 2],  # C=0, B=1, A=2
        include_lowest=True,
    ).astype(float)
    statics_df = statics_df.drop(columns=["_total_rev"])

    static_cols = ["idi", "cv2", "zero_rate", "gini", "hurst", "abc_tier_enc"]
    df = df.merge(statics_df[[config.COL_SKU_ID] + static_cols], on=config.COL_SKU_ID, how="left")
    feature_cols.extend(static_cols)

    # -------------------------------------------------------------------------
    # R. Promo elasticity features (per-SKU, computed from full history)
    # These are static per-SKU values — leakage-safe as they use full history
    # -------------------------------------------------------------------------
    elast_rows = []
    for sku, grp in df.groupby(config.COL_SKU_ID):
        y    = grp[config.COL_SALES].values
        disc = grp[config.COL_DISCOUNT_PCT].values if config.COL_DISCOUNT_PCT in grp else np.zeros(len(y))

        # Promo lift: mean sales during high-promo vs baseline
        high_mask = disc > 0.20
        base_mask = disc < 0.08
        promo_mean   = float(y[high_mask].mean()) if high_mask.sum() >= 2 else float(y.mean())
        baseline_mean= float(y[base_mask].mean()) if base_mask.sum() >= 2 else max(float(y.mean()), 1e-6)
        promo_lift   = float(np.clip(promo_mean / (baseline_mean + 1e-6), 0.5, 10.0))

        # Promo frequency: fraction of weeks with discount > 0.15
        promo_freq = float(high_mask.mean())

        # Demand volatility on promo vs non-promo weeks
        promo_cv  = float(y[high_mask].std() / (y[high_mask].mean() + 1e-6)) if high_mask.sum() >= 4 else 0.0
        nopromo_cv= float(y[base_mask].std() / (y[base_mask].mean() + 1e-6)) if base_mask.sum() >= 4 else 0.0

        # Elasticity: corr(sales, discount_pct) — positive means promo drives demand
        if len(y) >= 13 and disc.std() > 1e-6:
            corr = float(np.corrcoef(y, disc)[0, 1])
        else:
            corr = 0.0

        elast_rows.append({
            config.COL_SKU_ID: sku,
            "promo_lift":    promo_lift,
            "promo_freq":    promo_freq,
            "promo_cv":      promo_cv,
            "nopromo_cv":    nopromo_cv,
            "promo_elasticity": corr,
        })

    elast_df = pd.DataFrame(elast_rows)
    elast_cols = ["promo_lift", "promo_freq", "promo_cv", "nopromo_cv", "promo_elasticity"]
    df = df.merge(elast_df[[config.COL_SKU_ID] + elast_cols], on=config.COL_SKU_ID, how="left")
    feature_cols.extend(elast_cols)

    # -------------------------------------------------------------------------
    # Final NaN fill: lag/rolling columns are NaN at series start — fill with 0
    # -------------------------------------------------------------------------
    lag_roll_cols = [
        c
        for c in feature_cols
        if c.startswith(
            (
                "lag_",
                "roll",
                "log1p_lag",
                "log1p_roll",
                "mom",
                "discount_roll",
                "list_price_roll",
                "list_price_lag",
                "discount_pct_lag",
                "price_vs",
            )
        )
    ]
    df[lag_roll_cols] = df[lag_roll_cols].fillna(0.0)

    # -------------------------------------------------------------------------
    # Q. Promo / regime / external features (V2)
    # -------------------------------------------------------------------------

    # Q1. Is on promo this week (lagged 1w to avoid leakage)
    df["is_on_promo"] = (df["discount_pct_lag1"] > 0.15).astype(float)
    feature_cols.append("is_on_promo")

    # Q2. Promo frequency over last 13 weeks
    df["promo_freq_13w"] = _rolling_per_sku(df, config.COL_DISCOUNT_PCT, 13,
                                             func="mean")
    df["promo_freq_13w"] = (df["promo_freq_13w"] > 0.15).astype(float)
    feature_cols.append("promo_freq_13w")

    # Q3. Price drop flag: current price < 85% of 13w average
    df["price_drop_flag"] = (df["price_vs_roll13"] < -0.15).astype(float)
    feature_cols.append("price_drop_flag")

    # Q4. Post-promo hangover: was on promo 2w ago, not on promo 1w ago
    disc_lag2 = _shift_per_sku(df, config.COL_DISCOUNT_PCT, 2)
    df["post_promo_flag"] = (
        (disc_lag2 > 0.15) & (df["discount_pct_lag1"] <= 0.15)
    ).astype(float)
    feature_cols.append("post_promo_flag")

    # Q5. Demand acceleration: roll4_mean / roll13_mean (is demand speeding up?)
    roll13_safe = df["roll13_mean"].replace(0, np.nan)
    df["acceleration"] = (df["roll4_mean"] / roll13_safe).fillna(1.0).clip(0.0, 5.0)
    feature_cols.append("acceleration")

    # Q6. YoY growth proxy: roll13_mean / lag_52-based 13w mean
    lag52_roll = _shift_per_sku(df, config.COL_SALES, 52)
    df["yoy_growth"] = (df["roll4_mean"] / lag52_roll.replace(0, np.nan)).fillna(1.0).clip(0.1, 10.0)
    feature_cols.append("yoy_growth")

    # Q7. Weeks to Christmas (proximity signal, cycles annually)
    woy = df[config.COL_TIMESTAMP].dt.isocalendar().week.astype(float)
    df["weeks_to_christmas"] = ((52 - woy) % 52).clip(0, 26).astype(float)
    feature_cols.append("weeks_to_christmas")

    # Q8. Is Q4 ramp (Oct-Dec buildup, weeks 40-52)
    df["is_q4_ramp"] = woy.between(40, 52).astype(float)
    feature_cols.append("is_q4_ramp")

    # Q9. Is post-holiday (Jan-Feb slowdown, weeks 1-8)
    df["is_post_holiday"] = woy.between(1, 8).astype(float)
    feature_cols.append("is_post_holiday")

    # Q10. Demand regime (rule-based: 0=baseline,1=promo,2=post-promo,3=launch,4=decline)
    regime = np.zeros(len(df), dtype=float)
    regime[df["is_on_promo"] > 0] = 1.0
    regime[df["post_promo_flag"] > 0] = 2.0
    regime[df["weeks_since_first_sale"] < 13] = 3.0
    decline_mask = (df["acceleration"] < 0.5) & (df["roll13_mean"] > 0)
    regime[decline_mask] = 4.0
    df["demand_regime"] = regime
    feature_cols.append("demand_regime")

    # Q11. Promo-sales correlation proxy (rolling 13w corr of sales and discount)
    # Computed as: roll13 of (sales * discount_pct) / (roll13_sales * roll13_disc)
    # Simplified: if promo_freq_13w > 0.3 AND roll13_mean > global_mean → promo_driven_flag
    global_mean = df.groupby(config.COL_TIMESTAMP)[config.COL_SALES].transform("mean")
    df["promo_driven_flag"] = (
        (df["promo_freq_13w"] > 0.0) & (df["roll13_mean"] > global_mean * 0.5)
    ).astype(float)
    feature_cols.append("promo_driven_flag")

    # -------------------------------------------------------------------------
    # Final NaN fill for new features
    # -------------------------------------------------------------------------
    new_feat_cols = [
        "is_on_promo", "promo_freq_13w", "price_drop_flag", "post_promo_flag",
        "acceleration", "yoy_growth", "weeks_to_christmas", "is_q4_ramp",
        "is_post_holiday", "demand_regime", "promo_driven_flag",
    ]
    df[new_feat_cols] = df[new_feat_cols].fillna(0.0)

    lag_roll_cols = [
        c
        for c in feature_cols
        if c.startswith(
            (
                "lag_",
                "roll",
                "log1p_lag",
                "log1p_roll",
                "mom",
                "discount_roll",
                "list_price_roll",
                "list_price_lag",
                "discount_pct_lag",
                "price_vs",
            )
        )
    ]
    df[lag_roll_cols] = df[lag_roll_cols].fillna(0.0)

    logger.info(
        "Features: %d rows × %d cols (%d feature cols)",
        len(df),
        len(df.columns),
        len(feature_cols),
    )

    return FeaturesResult(features=df, feature_cols=feature_cols)


# ---------------------------------------------------------------------------
# Phase 5b: Cluster-context features (LOO aggregates)
# ---------------------------------------------------------------------------

#: Names of the 7 LOO cluster aggregates (in order added)
CLUSTER_FEATURE_COLS: list[str] = [
    "cluster_loo_lag1_mean",
    "cluster_loo_lag1_sum",
    "cluster_loo_lag1_std",
    "cluster_loo_lag1_median",
    "cluster_loo_roll4_mean",
    "cluster_loo_roll13_mean",
    "cluster_loo_nonzero_rate",
]


def add_cluster_features(
    features: pd.DataFrame,
    segments: pd.DataFrame,
) -> pd.DataFrame:
    """Append cluster_id + 7 LOO lag-1 cluster aggregates to the feature grid.

    Leave-One-Out (LOO) means: for SKU i in cluster c at time t, the aggregate
    is computed over all OTHER SKUs in the cluster, so no self-leakage.

    Leakage note: all aggregates use lag_1 / roll4_mean / roll13_mean which are
    already shifted — no contemporaneous signal flows in.

    Median LOO: exact LOO median requires removing self from a sorted array;
    for clusters with >= 5 members the cluster median is used directly
    (self-influence is <20% of the sorted sequence). For tiny clusters (< 5),
    the LOO sum/mean are already exact and median is set to the LOO mean.

    Parameters
    ----------
    features:
        Output of build_features() — 17,539 × 98.
    segments:
        Output of segment_and_cluster() — one row per SKU with cluster_id.

    Returns
    -------
    DataFrame with 8 additional columns: cluster_id + 7 LOO aggregates.
    Shape: 17,539 × 106.
    """
    df = features.merge(
        segments[[config.COL_SKU_ID, "cluster_id"]],
        on=config.COL_SKU_ID,
        how="left",
    )
    df = df.sort_values([config.COL_SKU_ID, config.COL_TIMESTAMP]).reset_index(drop=True)

    grp_key = ["cluster_id", config.COL_TIMESTAMP]

    # ── LOO lag_1 stats ───────────────────────────────────────────────────
    # Cluster totals at each (cluster, timestamp)
    ct_sum = df.groupby(grp_key)["lag_1"].transform("sum")
    ct_count = df.groupby(grp_key)["lag_1"].transform("count")
    ct_sum_sq = df.groupby(grp_key)["lag_1"].transform(lambda x: (x**2).sum())
    ct_nz = df.groupby(grp_key)["lag_1"].transform(lambda x: (x > 0).sum())

    loo_sum = ct_sum - df["lag_1"]
    loo_count = (ct_count - 1).clip(lower=1)

    df["cluster_loo_lag1_mean"] = (loo_sum / loo_count).fillna(0.0)
    df["cluster_loo_lag1_sum"] = loo_sum.fillna(0.0)

    # LOO variance: Var_LOO = (sum_sq - self^2 - loo_sum^2/loo_count) / (loo_count - 1)
    loo_sum_sq = ct_sum_sq - df["lag_1"] ** 2
    loo_var_num = loo_sum_sq - (loo_sum**2 / loo_count)
    loo_var = (loo_var_num / (loo_count - 1).clip(lower=1)).clip(lower=0.0)
    df["cluster_loo_lag1_std"] = np.sqrt(loo_var).fillna(0.0)

    # LOO median: use cluster median for large clusters; LOO mean for tiny ones
    ct_median = df.groupby(grp_key)["lag_1"].transform("median")
    small_cluster = ct_count <= 4
    df["cluster_loo_lag1_median"] = np.where(
        small_cluster,
        df["cluster_loo_lag1_mean"],  # exact LOO mean for tiny clusters
        ct_median,  # cluster median ≈ LOO median for larger ones
    )

    # LOO nonzero rate: fraction of OTHER cluster members with lag_1 > 0
    loo_nz = ct_nz - (df["lag_1"] > 0).astype(float)
    df["cluster_loo_nonzero_rate"] = (loo_nz / loo_count).fillna(0.0).clip(0.0, 1.0)

    # ── LOO rolling cluster aggregates (use per-SKU roll cols) ────────────
    for roll_col, out_col in [
        ("roll4_mean", "cluster_loo_roll4_mean"),
        ("roll13_mean", "cluster_loo_roll13_mean"),
    ]:
        ct_roll_sum = df.groupby(grp_key)[roll_col].transform("sum")
        loo_roll_sum = ct_roll_sum - df[roll_col]
        df[out_col] = (loo_roll_sum / loo_count).fillna(0.0)

    logger.info(
        "Cluster features: %d rows × %d cols (+cluster_id + 7 LOO cols = 106 total)",
        len(df),
        len(df.columns),
    )

    return df


# ---------------------------------------------------------------------------
# Phase 6b: Hierarchy-context features
# ---------------------------------------------------------------------------

#: Names of the 8 hierarchy-context columns (in order added)
HIERARCHY_FEATURE_COLS: list[str] = [
    "hier_pt_loo_lag1_mean",
    "hier_pt_loo_roll4_mean",
    "hier_pt_loo_roll13_mean",
    "hier_pt_loo_nonzero_rate",
    "hier_total_lag1_mean",
    "hier_total_roll52_mean",
    "hier_total_yoy",
    "hier_pt_n_variants",
]


def add_hierarchy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append 8 hierarchy-context columns to the 106-column cluster feature grid.

    Leakage: all aggregates are built from lag_1 / roll4_mean / roll52_mean —
    already shifted — so no contemporaneous self-signal leaks in.

    Locked tweak (doc §5): total-level YoY is NON-LOO.  At the portfolio level
    the LOO adjustment would amplify ratio noise (the portfolio minus one SKU
    is almost identical to the full portfolio, so the denominator is nearly
    unchanged but any numerical error gets magnified in the ratio).  Total-level
    features are therefore computed over the full cross-section.

    Parameters
    ----------
    df:
        Output of add_cluster_features() — 17,539 × 106.
        Must contain product_type, lag_1, roll4_mean, roll13_mean, roll52_mean.

    Returns
    -------
    DataFrame with 8 additional columns. Shape: 17,539 × 114.
    """
    df = df.copy().sort_values([config.COL_SKU_ID, config.COL_TIMESTAMP]).reset_index(drop=True)

    # Normalise product_type: fill nulls with 'unknown' so groupby is stable
    df["product_type"] = df["product_type"].fillna("unknown")

    # ── L1 (product_type) LOO aggregates ─────────────────────────────────────
    grp_pt = ["product_type", config.COL_TIMESTAMP]

    ct_sum = df.groupby(grp_pt)["lag_1"].transform("sum")
    ct_count = df.groupby(grp_pt)["lag_1"].transform("count")
    ct_nz = df.groupby(grp_pt)["lag_1"].transform(lambda x: (x > 0).sum())
    loo_sum = ct_sum - df["lag_1"]
    loo_count = (ct_count - 1).clip(lower=1)

    df["hier_pt_loo_lag1_mean"] = (loo_sum / loo_count).fillna(0.0)

    for roll_col, out_col in [
        ("roll4_mean", "hier_pt_loo_roll4_mean"),
        ("roll13_mean", "hier_pt_loo_roll13_mean"),
    ]:
        ct_roll = df.groupby(grp_pt)[roll_col].transform("sum")
        df[out_col] = ((ct_roll - df[roll_col]) / loo_count).fillna(0.0)

    loo_nz = ct_nz - (df["lag_1"] > 0).astype(float)
    df["hier_pt_loo_nonzero_rate"] = (loo_nz / loo_count).clip(0.0, 1.0).fillna(0.0)

    # ── L0 (total portfolio) NON-LOO aggregates ───────────────────────────────
    df["hier_total_lag1_mean"] = (
        df.groupby(config.COL_TIMESTAMP)["lag_1"].transform("mean").fillna(0.0)
    )
    df["hier_total_roll52_mean"] = (
        df.groupby(config.COL_TIMESTAMP)["roll52_mean"].transform("mean").fillna(0.0)
    )

    # YoY: mean(portfolio lag_1 at t) / mean(portfolio lag_1 at t-52).
    # Non-LOO: computed over all SKUs, no self-exclusion.
    # Clipped to [0.1, 10] to avoid extreme ratios from sparse early history.
    total_ts = (
        df.groupby(config.COL_TIMESTAMP)["lag_1"]
        .mean()
        .reset_index()
        .rename(columns={"lag_1": "_tot"})
        .sort_values(config.COL_TIMESTAMP)
    )
    total_ts["_tot_y1"] = total_ts["_tot"].shift(52)
    total_ts["hier_total_yoy"] = (
        (total_ts["_tot"] / total_ts["_tot_y1"].replace(0.0, np.nan)).fillna(1.0).clip(0.1, 10.0)
    )
    df = df.merge(
        total_ts[[config.COL_TIMESTAMP, "hier_total_yoy"]],
        on=config.COL_TIMESTAMP,
        how="left",
    )
    df["hier_total_yoy"] = df["hier_total_yoy"].fillna(1.0)

    # ── Static hierarchy feature ──────────────────────────────────────────────
    # Number of SKUs in this product_type group (stable across time).
    pt_size = df.groupby("product_type")[config.COL_SKU_ID].transform("nunique").astype(float)
    df["hier_pt_n_variants"] = pt_size.fillna(1.0)

    logger.info(
        "Hierarchy features: %d rows × %d cols (+8 hier cols = 114 total)",
        len(df),
        len(df.columns),
    )

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Persistence helper
# ---------------------------------------------------------------------------


def save_features(result: FeaturesResult, path: None = None) -> None:
    """Write feature DataFrame to data/processed/features.parquet."""
    import pathlib

    out = pathlib.Path(path) if path else config.DATA_PROCESSED / "features.parquet"
    result.features.to_parquet(out, index=False)
    logger.info("Wrote features → %s", out)
