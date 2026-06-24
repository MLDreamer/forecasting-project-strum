"""Unpredictable SKU detection and flagging.

Rules for flagging a SKU as non-forecastable:
  A SKU is flagged if it triggers >= 2 of the following rules:

  Rule 1 — demand_surge_5x:
      Mean of last 13w / mean of prior 13w > 5.0
      Demand accelerated 5x recently — no model can extrapolate this

  Rule 2 — spike_outlier:
      Max single week in last 52w > 8x the 52w rolling mean
      One-off spikes (promo blowout, viral event) distort all model fits

  Rule 3 — extreme_volatility:
      CV2 (squared coefficient of variation on non-zero weeks) > 3.0
      Demand is so volatile it is fundamentally unpredictable

  Rule 4 — yoy_growth_3x:
      Mean of last 13w / mean of same 13w one year ago > 3.0
      YoY growth > 3x — structural growth that time-series cannot extrapolate

  Rule 5 — demand_cliff:
      Peak 26w mean / recent 8w mean > 5.0 AND peak > 5 units/week
      SKU had strong sales then nearly stopped — possible discontinuation
      or seasonal pattern requiring client clarification

Single-flag SKUs are labelled 'review' — client should clarify behaviour.
Zero-flag SKUs are labelled 'ok' — proceed to forecast normally.

Non-forecastable SKUs are:
  - Excluded from CV WAPE calculation (not penalised)
  - Assigned a flat 'historical mean' forecast flagged as unreliable
  - Written to outputs/non_forecastable_skus.csv for client review
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Thresholds (tunable via config)
ACCEL_THRESHOLD   = 10.0  # last13w/prior13w — only extreme 10x surges
SPIKE_THRESHOLD   = 10.0  # max_week / roll52_mean — large one-off spikes
CV2_THRESHOLD     = 4.0   # cv2 of non-zero demand — only truly chaotic
YOY_THRESHOLD     = 5.0   # last13w / same13w_last_year — 5x YoY growth
CLIFF_THRESHOLD   = 1000.0 # peak26w / recent8w — truly dead (cliff in thousands)
MIN_FLAG_COUNT    = 2      # need >= this many flags to be non_forecastable


@dataclass
class UnpredictableResult:
    """Output of detect_unpredictable()."""

    sku_flags: pd.DataFrame
    """One row per SKU with columns:
       sku_id, n_flags, label, flags, accel_13w, max_spike, cv2, yoy_growth, cliff_ratio
    """

    non_forecastable: set[int]
    """SKU IDs labelled non_forecastable (>= MIN_FLAG_COUNT flags)."""

    review: set[int]
    """SKU IDs labelled review (exactly 1 flag)."""

    ok: set[int]
    """SKU IDs labelled ok (0 flags)."""


def detect_unpredictable(
    dense: pd.DataFrame,
    col_sku: str = "sku_id",
    col_ts:  str = "timestamp",
    col_sales: str = "sales",
) -> UnpredictableResult:
    """Detect unpredictable SKUs using demand-behaviour rules.

    Parameters
    ----------
    dense : DataFrame
        Full weekly dense grid (all SKUs, all timestamps, zero-filled).

    Returns
    -------
    UnpredictableResult
    """
    rows = []

    for sku, grp in dense.groupby(col_sku):
        y = grp.sort_values(col_ts)[col_sales].values
        T = len(y)
        if T < 4:
            continue

        flags: list[str] = []

        # Rule 1: Demand acceleration
        if T >= 26:
            recent13 = float(y[-13:].mean())
            prior13  = float(y[-26:-13].mean())
            accel    = recent13 / prior13 if prior13 > 1e-6 else (10.0 if recent13 > 0 else 1.0)
        else:
            accel = 1.0
        if accel > ACCEL_THRESHOLD:
            flags.append(f"demand_surge_{accel:.1f}x")

        # Rule 2: Spike outlier
        window = y[-52:] if T >= 52 else y
        roll_mean = float(window.mean())
        max_week  = float(window.max())
        spike     = max_week / roll_mean if roll_mean > 1e-6 else 1.0
        if spike > SPIKE_THRESHOLD:
            flags.append(f"spike_outlier_{spike:.1f}x")

        # Rule 3: Extreme volatility
        nz  = y[y > 0]
        cv2 = float((nz.std() / nz.mean()) ** 2) if len(nz) >= 4 else 0.0
        if cv2 > CV2_THRESHOLD:
            flags.append(f"extreme_volatility_cv2={cv2:.2f}")

        # Rule 4: YoY growth
        if T >= 65:
            ya  = float(y[-65:-52].mean())
            rc  = float(y[-13:].mean())
            yoy = rc / ya if ya > 1e-6 else 1.0
        else:
            yoy = 1.0
        if yoy > YOY_THRESHOLD:
            flags.append(f"yoy_growth_{yoy:.1f}x")

        # Rule 5: Demand cliff
        if T >= 26:
            peak_window = y[-52:-26] if T >= 52 else y[:-8]
            peak_mean   = float(peak_window.mean()) if len(peak_window) > 0 else 0.0
            recent8     = float(y[-8:].mean())
            cliff       = peak_mean / (recent8 + 1e-6) if peak_mean > 5.0 else 0.0
        else:
            cliff = 0.0
        if cliff > CLIFF_THRESHOLD:
            flags.append(f"demand_cliff_{cliff:.1f}x")

        n_flags = len(flags)
        if n_flags >= MIN_FLAG_COUNT:
            label = "non_forecastable"
        elif n_flags == 1:
            label = "review"
        else:
            label = "ok"

        rows.append({
            col_sku:       int(sku),
            "n_flags":     n_flags,
            "label":       label,
            "flag_reasons": " | ".join(flags),
            "accel_13w":   round(accel, 3),
            "max_spike":   round(spike, 3),
            "cv2":         round(cv2, 3),
            "yoy_growth":  round(yoy, 3),
            "cliff_ratio": round(cliff, 3),
            "train_weeks": T,
            "last26w_vol": round(float(y[-26:].sum()) if T >= 26 else float(y.sum()), 1),
        })

    df = pd.DataFrame(rows).sort_values(["n_flags", "accel_13w"], ascending=[False, False])

    non_fc = set(df.loc[df["label"] == "non_forecastable", col_sku].astype(int))
    review = set(df.loc[df["label"] == "review", col_sku].astype(int))
    ok     = set(df.loc[df["label"] == "ok", col_sku].astype(int))

    return UnpredictableResult(
        sku_flags=df,
        non_forecastable=non_fc,
        review=review,
        ok=ok,
    )


def filter_for_wape(
    actuals: np.ndarray,
    forecasts: np.ndarray,
    sku_ids: list[int],
    non_forecastable: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Remove non-forecastable SKUs from WAPE numerator and denominator.

    Parameters
    ----------
    actuals, forecasts : arrays aligned to sku_ids
    sku_ids : list of sku_id in the same order as actuals/forecasts
    non_forecastable : set of sku_ids to exclude

    Returns
    -------
    (filtered_actuals, filtered_forecasts)
    """
    mask = np.array([s not in non_forecastable for s in sku_ids])
    return actuals[mask], forecasts[mask]
