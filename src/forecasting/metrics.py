"""Forecast evaluation metrics.

All metrics accept numpy arrays and support optional revenue weighting and
per-horizon slicing.  No pandas dependency — pure numpy throughout so they
can be called inside tight CV loops without DataFrame overhead.

Metrics implemented
-------------------
- wape          Revenue-weighted Absolute Percentage Error (selection metric)
- mase          Mean Absolute Scaled Error
- pinball       Pinball / quantile loss per quantile level
- crps          Continuous Ranked Probability Score (proper scoring rule)
- wis           Weighted Interval Score (calibration-aware)
- coverage      Empirical interval coverage at one or more PI levels
- smape         Symmetric Mean Absolute Percentage Error

All scalar-returning functions also have a `_per_horizon` variant that returns
a 1-D array of shape (H,) — one value per forecast horizon step.

Conventions
-----------
- `y_true`   shape (N,)  or (N, H)  — actual sales
- `y_pred`   shape (N,)  or (N, H)  — point forecast (P50)
- `quantiles` shape (N, Q) or (N, H, Q) — Q quantile forecasts
- `q_levels` shape (Q,)  — quantile probabilities, e.g. [0.1, 0.5, 0.9]
- `weights`  shape (N,)  or (N, H) — revenue weights (default: uniform)
- All inputs are non-negative (sales >= 0).
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _weights(
    weights: np.ndarray | None,
    shape: tuple[int, ...],
) -> np.ndarray:
    """Return uniform weights if None; otherwise validate and normalise."""
    if weights is None:
        w = np.ones(shape[0], dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.shape[0] != shape[0]:
            raise ValueError(f"weights length {w.shape[0]} != N={shape[0]}")
    total = w.sum()
    return w / total if total > 0 else w


def _broadcast_weights(
    w: np.ndarray,  # shape (N,)
    H: int,
) -> np.ndarray:
    """Tile per-SKU weights across H horizon steps -> shape (N, H)."""
    return np.tile(w[:, None], (1, H))  # (N, H)


# ---------------------------------------------------------------------------
# WAPE — revenue-weighted absolute percentage error
# ---------------------------------------------------------------------------


def wape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Revenue-weighted WAPE.

    WAPE = sum(w * |y_true - y_pred|) / sum(w * y_true)

    When all y_true == 0, returns 0.0 (no demand, no error to measure).
    Weights default to y_true (revenue weighting = weight by actual demand).
    If explicit weights are supplied they override the default revenue weight.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    if weights is None:
        w = y_true.copy()
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()

    denom = w.sum()
    if denom == 0.0:
        return 0.0
    return float(np.dot(w, np.abs(y_true - y_pred)) / denom)


def wape_per_horizon(
    y_true: np.ndarray,  # (N, H)
    y_pred: np.ndarray,  # (N, H)
    weights: np.ndarray | None = None,  # (N,)
) -> np.ndarray:
    """Per-horizon WAPE.  Returns shape (H,)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    N, H = y_true.shape
    w = _weights(weights, (N,))
    result = np.empty(H)
    for h in range(H):
        wh = y_true[:, h] if weights is None else w
        denom = wh.sum()
        result[h] = (np.dot(wh, np.abs(y_true[:, h] - y_pred[:, h])) / denom) if denom > 0 else 0.0
    return result


# ---------------------------------------------------------------------------
# MASE — mean absolute scaled error
# ---------------------------------------------------------------------------


def mase(
    y_true: np.ndarray,  # (N,) or (N, H)
    y_pred: np.ndarray,  # (N,) or (N, H)
    y_train: np.ndarray,  # (T,) — in-sample series used to compute MAE of naive
    seasonality: int = 52,  # seasonal period for naive baseline (52 = annual)
    weights: np.ndarray | None = None,
) -> float:
    """MASE with seasonal naive denominator.

    scale = mean(|y_train[t] - y_train[t - seasonality]|)
    MASE  = MAE(y_true, y_pred) / scale
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    y_train = np.asarray(y_train, dtype=np.float64)

    if len(y_train) <= seasonality:
        # Fall back to mean absolute value as scale
        scale = np.abs(y_train).mean() or 1.0
    else:
        scale = np.abs(y_train[seasonality:] - y_train[:-seasonality]).mean()
        scale = scale if scale > 0 else 1.0

    w = _weights(weights, y_true.shape)
    mae = float(np.dot(w, np.abs(y_true - y_pred)))
    return mae / scale


# ---------------------------------------------------------------------------
# Pinball loss
# ---------------------------------------------------------------------------


def pinball(
    y_true: np.ndarray,  # (N,) or (N, H)
    quantile_pred: np.ndarray,  # same shape as y_true
    q: float,  # quantile level in (0, 1)
    weights: np.ndarray | None = None,
) -> float:
    """Pinball (quantile) loss for a single quantile level q.

    L_q(y, f) = (1-q)(f - y)  if f >= y
                q (y - f)     if f < y
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    f = np.asarray(quantile_pred, dtype=np.float64).ravel()
    w = _weights(weights, y.shape)
    diff = y - f
    loss = np.where(diff >= 0, q * diff, (q - 1) * diff)
    return float(np.dot(w, loss))


def pinball_all_quantiles(
    y_true: np.ndarray,  # (N,) or (N, H)
    quantiles: np.ndarray,  # (N, Q) or (N, H, Q)
    q_levels: np.ndarray,  # (Q,)
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Pinball loss for every quantile level.  Returns shape (Q,)."""
    y = np.asarray(y_true, dtype=np.float64).ravel()
    Q = len(q_levels)
    losses = np.empty(Q)
    for qi in range(Q):
        if quantiles.ndim == 2:
            f = quantiles[:, qi].ravel()
        else:  # (N, H, Q) — flatten first two dims
            f = quantiles[:, :, qi].ravel()
            y = y_true.ravel()
        losses[qi] = pinball(y, f, float(q_levels[qi]), weights)
    return losses


# ---------------------------------------------------------------------------
# CRPS — Continuous Ranked Probability Score
# ---------------------------------------------------------------------------


def crps_from_quantiles(
    y_true: np.ndarray,  # (N,)
    quantiles: np.ndarray,  # (N, Q)
    q_levels: np.ndarray,  # (Q,) in [0,1]
    weights: np.ndarray | None = None,
) -> float:
    """Approximate CRPS via trapezoidal integration over pinball losses.

    CRPS = 2 * integral_0^1 pinball_q(y, F^{-1}(q)) dq

    Using the identity: CRPS = 2 * sum_q pinball_q * dq (trapezoidal rule).
    This is the standard quantile approximation from Gneiting & Raftery (2007).
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    Q_arr = np.asarray(quantiles, dtype=np.float64)  # (N, Q)
    q = np.asarray(q_levels, dtype=np.float64)  # (Q,)
    w = _weights(weights, y.shape)

    # Compute per-sample pinball losses for each q, then integrate
    total = 0.0
    dq = np.diff(q, prepend=0.0, append=1.0)  # trapz widths
    for qi in range(len(q)):
        f = Q_arr[:, qi]
        diff = y - f
        loss_i = np.where(diff >= 0, q[qi] * diff, (q[qi] - 1) * diff)
        total += float(np.dot(w, loss_i)) * dq[qi] * 2.0

    return total


def crps_per_horizon(
    y_true: np.ndarray,  # (N, H)
    quantiles: np.ndarray,  # (N, H, Q)
    q_levels: np.ndarray,  # (Q,)
    weights: np.ndarray | None = None,  # (N,)
) -> np.ndarray:
    """Per-horizon CRPS.  Returns shape (H,)."""
    y = np.asarray(y_true, dtype=np.float64)
    Q = np.asarray(quantiles, dtype=np.float64)
    N, H, _ = Q.shape
    w = _weights(weights, (N,))
    result = np.empty(H)
    for h in range(H):
        result[h] = crps_from_quantiles(y[:, h], Q[:, h, :], q_levels, w * N)
    return result


# ---------------------------------------------------------------------------
# WIS — Weighted Interval Score
# ---------------------------------------------------------------------------


def wis(
    y_true: np.ndarray,  # (N,)
    quantiles: np.ndarray,  # (N, Q)  — must include paired lower/upper + median
    q_levels: np.ndarray,  # (Q,)
    weights: np.ndarray | None = None,
) -> float:
    """Weighted Interval Score (Bracher et al. 2021).

    WIS = (1 / (K+0.5)) * [0.5 * |y - median| + sum_k alpha_k/2 * IS_k]

    where IS_k = (upper_k - lower_k) + 2/alpha_k * penalty
    and   penalty = max(0, lower_k - y) + max(0, y - upper_k)

    Assumes q_levels is symmetric around 0.5 and includes 0.5 itself.
    Pairs are matched as (q_levels[i], q_levels[-(i+1)]) for i < len//2.
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    Q = np.asarray(quantiles, dtype=np.float64)  # (N, Q)
    q = np.asarray(q_levels, dtype=np.float64)
    w = _weights(weights, y.shape)

    n_q = len(q)
    med_idx = n_q // 2  # index of the median

    total = 0.5 * np.dot(w, np.abs(y - Q[:, med_idx]))
    n_intervals = 0

    # Walk pairs from outside in
    i = 0
    j = n_q - 1
    while i < j:
        alpha = 2.0 * q[i]  # coverage level: 1 - alpha
        lower = Q[:, i]
        upper = Q[:, j]
        width = upper - lower
        penalty = np.maximum(0.0, lower - y) + np.maximum(0.0, y - upper)
        is_k = width + (2.0 / max(alpha, 1e-9)) * penalty
        total += (alpha / 2.0) * np.dot(w, is_k)
        n_intervals += 1
        i += 1
        j -= 1

    denom = n_intervals + 0.5
    return float(total / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def coverage(
    y_true: np.ndarray,  # (N,)
    lower: np.ndarray,  # (N,)
    upper: np.ndarray,  # (N,)
    weights: np.ndarray | None = None,
) -> float:
    """Empirical interval coverage: fraction of y_true inside [lower, upper]."""
    y = np.asarray(y_true, dtype=np.float64).ravel()
    lo = np.asarray(lower, dtype=np.float64).ravel()
    hi = np.asarray(upper, dtype=np.float64).ravel()
    w = _weights(weights, y.shape)
    inside = ((y >= lo) & (y <= hi)).astype(np.float64)
    return float(np.dot(w, inside))


def coverage_80(
    y_true: np.ndarray,
    quantiles: np.ndarray,  # (N, Q)
    q_levels: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """80% PI coverage: uses q=0.10 and q=0.90."""
    q = np.asarray(q_levels)
    lo_idx = int(np.argmin(np.abs(q - 0.10)))
    hi_idx = int(np.argmin(np.abs(q - 0.90)))
    return coverage(y_true, quantiles[:, lo_idx], quantiles[:, hi_idx], weights)


def coverage_90(
    y_true: np.ndarray,
    quantiles: np.ndarray,
    q_levels: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """90% PI coverage: uses q=0.05 and q=0.95."""
    q = np.asarray(q_levels)
    lo_idx = int(np.argmin(np.abs(q - 0.05)))
    hi_idx = int(np.argmin(np.abs(q - 0.95)))
    return coverage(y_true, quantiles[:, lo_idx], quantiles[:, hi_idx], weights)


# ---------------------------------------------------------------------------
# sMAPE
# ---------------------------------------------------------------------------


def smape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Symmetric MAPE.

    sMAPE = mean(2|y - f| / (|y| + |f| + eps))
    eps = 1e-8 to avoid division by zero when both are 0.
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    f = np.asarray(y_pred, dtype=np.float64).ravel()
    w = _weights(weights, y.shape)
    denom = np.abs(y) + np.abs(f) + 1e-8
    return float(np.dot(w, 2.0 * np.abs(y - f) / denom))


# ---------------------------------------------------------------------------
# Convenience: compute all scalar metrics at once
# ---------------------------------------------------------------------------


def evaluate(
    y_true: np.ndarray,  # (N,)
    y_pred: np.ndarray,  # (N,)  — P50
    quantiles: np.ndarray,  # (N, Q)
    q_levels: np.ndarray,  # (Q,)
    y_train: np.ndarray | None = None,  # (T,) for MASE; None → skip
    weights: np.ndarray | None = None,
    seasonality: int = 52,
) -> dict[str, float]:
    """Compute all scalar metrics and return as a dict.

    Keys: wape, crps, wis, coverage_80, coverage_90, smape,
          pinball_<q> for each quantile, mase (if y_train provided).
    """
    result: dict[str, float] = {
        "wape": wape(y_true, y_pred, weights),
        "crps": crps_from_quantiles(y_true, quantiles, q_levels, weights),
        "wis": wis(y_true, quantiles, q_levels, weights),
        "coverage_80": coverage_80(y_true, quantiles, q_levels, weights),
        "coverage_90": coverage_90(y_true, quantiles, q_levels, weights),
        "smape": smape(y_true, y_pred, weights),
    }
    if y_train is not None:
        result["mase"] = mase(y_true, y_pred, y_train, seasonality, weights)

    q = np.asarray(q_levels)
    pin = pinball_all_quantiles(y_true, quantiles, q_levels, weights)
    for i, qi in enumerate(q):
        result[f"pinball_{qi:.2f}"] = float(pin[i])

    return result


# ---------------------------------------------------------------------------
# Locked selection metric (S1.2 — do not change the protocol)
# ---------------------------------------------------------------------------


def revenue_weighted_wape(
    y_true: np.ndarray,  # (n_sku, H) — actual sales
    y_pred: np.ndarray,  # (n_sku, H) — point forecast (P50)
    price: np.ndarray,  # (n_sku,)   — fixed per-unit price from master (training-time)
) -> float:
    """Standard revenue-weighted WAPE — the locked selection metric.

    Formula: sum(price_i * sum_h|y_i,h - f_i,h|) / sum(price_i * sum_h y_i,h)

    Protocol (locked — do not deviate):
    - `price` = fixed per-SKU unit price from the product master at training time.
      NOT holdout actuals, NOT trailing-volume weights (both destabilise fold 2).
    - **Pool** all eligible (SKU, week) rows across selection folds 2-4 into one
      ratio. Do not average per-fold WAPEs.
    - Per-fold eligibility: include SKU only if first_sale <= fold_origin - 26w.
      Brand-new SKUs (no training history at origin) go to the cold-start route,
      not into selection WAPE.
    - CRPS and 80%-coverage are diagnostics and tiebreakers, never the primary metric.

    Returns 0.0 when y_true is all-zero (no denominator risk).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    price = np.asarray(price, dtype=np.float64).ravel()

    abs_err = np.abs(y_true - y_pred).sum(axis=1)  # (n_sku,)
    actual = y_true.sum(axis=1)  # (n_sku,)

    num = float((price * abs_err).sum())
    den = float((price * actual).sum())
    return num / den if den > 0 else 0.0
