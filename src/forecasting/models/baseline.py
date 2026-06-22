"""Baseline models: SeasonalNaive and ZeroForecast.

These are the universal fallback and comparison floor.  Phase 15 selection
rejects any model that cannot beat them.

SeasonalNaive
-------------
Forecast for horizon step h = sales[t - 52 + ((h-1) % 52)] (last-year-same-week).
If the series has < 52 weeks, falls back to the global mean of available history.
Registered for ALL segments (universal fallback).

ZeroForecast
------------
Returns zero for every quantile at every horizon.
Registered for: discontinued (dormant SKUs that should emit zero).
Also serves as the floor comparison for SKUs whose actuals are near zero.

Both use ForecastResult.from_quantiles() so they flow through the same
_finalize() path (floor + sort) as every other model.
"""

from __future__ import annotations

import logging

import numpy as np

from forecasting.models.base import ForecastModel, ForecastResult
from forecasting.registry import ALL_SEGMENTS, register_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ZeroForecast
# ---------------------------------------------------------------------------


@register_model("zero_forecast", segments=["discontinued"])
class ZeroForecast(ForecastModel):
    """Returns all-zero quantile forecasts.

    Used for discontinued/dormant SKUs and as the absolute floor comparison
    in Phase 15 selection.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> ZeroForecast:  # noqa: ARG002
        self._fitted = True
        return self

    def fit_series(self, series_dict: dict[str, np.ndarray]) -> ZeroForecast:  # noqa: ARG002
        self._sku_ids = list(series_dict.keys())
        self._fitted = True
        return self

    def predict(self, X: np.ndarray, horizon: int) -> ForecastResult:
        n_sku = (
            X.shape[0] if X.ndim >= 1 and X.shape[0] > 0 else len(getattr(self, "_sku_ids", [1]))
        )
        n_q = len(self.q_levels)
        zeros = np.zeros((n_sku, horizon, n_q), dtype=np.float64)
        return ForecastResult.from_quantiles(zeros, self.q_levels)


# ---------------------------------------------------------------------------
# SeasonalNaive
# ---------------------------------------------------------------------------


@register_model("seasonal_naive", segments=list(ALL_SEGMENTS))
class SeasonalNaive(ForecastModel):
    """Last-year-same-week point forecast with empirical quantile intervals.

    Point forecast:
        f[h] = y[t - 52 + ((h-1) % 52)]   if history >= 52 weeks
             = mean(y)                       otherwise (short-history fallback)

    Quantile intervals (conformal style):
        residuals = |y[t] - y[t-52]| over available history
        For q > 0.5: f[h] + quantile(residuals, q)
        For q < 0.5: max(0, f[h] - quantile(residuals, 1-q))
        For q = 0.5: f[h] (point forecast)

    Registered for ALL segments — this is the universal fallback that covers
    every SKU including those that other models cannot fit.
    """

    SEASON: int = 52  # weekly annual seasonality

    def __init__(self, q_levels: np.ndarray | None = None) -> None:
        super().__init__(q_levels)
        self._sku_params: dict[str, dict] = {}
        self._skipped_skus: set[str] = set()
        self._sku_series: dict[str, np.ndarray] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> SeasonalNaive:  # noqa: ARG002
        self._fitted = True
        return self

    def fit_series(self, series_dict: dict[str, np.ndarray]) -> SeasonalNaive:
        """Fit seasonal naive on a dict of {sku_id: y_array}."""
        self._sku_series = {k: v.copy() for k, v in series_dict.items()}
        self._sku_params = {}
        self._skipped_skus = set()

        for uid, y in series_dict.items():
            if len(y) == 0:
                self._skipped_skus.add(uid)
                continue

            T = len(y)
            if T >= self.SEASON:
                # Use last full seasonal cycle as the point forecast template
                seasonal_template = y[-self.SEASON :]
                # Residuals: |y[t] - y[t-52]| over overlapping window
                residuals = np.abs(y[self.SEASON :] - y[: -self.SEASON])
            else:
                # Short history: use global mean as template
                seasonal_template = np.full(self.SEASON, y.mean())
                # Residuals: |y[t] - mean| as a rough spread estimate
                residuals = np.abs(y - y.mean())

            self._sku_params[uid] = {
                "template": seasonal_template,
                "residuals": residuals,
                "n_train": T,
            }

        self._fitted = True
        logger.info(
            "SeasonalNaive: fit %d SKUs (%d skipped)",
            len(self._sku_params),
            len(self._skipped_skus),
        )
        return self

    def predict(self, X: np.ndarray, horizon: int) -> ForecastResult:  # noqa: ARG002
        """Generate seasonal naive forecasts with conformal quantile intervals."""
        if not self._fitted:
            raise RuntimeError("Call fit_series() before predict().")

        uid_order = sorted(self._sku_series.keys())
        n_sku = len(uid_order)
        n_q = len(self.q_levels)
        q_cube = np.zeros((n_sku, horizon, n_q))

        for i, uid in enumerate(uid_order):
            if uid in self._skipped_skus:
                continue

            params = self._sku_params[uid]
            template = params["template"]  # (52,)
            residuals = params["residuals"]  # (T-52,) or (T,)

            # Point forecast: cycle through the seasonal template
            point = np.array([template[(h - 1) % self.SEASON] for h in range(1, horizon + 1)])
            point = np.maximum(0.0, point)

            # Conformal quantile intervals from residuals
            if len(residuals) == 0:
                # No residuals → use point forecast for all quantiles
                q_cube[i] = point[:, None] * np.ones((1, n_q))
                continue

            for qi, q in enumerate(self.q_levels):
                if abs(q - 0.5) < 1e-9:
                    q_cube[i, :, qi] = point
                elif q > 0.5:
                    thresh = float(np.quantile(residuals, q))
                    q_cube[i, :, qi] = point + thresh
                else:
                    thresh = float(np.quantile(residuals, 1.0 - q))
                    q_cube[i, :, qi] = np.maximum(0.0, point - thresh)

        return ForecastResult.from_quantiles(q_cube, self.q_levels)


# ---------------------------------------------------------------------------
# TrendSeasonalModel — for erratic segment (S1.3)
# ---------------------------------------------------------------------------


@register_model("trend_seasonal", segments=["erratic", "smooth"])
class TrendSeasonalModel(ForecastModel):
    """Seasonal naive × clipped YoY growth multiplier.

    For erratic SKUs where demand is growing or shrinking.  The multiplier
    is clipped to [0.5, 3.0] so it cannot blow up in volatile periods
    (the key difference from Theta, which over-shoots in fold-4 conditions).

    Point forecast:
        g = clip(mean(y[-13:]) / mean(y[-65:-52]), 0.5, 3.0)  -- YoY growth
        f[h] = max(0, seasonal_base[h] * g)

    Falls back to SeasonalNaive when < 65 weeks of history (can't compute YoY).
    Intervals: same split-conformal approach as SeasonalNaive.
    """

    SEASON: int = 52
    GROWTH_CLIP: tuple[float, float] = (0.5, 3.0)
    MIN_YOY_WEEKS: int = 65  # need at least this for YoY window

    def __init__(self, q_levels: np.ndarray | None = None) -> None:
        super().__init__(q_levels)
        self._sku_params: dict[str, dict] = {}
        self._skipped_skus: set[str] = set()
        self._sku_series: dict[str, np.ndarray] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> TrendSeasonalModel:  # noqa: ARG002
        self._fitted = True
        return self

    def fit_series(self, series_dict: dict[str, np.ndarray]) -> TrendSeasonalModel:
        self._sku_series = {k: v.copy() for k, v in series_dict.items()}
        self._sku_params = {}
        self._skipped_skus = set()

        for uid, y in series_dict.items():
            if len(y) == 0:
                self._skipped_skus.add(uid)
                continue

            T = len(y)

            # S1.4 Stockout gate: if last 8 weeks are all zero, the SKU is dead
            # at the forecast origin.  Damp to a near-zero constant forecast.
            dead_at_origin = T >= 8 and float(y[-8:].sum()) == 0.0
            if dead_at_origin:
                hist_mean = float(y[-52:-8].mean()) if T >= 60 else float(y.mean())
                near_zero = max(0.0, hist_mean * 0.05)
                self._sku_params[uid] = {
                    "template": np.full(self.SEASON, near_zero),
                    "growth": 1.0,
                    "residuals": np.array([near_zero]),
                    "dead": True,
                }
                continue

            # Compute YoY growth multiplier
            if T >= self.MIN_YOY_WEEKS:
                recent = y[-13:].mean()
                year_ago = y[-65:-52].mean()
                if year_ago > 1e-6:
                    growth = float(np.clip(recent / year_ago, *self.GROWTH_CLIP))
                else:
                    growth = 1.0
            else:
                growth = 1.0  # not enough history → no trend adjustment

            # Seasonal template (last full 52-week cycle)
            if T >= self.SEASON:
                template = y[-self.SEASON :]
                residuals = np.abs(y[self.SEASON :] - y[: -self.SEASON])
            else:
                template = np.full(self.SEASON, y.mean())
                residuals = np.abs(y - y.mean())

            self._sku_params[uid] = {
                "template": template,
                "growth": growth,
                "residuals": residuals,
                "dead": False,
            }

        self._fitted = True
        return self

    def predict(self, X: np.ndarray, horizon: int) -> ForecastResult:  # noqa: ARG002
        if not self._fitted:
            raise RuntimeError("Call fit_series() before predict().")

        uid_order = sorted(self._sku_series.keys())
        n_sku = len(uid_order)
        n_q = len(self.q_levels)
        q_cube = np.zeros((n_sku, horizon, n_q))

        for i, uid in enumerate(uid_order):
            if uid in self._skipped_skus:
                continue

            params = self._sku_params[uid]
            template = params["template"]
            growth = params["growth"]
            residuals = params["residuals"]

            base = np.array([template[(h - 1) % self.SEASON] for h in range(1, horizon + 1)])
            point = np.maximum(0.0, base * growth)

            if len(residuals) == 0:
                q_cube[i] = point[:, None] * np.ones((1, n_q))
                continue

            for qi, q in enumerate(self.q_levels):
                if abs(q - 0.5) < 1e-9:
                    q_cube[i, :, qi] = point
                elif q > 0.5:
                    thresh = float(np.quantile(residuals, q))
                    q_cube[i, :, qi] = point + thresh
                else:
                    thresh = float(np.quantile(residuals, 1.0 - q))
                    q_cube[i, :, qi] = np.maximum(0.0, point - thresh)

        return ForecastResult.from_quantiles(q_cube, self.q_levels)


# ---------------------------------------------------------------------------
# RecentLevelModel — for smooth segment (S1.3)
# ---------------------------------------------------------------------------


@register_model("recent_level", segments=["smooth"])
class RecentLevelModel(ForecastModel):
    """8-week mean as a constant-level forecast.

    Appropriate for smooth SKUs because:
    - If the SKU is actively selling: the recent mean is the best stable estimate.
    - If the SKU has gone silent (stockout / discontinued): the 8-week mean ≈ 0,
      which is the correct near-zero forecast without needing a separate gate.

    Intervals: conformal from the difference between recent 8-week values and
    their mean (captures the local spread without needing seasonal residuals).
    """

    LEVEL_WINDOW: int = 8

    def __init__(self, q_levels: np.ndarray | None = None) -> None:
        super().__init__(q_levels)
        self._sku_params: dict[str, dict] = {}
        self._skipped_skus: set[str] = set()
        self._sku_series: dict[str, np.ndarray] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> RecentLevelModel:  # noqa: ARG002
        self._fitted = True
        return self

    def fit_series(self, series_dict: dict[str, np.ndarray]) -> RecentLevelModel:
        self._sku_series = {k: v.copy() for k, v in series_dict.items()}
        self._sku_params = {}
        self._skipped_skus = set()

        for uid, y in series_dict.items():
            if len(y) == 0:
                self._skipped_skus.add(uid)
                continue

            window = y[-self.LEVEL_WINDOW :] if len(y) >= self.LEVEL_WINDOW else y
            level = float(window.mean())
            residuals = np.abs(window - level)

            self._sku_params[uid] = {"level": level, "residuals": residuals}

        self._fitted = True
        return self

    def predict(self, X: np.ndarray, horizon: int) -> ForecastResult:  # noqa: ARG002
        if not self._fitted:
            raise RuntimeError("Call fit_series() before predict().")

        uid_order = sorted(self._sku_series.keys())
        n_sku = len(uid_order)
        n_q = len(self.q_levels)
        q_cube = np.zeros((n_sku, horizon, n_q))

        for i, uid in enumerate(uid_order):
            if uid in self._skipped_skus:
                continue

            params = self._sku_params[uid]
            level = params["level"]
            residuals = params["residuals"]
            point = np.full(horizon, max(0.0, level))

            if len(residuals) == 0:
                q_cube[i] = point[:, None] * np.ones((1, n_q))
                continue

            for qi, q in enumerate(self.q_levels):
                if abs(q - 0.5) < 1e-9:
                    q_cube[i, :, qi] = point
                elif q > 0.5:
                    thresh = float(np.quantile(residuals, q))
                    q_cube[i, :, qi] = point + thresh
                else:
                    thresh = float(np.quantile(residuals, 1.0 - q))
                    q_cube[i, :, qi] = np.maximum(0.0, point - thresh)

        return ForecastResult.from_quantiles(q_cube, self.q_levels)
