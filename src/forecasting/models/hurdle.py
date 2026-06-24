"""Hurdle model for intermittent and promo-driven demand.

Two-part model:
  Part 1 — occurrence: P(demand > 0) from logistic regression on recent zero-rate
  Part 2 — size: E[demand | demand > 0] from gamma/empirical on non-zero values

Point forecast = P(demand > 0) * E[demand | demand > 0]
Quantiles via Monte-Carlo: for each horizon step, draw Bernoulli(p) then
if nonzero draw from Gamma(shape, scale) fitted on nonzero history.

Registered for: intermittent, lumpy, promo_driven.
"""

from __future__ import annotations

import logging
import numpy as np

from forecasting.models.base import ForecastModel, ForecastResult
from forecasting.registry import register_model

logger = logging.getLogger(__name__)

N_SAMPLES = 500
MIN_NZ = 4


@register_model("hurdle", segments=["intermittent", "lumpy", "promo_driven", "erratic"])
class HurdleModel(ForecastModel):
    """Two-part hurdle: Bernoulli occurrence × Gamma size.

    Occurrence probability is estimated from the recent 13-week zero rate.
    Size distribution is fitted via Method of Moments on non-zero values.
    Seasonal pattern is captured by re-weighting occurrence by week-of-year
    relative occurrence rate (if ≥ 52 weeks of history).
    """

    LEVEL_WINDOW = 13
    SEASON = 52

    def __init__(self, q_levels: np.ndarray | None = None) -> None:
        super().__init__(q_levels)
        self._sku_params: dict[str, dict] = {}
        self._skipped: set[str] = set()
        self._sku_series: dict[str, np.ndarray] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HurdleModel":  # noqa: ARG002
        self._fitted = True
        return self

    def fit_series(self, series_dict: dict[str, np.ndarray]) -> "HurdleModel":
        self._sku_series = {k: v.copy() for k, v in series_dict.items()}
        self._sku_params = {}
        self._skipped = set()

        for uid, y in series_dict.items():
            if len(y) == 0:
                self._skipped.add(uid)
                continue

            T = len(y)
            nz = y[y > 0]

            # Occurrence probability from recent window
            window = y[-self.LEVEL_WINDOW:] if T >= self.LEVEL_WINDOW else y
            p_occur = float((window > 0).mean())
            p_occur = max(0.01, min(0.99, p_occur))

            # Size distribution (Gamma MoM on non-zero values)
            if len(nz) >= MIN_NZ:
                mu_nz = float(nz.mean())
                var_nz = float(nz.var(ddof=1)) if len(nz) > 1 else mu_nz
                if var_nz <= 0 or mu_nz <= 0:
                    shape, scale = 1.0, mu_nz
                else:
                    shape = mu_nz ** 2 / var_nz
                    scale = var_nz / mu_nz
            else:
                shape = 1.0
                scale = float(y.mean()) if y.mean() > 0 else 1.0

            # Seasonal occurrence weights (week-of-year based, if ≥ 52w)
            seasonal_p = None
            if T >= self.SEASON:
                # Compute occurrence rate per week-of-year position
                woy_occur = np.zeros(self.SEASON)
                woy_count = np.zeros(self.SEASON)
                for t in range(T):
                    w = t % self.SEASON
                    woy_occur[w] += float(y[t] > 0)
                    woy_count[w] += 1.0
                with np.errstate(invalid="ignore", divide="ignore"):
                    rates = np.where(woy_count > 0, woy_occur / woy_count, p_occur)
                seasonal_p = np.clip(rates, 0.01, 0.99)

            self._sku_params[uid] = {
                "p_occur": p_occur,
                "shape": shape,
                "scale": scale,
                "seasonal_p": seasonal_p,
                "last_t": T,
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

        rng = np.random.default_rng(42)

        for i, uid in enumerate(uid_order):
            if uid in self._skipped:
                continue

            p = self._sku_params[uid]
            p_occur = p["p_occur"]
            shape = p["shape"]
            scale = p["scale"]
            seasonal_p = p["seasonal_p"]
            last_t = p["last_t"]

            for h in range(horizon):
                # Occurrence probability (seasonal if available)
                if seasonal_p is not None:
                    step_p = float(seasonal_p[(last_t + h) % self.SEASON])
                else:
                    step_p = p_occur

                # Monte-Carlo: N_SAMPLES draws
                occur = rng.binomial(1, step_p, N_SAMPLES).astype(float)
                sizes = rng.gamma(shape, scale, N_SAMPLES)
                samples = occur * sizes
                samples = np.maximum(0.0, samples)

                for qi, q in enumerate(self.q_levels):
                    q_cube[i, h, qi] = float(np.quantile(samples, q))

        return ForecastResult.from_quantiles(q_cube, self.q_levels)
