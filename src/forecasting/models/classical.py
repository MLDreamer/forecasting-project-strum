"""Classical univariate models with split-conformal prediction intervals.

Models
------
AutoETS    — statsforecast AutoETS (auto error/trend/seasonality selection)
AutoARIMA  — statsforecast AutoARIMA
DotTheta   — statsforecast DynamicOptimizedTheta

Split-conformal quantiles (design)
-----------------------------------
1. Reserve the last `cal_fraction` of each SKU's training series as a
   calibration split.
2. Fit the model on the pre-calibration portion.
3. Forecast over the calibration window; compute absolute residuals.
4. For each quantile q, threshold_q = quantile(|residuals|, q).
5. Final quantile forecast = point_forecast ± threshold_q
   (upper = point + threshold, lower = max(0, point - threshold)).
6. Re-fit on the full training series before predicting the test horizon.

Skip rule: SKUs with fewer than `MIN_HISTORY_WEEKS` non-zero observations
are skipped (point forecast = 0, conformal intervals = 0).

Lazy imports: statsforecast is imported inside methods so registering the
module does not require statsforecast to be installed for unrelated tests.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from forecasting.models.base import ForecastModel, ForecastResult
from forecasting.registry import register_model

logger = logging.getLogger(__name__)

MIN_HISTORY_WEEKS: int = 26  # skip SKUs with fewer observations
CAL_FRACTION: float = 0.25  # fraction of training window used for calibration
SEASON_LENGTH: int = 52  # weekly annual seasonality


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_sf_df(
    series_dict: dict[str, np.ndarray],
    freq: str = "W-SAT",
) -> pd.DataFrame:
    """Build a StatsForecast-compatible long DataFrame from a dict of series."""
    rows = []
    for uid, y in series_dict.items():
        T = len(y)
        ds = pd.date_range(start="2020-01-04", periods=T, freq=freq)
        for d, v in zip(ds, y, strict=False):
            rows.append({"unique_id": uid, "ds": d, "y": float(v)})
    return pd.DataFrame(rows)


def _conformal_quantiles(
    residuals: np.ndarray,  # 1-D absolute residuals from calibration
    point_forecast: np.ndarray,  # shape (H,) point forecast on test horizon
    q_levels: np.ndarray,  # shape (n_q,)
) -> np.ndarray:
    """Compute split-conformal quantile forecasts for one SKU.

    Returns shape (H, n_q): for each horizon step and each quantile,
    PI = point ± threshold_q.  Upper quantiles > 0.5 get +threshold,
    lower quantiles < 0.5 get point - threshold (floored at 0).
    Median (q=0.5) = point forecast.
    """
    H = len(point_forecast)
    n_q = len(q_levels)
    result = np.empty((H, n_q))

    if len(residuals) == 0:
        # No calibration data — return point forecast for all quantiles
        result[:, :] = point_forecast[:, None]
        return result

    for qi, q in enumerate(q_levels):
        if abs(q - 0.5) < 1e-9:
            result[:, qi] = point_forecast
        elif q > 0.5:
            # Upper tail: point + threshold at coverage q
            thresh = float(np.quantile(residuals, q))
            result[:, qi] = point_forecast + thresh
        else:
            # Lower tail: point - threshold at coverage (1-q)
            thresh = float(np.quantile(residuals, 1.0 - q))
            result[:, qi] = np.maximum(0.0, point_forecast - thresh)

    return result


# ---------------------------------------------------------------------------
# Base class for all statsforecast-backed models
# ---------------------------------------------------------------------------


class _StatsforecastModel(ForecastModel):
    """Base for AutoETS / AutoARIMA / DotTheta — handles batch fitting and
    split-conformal interval construction."""

    min_history_weeks: int = MIN_HISTORY_WEEKS

    def __init__(
        self,
        q_levels: np.ndarray | None = None,
        cal_fraction: float = CAL_FRACTION,
        season_length: int = SEASON_LENGTH,
        n_jobs: int = 1,
    ) -> None:
        super().__init__(q_levels)
        self.cal_fraction = cal_fraction
        self.season_length = season_length
        self.n_jobs = n_jobs

        # Populated by fit()
        self._sku_series: dict[str, np.ndarray] = {}
        self._conformal_thresholds: dict[str, np.ndarray] = {}
        self._skipped_skus: set[str] = set()

    def _make_model(self):  # type: ignore[return]
        """Return the statsforecast model instance. Override in subclasses."""
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray) -> _StatsforecastModel:  # noqa: ARG002
        """Fit is not used directly; call fit_series() instead."""
        self._fitted = True
        return self

    def fit_series(self, series_dict: dict[str, np.ndarray]) -> _StatsforecastModel:
        """Fit on a dict of {sku_id: y_array} and build conformal thresholds.

        Parameters
        ----------
        series_dict : {sku_id: np.ndarray}
            Training series for each SKU.  Arrays are floats, non-negative.
        """
        from statsforecast import StatsForecast  # lazy import

        self._sku_series = {k: v.copy() for k, v in series_dict.items()}
        self._conformal_thresholds = {}
        self._skipped_skus = set()

        # --- Identify skippable SKUs ---
        cal_series: dict[str, np.ndarray] = {}  # pre-cal portion
        full_series: dict[str, np.ndarray] = {}  # full series (for re-fit)

        for uid, y in series_dict.items():
            if len(y) < self.min_history_weeks:
                self._skipped_skus.add(uid)
                logger.debug(
                    "Skipping %s: only %d weeks (< %d)", uid, len(y), self.min_history_weeks
                )
                continue
            n_cal = max(1, int(len(y) * self.cal_fraction))
            cal_series[uid] = y[:-n_cal]
            full_series[uid] = y

        if not cal_series:
            logger.warning("All SKUs skipped in %s.fit_series()", self.__class__.__name__)
            self._fitted = True
            return self

        # --- Fit on pre-calibration, predict calibration window ---
        df_cal = _make_sf_df(cal_series)
        sf_cal = StatsForecast(
            models=[self._make_model()],
            freq="W-SAT",
            n_jobs=self.n_jobs,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sf_cal.fit(df_cal)

        # Collect conformal residuals per SKU
        for uid, y in series_dict.items():
            if uid in self._skipped_skus:
                continue
            n_cal = max(1, int(len(y) * self.cal_fraction))
            y_cal_true = y[-n_cal:]
            try:
                pred_cal = sf_cal.predict(h=n_cal)
                model_col = [c for c in pred_cal.columns if c not in ("unique_id", "ds")][0]
                y_hat_cal = pred_cal[pred_cal["unique_id"] == uid][model_col].values
                residuals = np.abs(y_cal_true - y_hat_cal[: len(y_cal_true)])
            except Exception as exc:
                logger.warning("Calibration failed for %s: %s — using zero residuals", uid, exc)
                residuals = np.zeros(1)
            self._conformal_thresholds[uid] = residuals

        # --- Re-fit on full data ---
        df_full = _make_sf_df(full_series)
        self._sf_full = StatsForecast(
            models=[self._make_model()],
            freq="W-SAT",
            n_jobs=self.n_jobs,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._sf_full.fit(df_full)

        self._model_col: str = [
            c for c in self._sf_full.predict(h=1).columns if c not in ("unique_id", "ds")
        ][0]

        self._fitted = True
        logger.info(
            "%s: fit %d SKUs (%d skipped)",
            self.__class__.__name__,
            len(full_series),
            len(self._skipped_skus),
        )
        return self

    def predict(self, X: np.ndarray, horizon: int) -> ForecastResult:
        """Generate conformal quantile forecasts.

        Parameters
        ----------
        X : np.ndarray
            Not used for univariate models (API compatibility).
        horizon : int
            Forecast horizon in weeks.

        Returns
        -------
        ForecastResult, shape (n_sku, horizon, n_q).
        """
        if not self._fitted:
            raise RuntimeError("Call fit_series() before predict().")

        uid_order = sorted(self._sku_series.keys())
        n_sku = len(uid_order)
        n_q = len(self.q_levels)
        q_cube = np.zeros((n_sku, horizon, n_q))

        # Predict point forecasts for non-skipped SKUs
        non_skipped = [u for u in uid_order if u not in self._skipped_skus]
        if non_skipped:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                preds = self._sf_full.predict(h=horizon)
            preds = preds[preds["unique_id"].isin(non_skipped)]

        for i, uid in enumerate(uid_order):
            if uid in self._skipped_skus:
                # Zero forecast for short-history SKUs
                q_cube[i, :, :] = 0.0
                continue

            point = preds[preds["unique_id"] == uid][self._model_col].values
            if len(point) < horizon:
                # Pad with last value if prediction is shorter than horizon
                pad = np.full(horizon - len(point), point[-1] if len(point) > 0 else 0.0)
                point = np.concatenate([point, pad])
            point = np.maximum(0.0, point[:horizon])

            residuals = self._conformal_thresholds.get(uid, np.zeros(1))
            q_cube[i] = _conformal_quantiles(residuals, point, self.q_levels)

        return ForecastResult.from_quantiles(q_cube, self.q_levels)

    def predict_series(
        self,
        horizon: int,
        sku_ids: list[str] | None = None,
    ) -> ForecastResult:
        """Convenience: predict with optional explicit SKU ordering."""
        if sku_ids is not None:
            # Temporarily reorder internal sku list
            orig = self._sku_series
            self._sku_series = {k: orig[k] for k in sku_ids if k in orig}
        result = self.predict(np.empty(0), horizon)
        if sku_ids is not None:
            self._sku_series = orig  # type: ignore[possibly-undefined]
        return result


# ---------------------------------------------------------------------------
# Concrete model classes
# ---------------------------------------------------------------------------


@register_model(
    "auto_ets",
    segments=["smooth", "erratic"],
)
class AutoETSModel(_StatsforecastModel):
    """AutoETS via statsforecast with split-conformal intervals."""

    def _make_model(self):  # type: ignore[return]
        from statsforecast.models import AutoETS  # lazy import

        return AutoETS(season_length=self.season_length)


@register_model(
    "auto_arima",
    segments=["smooth", "erratic"],
)
class AutoARIMAModel(_StatsforecastModel):
    """AutoARIMA via statsforecast with split-conformal intervals."""

    def _make_model(self):  # type: ignore[return]
        from statsforecast.models import AutoARIMA  # lazy import

        return AutoARIMA(season_length=self.season_length)


@register_model(
    "theta",
    segments=["smooth", "erratic", "intermittent"],
)
class ThetaModel(_StatsforecastModel):
    """Dynamic Optimized Theta via statsforecast with split-conformal intervals."""

    def _make_model(self):  # type: ignore[return]
        from statsforecast.models import DynamicOptimizedTheta  # lazy import

        return DynamicOptimizedTheta(season_length=self.season_length)
