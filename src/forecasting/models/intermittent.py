"""Intermittent and lumpy demand models.

Models
------
CrostonSBAModel   — Croston-SBA via statsforecast + ConformalIntervals
TSBModel          — Teunter-Syntetos-Babai (TSB) via statsforecast + ConformalIntervals
CompoundBernoulli — Compound Bernoulli-Gamma bootstrap (intermittent + lumpy)

Design
------
- Croston-SBA and TSB use statsforecast's built-in ConformalIntervals for
  prediction intervals — avoids rolling a separate calibration split for
  methods that already have a conformal wrapper.
- CompoundBernoulli fits a Bernoulli(p) × Gamma(shape, scale) model
  using MoM (Method of Moments) on the training series, then draws B=500
  Monte-Carlo paths and feeds them through ForecastResult.from_samples().
  This is appropriate for lumpy demand where both the occurrence probability
  AND the demand size vary considerably.
- All models skip SKUs with fewer than MIN_HISTORY_WEEKS observations
  (zero forecast returned).
- Lazy imports: statsforecast imported inside methods.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from forecasting.models.base import ForecastModel, ForecastResult
from forecasting.registry import register_model

logger = logging.getLogger(__name__)

MIN_HISTORY_WEEKS: int = 4  # intermittent models can work with less history
MIN_NONZERO: int = 3  # minimum non-zero demands for MoM fit
N_SAMPLES: int = 500  # bootstrap paths for CompoundBernoulli
SEASON_LENGTH: int = 52


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_sf_df(series_dict: dict[str, np.ndarray], freq: str = "W-SAT") -> pd.DataFrame:
    rows = []
    for uid, y in series_dict.items():
        ds = pd.date_range(start="2020-01-04", periods=len(y), freq=freq)
        for d, v in zip(ds, y, strict=False):
            rows.append({"unique_id": uid, "ds": d, "y": float(v)})
    return pd.DataFrame(rows)


def _extract_quantile_cols(
    pred_df: pd.DataFrame,
    uid: str,
    model_prefix: str,
    horizon: int,
    q_levels: np.ndarray,
) -> np.ndarray:
    """Extract quantile cube (horizon, n_q) from a statsforecast conformal prediction DF."""
    row = pred_df[pred_df["unique_id"] == uid].head(horizon)
    n_q = len(q_levels)
    H = min(len(row), horizon)
    q_cube = np.zeros((horizon, n_q))

    for qi, q in enumerate(q_levels):
        level = int(round(q * 100)) if q != 0.5 else None
        if level is None or level == 50:
            # Median = point forecast
            if model_prefix in row.columns:
                q_cube[:H, qi] = row[model_prefix].values[:H]
        elif q > 0.5:
            col = f"{model_prefix}-hi-{level}"
            if col in row.columns:
                q_cube[:H, qi] = row[col].values[:H]
            else:
                q_cube[:H, qi] = (
                    row[model_prefix].values[:H] if model_prefix in row.columns else 0.0
                )
        else:
            col = f"{model_prefix}-lo-{level}"
            if col in row.columns:
                q_cube[:H, qi] = row[col].values[:H]
            else:
                q_cube[:H, qi] = (
                    row[model_prefix].values[:H] if model_prefix in row.columns else 0.0
                )

    return q_cube


# ---------------------------------------------------------------------------
# Base class for statsforecast-backed intermittent models
# ---------------------------------------------------------------------------


class _IntermittentSFModel(ForecastModel):
    """Base for CrostonSBA and TSB — thin wrapper around statsforecast."""

    min_history_weeks: int = MIN_HISTORY_WEEKS

    def __init__(
        self,
        q_levels: np.ndarray | None = None,
        n_conformal_windows: int = 2,
        n_jobs: int = 1,
    ) -> None:
        super().__init__(q_levels)
        self.n_conformal_windows = n_conformal_windows
        self.n_jobs = n_jobs
        self._sku_series: dict[str, np.ndarray] = {}
        self._skipped_skus: set[str] = set()

    def _make_sf_model(self, ci):  # type: ignore[return]
        raise NotImplementedError

    def _model_col_prefix(self) -> str:
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray) -> _IntermittentSFModel:  # noqa: ARG002
        self._fitted = True
        return self

    def fit_series(self, series_dict: dict[str, np.ndarray]) -> _IntermittentSFModel:
        """Fit using split-conformal intervals (same approach as classical models).

        Fits on first (1 - cal_fraction) of each series, calibrates on the rest,
        then re-fits on the full series.
        """
        from statsforecast import StatsForecast  # lazy

        self._sku_series = {k: v.copy() for k, v in series_dict.items()}
        self._skipped_skus = set()
        self._conformal_thresholds: dict[str, np.ndarray] = {}

        cal_series: dict[str, np.ndarray] = {}
        full_series: dict[str, np.ndarray] = {}
        cal_fraction = 0.25

        for uid, y in series_dict.items():
            if len(y) < self.min_history_weeks:
                self._skipped_skus.add(uid)
                continue
            n_cal = max(1, int(len(y) * cal_fraction))
            cal_series[uid] = y[:-n_cal]
            full_series[uid] = y

        if not full_series:
            self._fitted = True
            return self

        # Fit on pre-cal portion, predict cal window for residuals
        df_cal = _make_sf_df(cal_series)
        sf_cal = StatsForecast(
            models=[self._make_sf_model(ci=None)],
            freq="W-SAT",
            n_jobs=self.n_jobs,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sf_cal.fit(df_cal)

        prefix = self._model_col_prefix()
        for uid, y in series_dict.items():
            if uid in self._skipped_skus:
                continue
            n_cal = max(1, int(len(y) * cal_fraction))
            y_cal_true = y[-n_cal:]
            try:
                pred_cal = sf_cal.predict(h=n_cal)
                y_hat = pred_cal[pred_cal["unique_id"] == uid][prefix].values
                residuals = np.abs(y_cal_true - y_hat[: len(y_cal_true)])
            except Exception as exc:
                logger.warning("Calibration failed for %s: %s", uid, exc)
                residuals = np.zeros(1)
            self._conformal_thresholds[uid] = residuals

        # Re-fit on full data
        df_full = _make_sf_df(full_series)
        self._sf = StatsForecast(
            models=[self._make_sf_model(ci=None)],
            freq="W-SAT",
            n_jobs=self.n_jobs,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._sf.fit(df_full)

        self._fitted = True
        logger.info(
            "%s: fit %d SKUs (%d skipped)",
            self.__class__.__name__,
            len(full_series),
            len(self._skipped_skus),
        )
        return self

    def predict(self, X: np.ndarray, horizon: int) -> ForecastResult:  # noqa: ARG002
        if not self._fitted:
            raise RuntimeError("Call fit_series() before predict().")

        uid_order = sorted(self._sku_series.keys())
        n_sku = len(uid_order)
        n_q = len(self.q_levels)
        q_cube = np.zeros((n_sku, horizon, n_q))
        prefix = self._model_col_prefix()

        non_skipped = [u for u in uid_order if u not in self._skipped_skus]
        if non_skipped:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                preds = self._sf.predict(h=horizon)

            for i, uid in enumerate(uid_order):
                if uid in self._skipped_skus:
                    continue
                point_row = preds[preds["unique_id"] == uid][prefix].values
                point = np.maximum(0.0, point_row[:horizon])
                residuals = self._conformal_thresholds.get(uid, np.zeros(1))

                # Use the same conformal helper as classical models
                from forecasting.models.classical import _conformal_quantiles  # lazy

                q_cube[i] = _conformal_quantiles(residuals, point, self.q_levels)

        return ForecastResult.from_quantiles(q_cube, self.q_levels)


# ---------------------------------------------------------------------------
# Concrete statsforecast models
# ---------------------------------------------------------------------------


@register_model("croston_sba", segments=["intermittent", "lumpy", "promo_driven"])
class CrostonSBAModel(_IntermittentSFModel):
    """Croston-SBA with split-conformal prediction intervals."""

    def _make_sf_model(self, ci=None):  # type: ignore[return]
        from statsforecast.models import CrostonSBA  # lazy

        return CrostonSBA()

    def _model_col_prefix(self) -> str:
        return "CrostonSBA"


@register_model("tsb", segments=["intermittent", "lumpy"])
class TSBModel(_IntermittentSFModel):
    """Teunter-Syntetos-Babai (TSB) with split-conformal prediction intervals."""

    def __init__(
        self,
        q_levels: np.ndarray | None = None,
        alpha_d: float = 0.1,
        alpha_p: float = 0.1,
        n_conformal_windows: int = 2,
        n_jobs: int = 1,
    ) -> None:
        super().__init__(q_levels, n_conformal_windows, n_jobs)
        self.alpha_d = alpha_d
        self.alpha_p = alpha_p

    def _make_sf_model(self, ci=None):  # type: ignore[return]
        from statsforecast.models import TSB  # lazy

        return TSB(alpha_d=self.alpha_d, alpha_p=self.alpha_p)

    def _model_col_prefix(self) -> str:
        return "TSB"


# ---------------------------------------------------------------------------
# Compound Bernoulli-Gamma bootstrap
# ---------------------------------------------------------------------------


def _fit_compound_bernoulli(
    y: np.ndarray,
) -> tuple[float, float, float]:
    """Fit p, Gamma shape and scale via MoM.

    Returns (p, shape, scale) where:
        p     = P(demand > 0)
        shape = alpha (Gamma shape)
        scale = beta  (Gamma scale)
    """
    p = float((y > 0).mean())
    nz = y[y > 0]

    if len(nz) < 2:
        # Degenerate: use empirical mean as point mass
        return p, 1.0, float(nz.mean()) if len(nz) > 0 else 1.0

    mu = float(nz.mean())
    var = float(nz.var(ddof=1))

    if var <= 0 or mu <= 0:
        return p, 1.0, mu

    shape = mu**2 / var
    scale = var / mu
    return p, shape, scale


@register_model("compound_bernoulli", segments=["intermittent", "lumpy", "promo_driven"])
class CompoundBernoulliModel(ForecastModel):
    """Compound Bernoulli-Gamma bootstrap for intermittent and lumpy demand.

    Fits Bernoulli(p) × Gamma(shape, scale) independently per SKU
    using Method of Moments on the training series, then generates
    N_SAMPLES Monte-Carlo paths and returns them via ForecastResult.from_samples().

    This model is appropriate when BOTH the occurrence probability AND the
    demand size distribution need to be captured (lumpy demand profile).
    """

    min_history_weeks: int = MIN_HISTORY_WEEKS

    def __init__(
        self,
        q_levels: np.ndarray | None = None,
        n_samples: int = N_SAMPLES,
        random_seed: int = 42,
    ) -> None:
        super().__init__(q_levels)
        self.n_samples = n_samples
        self.random_seed = random_seed
        self._params: dict[str, tuple[float, float, float]] = {}
        self._skipped_skus: set[str] = set()
        self._sku_series: dict[str, np.ndarray] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> CompoundBernoulliModel:  # noqa: ARG002
        self._fitted = True
        return self

    def fit_series(self, series_dict: dict[str, np.ndarray]) -> CompoundBernoulliModel:
        """Fit per-SKU Bernoulli-Gamma parameters.

        Parameters
        ----------
        series_dict : {sku_id: np.ndarray}
            Training series for each SKU.
        """
        self._sku_series = {k: v.copy() for k, v in series_dict.items()}
        self._params = {}
        self._skipped_skus = set()

        for uid, y in series_dict.items():
            nz = y[y > 0]
            if len(y) < self.min_history_weeks or len(nz) < MIN_NONZERO:
                self._skipped_skus.add(uid)
                continue
            self._params[uid] = _fit_compound_bernoulli(y)

        self._fitted = True
        logger.info(
            "CompoundBernoulli: fit %d SKUs (%d skipped)",
            len(self._params),
            len(self._skipped_skus),
        )
        return self

    def predict(self, X: np.ndarray, horizon: int) -> ForecastResult:  # noqa: ARG002
        """Generate probabilistic forecasts via bootstrap sampling."""
        if not self._fitted:
            raise RuntimeError("Call fit_series() before predict().")

        uid_order = sorted(self._sku_series.keys())
        n_sku = len(uid_order)
        rng = np.random.default_rng(self.random_seed)

        # Build sample cube (n_sku, horizon, n_samples)
        samples = np.zeros((n_sku, horizon, self.n_samples))

        for i, uid in enumerate(uid_order):
            if uid in self._skipped_skus:
                continue
            p, shape, scale = self._params[uid]
            # Draw occurrence indicators: shape (horizon, n_samples)
            occurs = rng.random((horizon, self.n_samples)) < p
            # Draw demand sizes: shape (horizon, n_samples)
            sizes = rng.gamma(shape, scale, size=(horizon, self.n_samples))
            samples[i] = occurs * sizes

        return ForecastResult.from_samples(samples, self.q_levels)
