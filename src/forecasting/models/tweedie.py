"""Per-SKU Tweedie GLM for lumpy demand.

Design (doc Phase 12):
- Compound Poisson-Gamma (Tweedie with 1 < p < 2, default p=1.5).
- Fit: seasonal GLM with annual Fourier features (sin/cos k=1).
- Mean → quantiles: simulate B=1000 compound Poisson-Gamma paths,
  then use ForecastResult.from_samples().
- Fallback chain (each level tries the next if it fails or has too few data):
    1. seasonal  — GLM with intercept + sin/cos features
    2. intercept — GLM with intercept only (constant mu per horizon)
    3. empirical — use historical nonzero mean as constant forecast
- Smoke check (doc): P50 < mean for right-skewed lumpy demand.
- Registered for: lumpy segment only.

Lazy imports: statsmodels imported inside methods.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

from forecasting import config as _cfg
from forecasting.models.base import ForecastModel, ForecastResult
from forecasting.registry import register_model

logger = logging.getLogger(__name__)

# Model constants
TWEEDIE_POWER: float = 1.5  # compound Poisson-Gamma regime
N_SAMPLES: int = 1000  # Monte-Carlo paths for quantile extraction
MIN_NONZERO: int = 4  # minimum non-zero obs for seasonal fit
MIN_TOTAL: int = 8  # minimum total obs for any fit
SEASON_LENGTH: int = 52  # weekly annual period


# ---------------------------------------------------------------------------
# Simulation helper
# ---------------------------------------------------------------------------


def simulate_tweedie(
    mu: float,
    phi: float,
    p: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw *n_samples* from a Tweedie(mu, phi, p) distribution.

    Uses the compound Poisson-Gamma representation valid for 1 < p < 2:
        lambda_ = mu^(2-p) / (phi*(2-p))   — Poisson rate
        alpha   = (2-p)/(p-1)               — Gamma shape
        theta   = phi*(p-1)*mu^(p-1)        — Gamma scale
    """
    if mu <= 0:
        return np.zeros(n_samples)

    lambda_ = mu ** (2 - p) / (phi * (2 - p))
    alpha = (2 - p) / (p - 1)
    theta = phi * (p - 1) * mu ** (p - 1)

    N = rng.poisson(lambda_, n_samples)
    samples = np.array([rng.gamma(alpha, theta, int(n)).sum() if n > 0 else 0.0 for n in N])
    return samples


# ---------------------------------------------------------------------------
# Fallback chain fit functions
# ---------------------------------------------------------------------------


def _build_seasonal_features(t: np.ndarray, period: int = SEASON_LENGTH) -> np.ndarray:
    """Return design matrix [1, sin(2πt/T), cos(2πt/T)]."""
    angle = 2.0 * np.pi * t / period
    return np.column_stack([np.ones(len(t)), np.sin(angle), np.cos(angle)])


def _fit_seasonal(y: np.ndarray, n_nonzero: int) -> tuple[object, str] | None:
    """Try seasonal Tweedie GLM. Returns (result, 'seasonal') or None."""
    if n_nonzero < MIN_NONZERO:
        return None
    try:
        import statsmodels.api as sm  # lazy

        n = len(y)
        t = np.arange(n, dtype=float)
        X = _build_seasonal_features(t)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            glm = sm.GLM(
                y,
                X,
                family=sm.families.Tweedie(
                    var_power=TWEEDIE_POWER,
                    link=sm.families.links.Log(),
                ),
            )
            res = glm.fit(maxiter=100, disp=False)
        if not res.converged:
            return None
        return res, "seasonal"
    except Exception:
        return None


def _fit_intercept(y: np.ndarray) -> tuple[object, str] | None:
    """Try intercept-only Tweedie GLM. Returns (result, 'intercept') or None."""
    try:
        import statsmodels.api as sm  # lazy

        n = len(y)
        X = np.ones((n, 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            glm = sm.GLM(
                y,
                X,
                family=sm.families.Tweedie(
                    var_power=TWEEDIE_POWER,
                    link=sm.families.links.Log(),
                ),
            )
            res = glm.fit(maxiter=50, disp=False)
        return res, "intercept"
    except Exception:
        return None


def _empirical_mu(y: np.ndarray) -> float:
    """Return mean of non-zero values, or 0 if none."""
    nz = y[y > 0]
    return float(nz.mean()) if len(nz) > 0 else 0.0


# ---------------------------------------------------------------------------
# SKU-level fit result
# ---------------------------------------------------------------------------


class _TweedieSKUFit:
    """Holds fitted model (or fallback params) for one SKU."""

    __slots__ = ("fit_mode", "_res", "_mu_const", "_phi", "_n_train")

    def __init__(
        self,
        fit_mode: str,
        res: object | None,
        mu_const: float,
        phi: float,
        n_train: int,
    ) -> None:
        self.fit_mode = fit_mode  # 'seasonal', 'intercept', 'empirical'
        self._res = res  # statsmodels result (None for empirical)
        self._mu_const = mu_const  # constant mu for empirical fallback
        self._phi = phi
        self._n_train = n_train

    def predict_mu(self, horizon: int) -> np.ndarray:
        """Predict mean demand for each of the next *horizon* steps."""
        if self._res is not None:
            t_future = np.arange(self._n_train, self._n_train + horizon, dtype=float)
            if self.fit_mode == "seasonal":
                X_fut = _build_seasonal_features(t_future)
            else:
                X_fut = np.ones((horizon, 1))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mu = self._res.predict(X_fut)
            return np.maximum(0.0, np.asarray(mu))
        else:
            return np.full(horizon, max(0.0, self._mu_const))


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------


@register_model("tweedie_glm", segments=["lumpy"])
class TweedieGLM(ForecastModel):
    """Per-SKU Tweedie GLM with compound Poisson-Gamma simulation.

    Fit modes (fallback chain, doc Phase 12):
        seasonal  — GLM with intercept + annual Fourier features
        intercept — intercept-only GLM (constant mean)
        empirical — historical nonzero mean (when GLM fails entirely)
    """

    min_history_weeks: int = MIN_TOTAL

    def __init__(
        self,
        q_levels: np.ndarray | None = None,
        n_samples: int = N_SAMPLES,
        tweedie_power: float = TWEEDIE_POWER,
        random_seed: int = _cfg.RANDOM_SEED,
    ) -> None:
        super().__init__(q_levels)
        self.n_samples = n_samples
        self.tweedie_power = tweedie_power
        self.random_seed = random_seed

        self._sku_fits: dict[str, _TweedieSKUFit] = {}
        self._skipped_skus: set[str] = set()
        self._sku_series: dict[str, np.ndarray] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> TweedieGLM:  # noqa: ARG002
        """Not used directly — call fit_series()."""
        self._fitted = True
        return self

    def fit_series(self, series_dict: dict[str, np.ndarray]) -> TweedieGLM:
        """Fit per-SKU Tweedie GLM with fallback chain.

        Parameters
        ----------
        series_dict : {sku_id: np.ndarray}
            Training series for each SKU.
        """
        self._sku_series = {k: v.copy() for k, v in series_dict.items()}
        self._sku_fits = {}
        self._skipped_skus = set()

        mode_counts: dict[str, int] = {"seasonal": 0, "intercept": 0, "empirical": 0}

        for uid, y in series_dict.items():
            nz = (y > 0).sum()

            if len(y) < MIN_TOTAL or nz == 0:
                self._skipped_skus.add(uid)
                continue

            # Try fallback chain
            fit_result = _fit_seasonal(y, int(nz))
            if fit_result is None:
                fit_result = _fit_intercept(y)

            if fit_result is not None:
                res, mode = fit_result
                phi = float(getattr(res, "scale", 1.0))
                phi = max(phi, 0.01)
                self._sku_fits[uid] = _TweedieSKUFit(
                    fit_mode=mode,
                    res=res,
                    mu_const=0.0,
                    phi=phi,
                    n_train=len(y),
                )
            else:
                # Empirical fallback
                mu_emp = _empirical_mu(y)
                self._sku_fits[uid] = _TweedieSKUFit(
                    fit_mode="empirical",
                    res=None,
                    mu_const=mu_emp,
                    phi=1.0,
                    n_train=len(y),
                )

            mode_counts[self._sku_fits[uid].fit_mode] += 1

        self._fitted = True
        logger.info(
            "TweedieGLM: fit %d SKUs (%d skipped) | modes: %s",
            len(self._sku_fits),
            len(self._skipped_skus),
            mode_counts,
        )
        return self

    def predict(self, X: np.ndarray, horizon: int) -> ForecastResult:  # noqa: ARG002
        """Generate probabilistic forecasts via Tweedie simulation."""
        if not self._fitted:
            raise RuntimeError("Call fit_series() before predict().")

        uid_order = sorted(self._sku_series.keys())
        n_sku = len(uid_order)
        rng = np.random.default_rng(self.random_seed)

        # samples shape: (n_sku, horizon, n_samples)
        samples = np.zeros((n_sku, horizon, self.n_samples))

        for i, uid in enumerate(uid_order):
            if uid in self._skipped_skus:
                continue

            fit = self._sku_fits[uid]
            mu_h = fit.predict_mu(horizon)  # (horizon,)

            for h, mu in enumerate(mu_h):
                samples[i, h] = simulate_tweedie(
                    mu=float(mu),
                    phi=fit._phi,
                    p=self.tweedie_power,
                    n_samples=self.n_samples,
                    rng=rng,
                )

        return ForecastResult.from_samples(samples, self.q_levels)
