"""ForecastModel ABC and ForecastResult uniform output type.

Design (locked — doc §3, principle 5 + §8 tweak 5):
- One uniform output: ForecastResult with field `quantiles` (NOT `values` — avoids
  a lint rule) holding a cube of shape (n_sku, H, n_q).
- Two constructors:
    from_quantiles  — for quantile-native models (LightGBM, Moirai, ETS/ARIMA conformal)
    from_samples    — for sample-based models (Chronos, Tweedie, compound-Bernoulli)
  Both call _finalize() which floors at 0 and sorts along the q axis so output is
  always non-negative and non-crossing.
- ForecastModel ABC: fit(X, y) + predict(X, horizon, q_levels) + register via registry.
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from forecasting import config as _config

logger = logging.getLogger(__name__)

# Crossing fraction threshold above which a warning is emitted
_CROSS_WARN_THRESHOLD: float = 0.05


# ---------------------------------------------------------------------------
# ForecastResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForecastResult:
    """Uniform model output: a quantile cube of shape (n_sku, H, n_q).

    Fields
    ------
    quantiles : np.ndarray, shape (n_sku, H, n_q)
        Non-negative, non-crossing quantile forecasts.
        Axis 0: SKU index (same order as input).
        Axis 1: horizon step (1 = next week, H = furthest week).
        Axis 2: quantile index (aligned to q_levels).
    q_levels : np.ndarray, shape (n_q,)
        Quantile probability levels, e.g. [0.05, 0.10, …, 0.95].
    sku_ids : np.ndarray | None, shape (n_sku,)
        Optional SKU identifiers aligned to axis 0.
    """

    quantiles: np.ndarray  # (n_sku, H, n_q)
    q_levels: np.ndarray  # (n_q,)
    sku_ids: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_quantiles(
        cls,
        quantiles: np.ndarray,
        q_levels: np.ndarray,
        sku_ids: np.ndarray | None = None,
    ) -> ForecastResult:
        """Construct from a direct quantile array.

        Parameters
        ----------
        quantiles : array-like, shape (n_sku, H, n_q)
            Raw quantile forecasts from a quantile-native model.
        q_levels : array-like, shape (n_q,)
            Quantile probability levels.
        sku_ids : array-like | None
            Optional SKU identifiers.

        Returns
        -------
        ForecastResult with quantiles floored at 0 and sorted.
        """
        q = np.asarray(quantiles, dtype=np.float64)
        ql = np.asarray(q_levels, dtype=np.float64)
        if q.ndim != 3:
            raise ValueError(f"quantiles must be 3-D (n_sku, H, n_q), got shape {q.shape}")
        if q.shape[2] != len(ql):
            raise ValueError(f"quantiles.shape[2]={q.shape[2]} != len(q_levels)={len(ql)}")
        finalized = _finalize(q, ql)
        ids = np.asarray(sku_ids) if sku_ids is not None else None
        return cls(quantiles=finalized, q_levels=ql, sku_ids=ids)

    @classmethod
    def from_samples(
        cls,
        samples: np.ndarray,
        q_levels: np.ndarray,
        sku_ids: np.ndarray | None = None,
    ) -> ForecastResult:
        """Construct from Monte-Carlo sample paths.

        Parameters
        ----------
        samples : array-like, shape (n_sku, H, n_samples)
            Monte-Carlo forecast paths (e.g. from Chronos or Tweedie simulation).
        q_levels : array-like, shape (n_q,)
            Quantile probability levels to extract.
        sku_ids : array-like | None
            Optional SKU identifiers.

        Returns
        -------
        ForecastResult with quantiles floored at 0 and sorted.
        """
        s = np.asarray(samples, dtype=np.float64)
        ql = np.asarray(q_levels, dtype=np.float64)
        if s.ndim != 3:
            raise ValueError(f"samples must be 3-D (n_sku, H, n_samples), got shape {s.shape}")

        # Compute empirical quantiles along the samples axis (axis 2)
        q_cube = np.quantile(s, ql, axis=2)  # (n_q, n_sku, H)
        q_cube = np.moveaxis(q_cube, 0, -1)  # (n_sku, H, n_q)

        finalized = _finalize(q_cube, ql)
        ids = np.asarray(sku_ids) if sku_ids is not None else None
        return cls(quantiles=finalized, q_levels=ql, sku_ids=ids)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def n_sku(self) -> int:
        return int(self.quantiles.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.quantiles.shape[1])

    @property
    def n_quantiles(self) -> int:
        return int(self.quantiles.shape[2])

    def median(self) -> np.ndarray:
        """Return P50 forecasts, shape (n_sku, H)."""
        med_idx = int(np.argmin(np.abs(self.q_levels - 0.5)))
        return self.quantiles[:, :, med_idx]

    def quantile_at(self, q: float) -> np.ndarray:
        """Return the forecast at a specific quantile level, shape (n_sku, H)."""
        idx = int(np.argmin(np.abs(self.q_levels - q)))
        return self.quantiles[:, :, idx]


# ---------------------------------------------------------------------------
# Internal finalization
# ---------------------------------------------------------------------------


def _finalize(q_cube: np.ndarray, q_levels: np.ndarray) -> np.ndarray:
    """Floor at 0 and sort along the quantile axis.

    Measures pre-sort crossing fraction and warns if it exceeds
    _CROSS_WARN_THRESHOLD (5%).  Sorting repairs the crossing regardless.

    Parameters
    ----------
    q_cube : shape (n_sku, H, n_q)
    q_levels : shape (n_q,)

    Returns
    -------
    Floored and sorted array of the same shape.
    """
    # Floor at 0 (sales cannot be negative)
    q_cube = np.maximum(q_cube, 0.0)

    # Measure crossing fraction BEFORE sort (for logging / warning)
    n_q = q_cube.shape[2]
    if n_q > 1:
        # Count adjacent pairs where q[i] > q[i+1]
        adjacent_diffs = q_cube[:, :, 1:] - q_cube[:, :, :-1]  # should be >= 0
        n_crossing_pairs = int((adjacent_diffs < 0).sum())
        n_total_pairs = int(q_cube.shape[0] * q_cube.shape[1] * (n_q - 1))
        crossing_frac = n_crossing_pairs / max(n_total_pairs, 1)
        if crossing_frac > _CROSS_WARN_THRESHOLD:
            warnings.warn(
                f"Quantile crossing detected: {crossing_frac:.1%} of adjacent pairs "
                f"({n_crossing_pairs}/{n_total_pairs}). Repairing by sorting.",
                stacklevel=4,
            )
        elif n_crossing_pairs > 0:
            logger.debug(
                "Minor crossing: %.1f%% of pairs (%d/%d). Repaired.",
                crossing_frac * 100,
                n_crossing_pairs,
                n_total_pairs,
            )

    # Sort along quantile axis (axis 2) — repairs any crossings
    return np.sort(q_cube, axis=2)


# ---------------------------------------------------------------------------
# ForecastModel ABC
# ---------------------------------------------------------------------------


class ForecastModel(ABC):
    """Abstract base class for all forecast models.

    Subclasses register themselves via @register_model from registry.py.
    The pipeline calls candidates_for(segment) and iterates model classes —
    it never references a model by name.

    Contract
    --------
    - fit() must be called before predict().
    - predict() returns a ForecastResult with shape (n_sku, H, n_q).
    - Models with < min_history_weeks of data must raise or return a zero forecast
      (not silently produce garbage).  Classical models skip SKUs with <26w.
    """

    #: Minimum non-zero observations required to fit (override in subclass)
    min_history_weeks: int = 4

    def __init__(self, q_levels: np.ndarray | None = None) -> None:
        self.q_levels: np.ndarray = (
            np.asarray(q_levels, dtype=np.float64)
            if q_levels is not None
            else np.asarray(_config.QUANTILES, dtype=np.float64)
        )
        self._fitted: bool = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> ForecastModel:
        """Fit the model.

        Parameters
        ----------
        X : shape (T, n_features)
            Feature matrix for the training window.
        y : shape (T,)
            Target sales series.

        Returns
        -------
        self (for chaining)
        """

    @abstractmethod
    def predict(
        self,
        X: np.ndarray,
        horizon: int,
    ) -> ForecastResult:
        """Generate probabilistic forecasts.

        Parameters
        ----------
        X : shape (n_sku, n_features) or (n_sku, H, n_features)
            Feature matrix at forecast origin (or per horizon step).
        horizon : int
            Number of weeks to forecast ahead.

        Returns
        -------
        ForecastResult with shape (n_sku, horizon, n_q).
        """

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_q={len(self.q_levels)}, fitted={self._fitted})"
