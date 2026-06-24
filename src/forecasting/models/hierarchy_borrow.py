"""Hierarchy borrowing model for cold_start / short_history SKUs.

For SKUs with < 52 weeks of history, own seasonal pattern is unreliable.
This model borrows the seasonal profile from the product_type hierarchy:
  - Compute median weekly sales profile across sibling SKUs (same product_type)
  - Scale it to match the new SKU's recent observed level
  - Blend own history vs sibling profile weighted by history length:
      weight_own = min(1.0, weeks_of_history / 52)
      weight_sib = 1 - weight_own

Registered for: cold_start, short_history.
If no siblings exist, falls back to SeasonalNaive behaviour (mean).
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from forecasting.models.base import ForecastModel, ForecastResult
from forecasting.registry import register_model

logger = logging.getLogger(__name__)

SEASON = 52
BLEND_FULL_WEEKS = 52  # at this many weeks, full own-history weight


@register_model("hierarchy_borrow", segments=["cold_start"])
class HierarchyBorrowModel(ForecastModel):
    """Seasonal profile borrowed from product_type siblings + own level scaling.

    Call fit_hierarchy() BEFORE fit_series() to provide sibling profiles.
    If fit_hierarchy() is not called, falls back to flat mean forecast.
    """

    def __init__(self, q_levels: np.ndarray | None = None) -> None:
        super().__init__(q_levels)
        self._sku_params: dict[str, dict] = {}
        self._skipped: set[str] = set()
        self._sku_series: dict[str, np.ndarray] = {}
        self._sibling_profiles: dict[str, np.ndarray] = {}  # product_type -> (52,) profile

    def fit_hierarchy(
        self,
        dense: pd.DataFrame,
        sku_product_type: dict[int, str],
        col_sku: str = "sku_id",
        col_ts: str = "timestamp",
        col_sales: str = "sales",
    ) -> "HierarchyBorrowModel":
        """Pre-compute per-product_type median seasonal profile from sibling SKUs.

        Only uses SKUs with >= SEASON weeks of history (they have a real profile).
        """
        profiles: dict[str, list] = {}
        for sku, pt in sku_product_type.items():
            grp = dense[dense[col_sku] == sku].sort_values(col_ts)
            y = grp[col_sales].values
            if len(y) < SEASON:
                continue
            # Last full season as profile
            profile = y[-SEASON:].copy()
            total = profile.sum()
            if total > 0:
                profile = profile / total  # normalise to shape only
            profiles.setdefault(pt, []).append(profile)

        for pt, prof_list in profiles.items():
            self._sibling_profiles[pt] = np.median(np.stack(prof_list), axis=0)

        logger.info(
            "HierarchyBorrow: built %d product_type profiles", len(self._sibling_profiles)
        )
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HierarchyBorrowModel":  # noqa: ARG002
        self._fitted = True
        return self

    def fit_series(
        self,
        series_dict: dict[str, np.ndarray],
        sku_product_type: dict[str, str] | None = None,
    ) -> "HierarchyBorrowModel":
        self._sku_series = {k: v.copy() for k, v in series_dict.items()}
        self._sku_params = {}
        self._skipped = set()

        for uid, y in series_dict.items():
            if len(y) == 0:
                self._skipped.add(uid)
                continue

            T = len(y)
            recent_level = float(y[-8:].mean()) if T >= 8 else float(y.mean())
            recent_level = max(recent_level, 0.0)

            # Own profile (if enough history)
            if T >= SEASON:
                own_profile = y[-SEASON:].copy()
            else:
                own_profile = None

            # Sibling profile
            pt = (sku_product_type or {}).get(uid)
            sib_profile = self._sibling_profiles.get(pt) if pt else None

            # Blend weight
            weight_own = min(1.0, T / BLEND_FULL_WEEKS)
            weight_sib = 1.0 - weight_own

            # Residuals for conformal intervals
            if T >= SEASON:
                residuals = np.abs(y[SEASON:] - y[:T - SEASON])
            elif T >= 4:
                residuals = np.abs(y - y.mean())
            else:
                residuals = np.array([recent_level * 0.5])

            self._sku_params[uid] = {
                "own_profile": own_profile,
                "sib_profile": sib_profile,
                "recent_level": recent_level,
                "weight_own": weight_own,
                "weight_sib": weight_sib,
                "residuals": residuals,
                "T": T,
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
            if uid in self._skipped:
                continue

            p = self._sku_params[uid]
            level = p["recent_level"]
            w_own = p["weight_own"]
            w_sib = p["weight_sib"]
            residuals = p["residuals"]

            # Build blended seasonal template
            if p["own_profile"] is not None and p["sib_profile"] is not None:
                # Both available: blend
                own_norm = p["own_profile"] / max(p["own_profile"].sum(), 1e-9)
                sib_norm = p["sib_profile"]
                blended = w_own * own_norm + w_sib * sib_norm
                # Scale to recent level
                template = blended * level * SEASON
            elif p["sib_profile"] is not None:
                # Only sibling: scale to recent level
                template = p["sib_profile"] * level * SEASON
            elif p["own_profile"] is not None:
                template = p["own_profile"]
            else:
                # Fallback: flat mean
                template = np.full(SEASON, max(level, 0.0))

            template = np.maximum(0.0, template)

            # Point forecast: cycle through template
            point = np.array([template[h % SEASON] for h in range(horizon)])

            # Conformal quantile intervals
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
