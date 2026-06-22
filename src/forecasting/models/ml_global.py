"""Cluster-pooled LightGBM quantile booster.

Design (locked — doc Phase 11):
- Trains one quantile booster per (cluster_id, quantile_level).
  8 clusters × 19 quantiles = 152 boosters.
- Multi-step is DIRECT: horizon_step (1..H) is a feature.
  Each training row: as-of-t feature vector + horizon_step=s, target = sales[t+s].
- Categoricals: sku_id, cluster_id, product_type, status, revenue_tier.
  LightGBM native handling (no encoding needed from caller).
- Early stopping: hold out last 13 weeks of each SKU's training window.
- Target-week features: OFF by default (Option A+, rejected in Phase 11.5 A/B).
  Flag kept for Phase 14 re-check.
- Crossing: 19 boosters trained independently → sort + floor via ForecastResult.
  29% crossing documented; repaired by _finalize().

Findings documented in modeling_decisions.md:
- Pre-sort crossing: 29.0% of adjacent pairs (per-cluster 1.7%–37.5%)
- P10–P90 coverage (held-out window): 70.4% (below guardrail floor 0.75)
- P05–P95 coverage: 87.5%
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecasting import config as _cfg
from forecasting.models.base import ForecastModel, ForecastResult
from forecasting.registry import register_model

logger = logging.getLogger(__name__)

# Architecture constants (locked)
N_CLUSTERS: int = 8
N_QUANTILES: int = 19
N_BOOSTERS: int = N_CLUSTERS * N_QUANTILES  # = 152

# Training constants
EARLY_STOPPING_WEEKS: int = 13  # hold out last 13 weeks per SKU for early stopping
MIN_ROWS_TO_TRAIN: int = 50  # minimum training rows per cluster/quantile

# Categorical feature names
CAT_FEATURES: list[str] = [
    "sku_id",
    "cluster_id",
    "product_type",
    "status",
    "revenue_tier",
]

# Default LightGBM hyperparameters
DEFAULT_LGBM_PARAMS: dict[str, object] = {
    "objective": "quantile",
    "metric": "quantile",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "random_state": _cfg.RANDOM_SEED,
}


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


@dataclass
class LGBMFitResult:
    """Stores the 152 fitted boosters and their feature importance."""

    boosters: dict[tuple[int, float], object]  # (cluster_id, q) -> LGBMRegressor
    feature_names: list[str]
    importance: pd.DataFrame  # columns: feature, cluster_id, q, importance
    n_skipped: int  # clusters/quantile combos skipped (too few rows)


# ---------------------------------------------------------------------------
# Training data builder
# ---------------------------------------------------------------------------


def build_training_rows(
    features_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    horizon: int,
    cutoff: pd.Timestamp | None = None,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Build direct multi-step training rows for all clusters.

    For each SKU i, timestamp t, horizon step s:
        X_row = features[i, t]  (as-of-t)
        X_row["horizon_step"] = s
        y = sales[i, t + s]

    Parameters
    ----------
    features_df : DataFrame, 17539 rows × 114 cols
        Output of add_hierarchy_features().
    segments_df : DataFrame, 229 rows
        Output of segment_and_cluster() — provides cluster_id, revenue_tier.
    horizon : int
        Number of forecast steps to create targets for.
    cutoff : pd.Timestamp | None
        Training cutoff. Rows at t > cutoff are excluded.
        Defaults to max(timestamp) in features_df.
    feature_cols : list[str] | None
        Numeric feature columns to include. Defaults to all non-base columns.

    Returns
    -------
    DataFrame with columns: all feature_cols + horizon_step + cluster_id +
    revenue_tier + categorical cols + target (sales at t+s).
    """
    df = features_df.copy()

    # Merge cluster_id and revenue_tier from segments.
    # Drop pre-existing cols first to avoid _x/_y suffixes from duplicate merge.
    for col in ("cluster_id", "revenue_tier"):
        if col in df.columns:
            df = df.drop(columns=[col])

    seg_cols = [_cfg.COL_SKU_ID, "cluster_id", "revenue_tier"]
    df = df.merge(
        segments_df[seg_cols],
        on=_cfg.COL_SKU_ID,
        how="left",
    )

    if cutoff is None:
        cutoff = df[_cfg.COL_TIMESTAMP].max()

    df = df[df[_cfg.COL_TIMESTAMP] <= cutoff].copy()
    df = df.sort_values([_cfg.COL_SKU_ID, _cfg.COL_TIMESTAMP]).reset_index(drop=True)

    # Default feature columns: everything except base + target
    base_cols = {
        _cfg.COL_TIMESTAMP,
        _cfg.COL_SALES,
        _cfg.COL_LIST_PRICE,
        _cfg.COL_DISCOUNT_PCT,
        _cfg.COL_PRODUCT_TYPE,
        _cfg.COL_STATUS,
        "is_potential_stockout",
    }
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in base_cols]

    # Build target: sales[t + s] for s in 1..horizon
    all_rows = []
    sku_groups = dict(tuple(df.groupby(_cfg.COL_SKU_ID)))

    for _sku, grp in sku_groups.items():
        grp = grp.reset_index(drop=True)
        T = len(grp)
        sales_arr = grp[_cfg.COL_SALES].values

        for s in range(1, horizon + 1):
            # Rows where we can look s steps ahead
            valid = T - s
            if valid <= 0:
                continue
            rows = grp.iloc[:valid].copy()
            rows["horizon_step"] = float(s)
            rows["target"] = sales_arr[s : s + valid]
            all_rows.append(rows)

    if not all_rows:
        return pd.DataFrame()

    train_df = pd.concat(all_rows, ignore_index=True)

    # Encode categoricals as integer codes for LightGBM
    for cat in CAT_FEATURES:
        if cat in train_df.columns:
            train_df[cat] = train_df[cat].astype("category").cat.codes.astype(np.int32)

    return train_df


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------


@register_model(
    "cluster_lgbm",
    segments=["smooth", "erratic", "intermittent", "lumpy", "cold_start"],
)
class ClusterPooledLGBM(ForecastModel):
    """Cluster-pooled LightGBM with 19 quantile boosters × 8 clusters = 152 boosters.

    Multi-step forecasting is DIRECT: horizon_step is a feature, one model
    per (cluster, quantile) combination.
    """

    def __init__(
        self,
        q_levels: np.ndarray | None = None,
        horizon: int = _cfg.FORECAST_HORIZON_WEEKS,
        lgbm_params: dict | None = None,
        early_stopping_rounds: int = 20,
        target_week_features: bool = False,  # Option A+ flag (default OFF)
    ) -> None:
        super().__init__(q_levels)
        self.horizon = horizon
        self.lgbm_params = lgbm_params or dict(DEFAULT_LGBM_PARAMS)
        self.early_stopping_rounds = early_stopping_rounds
        self.target_week_features = target_week_features

        self._fit_result: LGBMFitResult | None = None
        self._feature_cols: list[str] = []
        self._segments_df: pd.DataFrame | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> ClusterPooledLGBM:  # noqa: ARG002
        """Not used directly; call fit_dataframe() instead."""
        self._fitted = True
        return self

    def fit_dataframe(
        self,
        features_df: pd.DataFrame,
        segments_df: pd.DataFrame,
        cutoff: pd.Timestamp | None = None,
        feature_cols: list[str] | None = None,
    ) -> ClusterPooledLGBM:
        """Fit 152 quantile boosters on the full feature grid.

        Parameters
        ----------
        features_df : DataFrame, 17539 rows × 114+ cols
        segments_df : DataFrame, 229 rows (from segment_and_cluster)
        cutoff : pd.Timestamp | None
            Training data cutoff.
        feature_cols : list[str] | None
            Numeric feature columns (excluding base/target/categoricals).
        """
        import lightgbm as lgb  # lazy import

        self._segments_df = segments_df.copy()
        train_df = build_training_rows(features_df, segments_df, self.horizon, cutoff, feature_cols)

        if train_df.empty:
            logger.warning("Empty training DataFrame — nothing to fit.")
            self._fitted = True
            return self

        # Determine input feature columns (numeric + horizon_step)
        base_excl = {
            _cfg.COL_TIMESTAMP,
            _cfg.COL_SALES,
            "target",
            _cfg.COL_LIST_PRICE,
            _cfg.COL_DISCOUNT_PCT,
            "is_potential_stockout",
        }
        # All non-excluded columns (categoricals are already int-coded)
        all_feat = [c for c in train_df.columns if c not in base_excl | {"target"}]
        self._feature_cols = all_feat

        boosters: dict[tuple[int, float], object] = {}
        importance_rows: list[dict] = []
        n_skipped = 0

        cluster_ids = sorted(train_df["cluster_id"].unique())

        for cluster_id in cluster_ids:
            cluster_mask = train_df["cluster_id"] == cluster_id
            cluster_df = train_df[cluster_mask]

            if len(cluster_df) < MIN_ROWS_TO_TRAIN:
                logger.warning(
                    "Cluster %d: only %d rows — skipping all quantiles.",
                    cluster_id,
                    len(cluster_df),
                )
                n_skipped += len(self.q_levels)
                continue

            # Hold out last EARLY_STOPPING_WEEKS horizon-step rows as eval set
            # Sort by timestamp to ensure the eval set is the most recent observations
            cluster_df = cluster_df.sort_values(_cfg.COL_TIMESTAMP)
            n_eval = min(
                EARLY_STOPPING_WEEKS * len(cluster_df["sku_id"].unique()),
                int(0.15 * len(cluster_df)),
            )
            train_idx = slice(0, len(cluster_df) - n_eval)
            eval_idx = slice(len(cluster_df) - n_eval, None)

            X_tr = cluster_df.iloc[train_idx][self._feature_cols].values
            y_tr = cluster_df.iloc[train_idx]["target"].values
            X_ev = cluster_df.iloc[eval_idx][self._feature_cols].values
            y_ev = cluster_df.iloc[eval_idx]["target"].values

            for q in self.q_levels:
                params = dict(self.lgbm_params)
                params["alpha"] = float(q)

                model = lgb.LGBMRegressor(**params)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(
                        X_tr,
                        y_tr,
                        eval_set=[(X_ev, y_ev)],
                        callbacks=[
                            lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                            lgb.log_evaluation(period=-1),
                        ],
                    )

                boosters[(cluster_id, float(q))] = model
                importance_rows.append(
                    {
                        "feature": self._feature_cols,
                        "cluster_id": cluster_id,
                        "q": float(q),
                        "importance": model.feature_importances_.tolist(),
                    }
                )

            logger.debug(
                "Cluster %d: fitted %d boosters (%d train / %d eval rows)",
                cluster_id,
                len(self.q_levels),
                len(X_tr),
                len(X_ev),
            )

        # Flatten importance into a tidy DataFrame
        imp_rows = []
        for row in importance_rows:
            for feat, imp in zip(row["feature"], row["importance"], strict=False):
                imp_rows.append(
                    {
                        "feature": feat,
                        "cluster_id": row["cluster_id"],
                        "q": row["q"],
                        "importance": imp,
                    }
                )
        importance_df = pd.DataFrame(imp_rows)

        self._fit_result = LGBMFitResult(
            boosters=boosters,
            feature_names=self._feature_cols,
            importance=importance_df,
            n_skipped=n_skipped,
        )

        n_fitted = len(boosters)
        logger.info(
            "ClusterPooledLGBM: fitted %d/%d boosters (%d skipped)",
            n_fitted,
            N_BOOSTERS,
            n_skipped,
        )
        self._fitted = True
        return self

    def predict(self, X: np.ndarray, horizon: int) -> ForecastResult:  # noqa: ARG002
        """Not the primary interface; use predict_dataframe() for real data."""
        raise NotImplementedError(
            "Use predict_dataframe(features_df, segments_df, horizon) for this model."
        )

    def predict_dataframe(
        self,
        features_df: pd.DataFrame,
        segments_df: pd.DataFrame,
        horizon: int | None = None,
        cutoff: pd.Timestamp | None = None,
    ) -> ForecastResult:
        """Generate probabilistic forecasts for all SKUs.

        Parameters
        ----------
        features_df : DataFrame
            Feature grid including the forecast origin rows.
        segments_df : DataFrame
            Segment/cluster assignments for each SKU.
        horizon : int | None
            Forecast horizon. Defaults to self.horizon.
        cutoff : pd.Timestamp | None
            Forecast origin timestamp. Defaults to max(timestamp).

        Returns
        -------
        ForecastResult, shape (n_sku, horizon, n_q).
        """
        if self._fit_result is None:
            raise RuntimeError("Call fit_dataframe() before predict_dataframe().")

        H = horizon or self.horizon
        n_q = len(self.q_levels)

        # Get the forecast origin rows (last available row per SKU)
        df = features_df.copy()
        for col in ("cluster_id", "revenue_tier"):
            if col in df.columns:
                df = df.drop(columns=[col])
        df = df.merge(
            segments_df[[_cfg.COL_SKU_ID, "cluster_id", "revenue_tier"]],
            on=_cfg.COL_SKU_ID,
            how="left",
        )
        if cutoff is not None:
            df = df[df[_cfg.COL_TIMESTAMP] <= cutoff]

        # Take most recent row per SKU
        origin = df.sort_values(_cfg.COL_TIMESTAMP).groupby(_cfg.COL_SKU_ID).last().reset_index()

        # Encode categoricals
        for cat in CAT_FEATURES:
            if cat in origin.columns:
                origin[cat] = origin[cat].astype("category").cat.codes.astype(np.int32)

        sku_ids = sorted(origin[_cfg.COL_SKU_ID].unique())
        n_sku = len(sku_ids)
        sku_idx = {sku: i for i, sku in enumerate(sku_ids)}

        q_cube = np.zeros((n_sku, H, n_q))

        for qi, q in enumerate(self.q_levels):
            for h in range(1, H + 1):
                # Build prediction rows: one per SKU with horizon_step = h
                pred_rows = origin.copy()
                pred_rows["horizon_step"] = float(h)

                feat_cols = [c for c in self._feature_cols if c in pred_rows.columns]

                # Predict per cluster
                for cluster_id in pred_rows["cluster_id"].unique():
                    if pd.isna(cluster_id):
                        continue
                    key = (int(cluster_id), float(q))
                    if key not in self._fit_result.boosters:
                        continue
                    booster = self._fit_result.boosters[key]
                    c_mask = pred_rows["cluster_id"] == cluster_id
                    X_c = pred_rows.loc[c_mask, feat_cols].values
                    preds = booster.predict(X_c)
                    c_skus = pred_rows.loc[c_mask, _cfg.COL_SKU_ID].values
                    for sku, pred in zip(c_skus, preds, strict=False):
                        i = sku_idx.get(int(sku))
                        if i is not None:
                            q_cube[i, h - 1, qi] = float(pred)

        return ForecastResult.from_quantiles(q_cube, self.q_levels)

    @property
    def feature_importance(self) -> pd.DataFrame | None:
        """Return tidy importance DataFrame (feature, cluster_id, q, importance)."""
        if self._fit_result is None:
            return None
        return self._fit_result.importance

    @property
    def n_boosters_fitted(self) -> int:
        """Number of boosters successfully fitted."""
        if self._fit_result is None:
            return 0
        return len(self._fit_result.boosters)
