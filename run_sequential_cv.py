"""
run_sequential_cv.py
====================
Sequential learning across 4 rolling-origin folds.

Design:
- Fold 1 (warmup): fit all models, measure per-segment WAPE, tune LightGBM
  hyperparams and identify worst features.
- Fold 2: apply learnings from fold 1 (updated LightGBM params, feature mask,
  segment routing). Measure again.
- Fold 3: apply learnings from fold 2. Measure again.
- Fold 4: apply learnings from fold 3. Final routing locked.
- Final forecast: use fold-4-tuned routing and params on full history.

Per-fold outputs:
  - per-segment WAPE
  - SKU distribution (count + last-26w volume)
  - which model won which segment
  - what changed vs prior fold

Per-segment model routing (from docs):
  smooth/erratic  -> Theta (fold 3 best: 0.567), fallback SeasonalNaive
  lumpy           -> SeasonalNaive (fold 3 best: 0.565)
  intermittent    -> CompoundBernoulli / CrostonSBA
  cold_start      -> LightGBM (cluster-pooled with 114 features)
  discontinued    -> ZeroForecast

Sequential LightGBM tuning:
  Each fold measures per-cluster WAPE for LightGBM.
  If LightGBM underperforms SeasonalNaive on a cluster, increase
  num_leaves and reduce learning_rate for next fold.
  Feature importance from each fold drops bottom-20% features for next fold.

Run:
    python run_sequential_cv.py

Outputs (outputs/sequential_cv/):
    forecast_26w.csv          final 26-week P10/P50/P90
    fold_N_summary.csv        per-fold per-segment results (N=1..4)
    sequential_log.txt        what changed each fold and why
    cv_wape_final.csv         consolidated fold x model x segment WAPE
"""

from __future__ import annotations

import sys
import warnings
import logging
from pathlib import Path
from copy import deepcopy
from io import StringIO

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

OUT = ROOT / "outputs" / "sequential_cv"
OUT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.WARNING)

# ── Constants ─────────────────────────────────────────────────────────────────
HORIZON = 26
SEASON  = 52
MIN_TRAIN_WEEKS = 52   # eligibility filter for WAPE scoring
SELECTION_FOLDS = {2, 3, 4}

# Per-segment candidate models — V2 8-class routing
SEG_CANDIDATES = {
    "smooth":          ["theta", "auto_ets", "seasonal_naive", "cluster_lgbm"],
    "smooth_growing":  ["patchtst", "theta", "trend_seasonal", "auto_ets", "cluster_lgbm"],
    "smooth_stable":   ["seasonal_naive", "recent_level", "auto_ets", "cluster_lgbm"],
    "erratic":         ["patchtst", "theta", "trend_seasonal", "seasonal_naive", "cluster_lgbm", "hurdle"],
    "promo_driven":    ["cluster_lgbm", "hurdle", "seasonal_naive", "compound_bernoulli"],
    "lumpy":           ["seasonal_naive", "compound_bernoulli", "tsb", "tweedie_glm",
                        "hurdle", "cluster_lgbm"],
    "intermittent":    ["seasonal_naive", "hurdle", "compound_bernoulli", "croston_sba",
                        "cluster_lgbm"],
    "cold_start":      ["hierarchy_borrow", "chronos_tiny", "seasonal_naive",
                        "trend_seasonal", "cluster_lgbm"],
    "discontinued":    ["zero_forecast"],
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_pipeline_data():
    from forecasting.io import load_all
    from forecasting.lifecycle import infer_lifecycle
    from forecasting.densify import densify
    from forecasting.features import build_features, add_cluster_features, add_hierarchy_features
    from forecasting.segment import segment_and_cluster

    print("[1] Loading + densify...")
    data = load_all(ROOT / "data/raw")
    lc   = infer_lifecycle(data.sales, data.master)
    dr   = densify(data.sales, lc, data.joined)

    print("[2] Building 114 features...")
    feats = build_features(dr.dense, lc)
    segs  = segment_and_cluster(feats.features, lc)
    full  = add_cluster_features(feats.features, segs.segments)
    full  = add_hierarchy_features(full)

    print(f"    Features: {full.shape[1]} cols  |  SKUs: {segs.segments.shape[0]}")
    print(f"    Segments: {segs.segments.sb_class.value_counts().to_dict()}")
    return dr.dense, full, lc, segs


# ── Fold origin computation ───────────────────────────────────────────────────

def get_fold_origins(dense, n_folds=4):
    from forecasting import config
    ts_max = dense[config.COL_TIMESTAMP].max()
    return {
        fold: ts_max - pd.Timedelta(weeks=(n_folds - fold + 1) * HORIZON)
        for fold in range(1, n_folds + 1)
    }


# ── WAPE helpers ──────────────────────────────────────────────────────────────

def plain_wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    d = float(actual.sum())
    return float(np.abs(actual - forecast).sum() / d) if d > 0 else 1.0


def eligible_mask(sku_order, train, actuals, sku_segments, config):
    """>=52w training, non-discontinued, actual>0 in holdout."""
    from forecasting import config as cfg
    mask = []
    for i, sku in enumerate(sku_order):
        y = train[train[cfg.COL_SKU_ID] == sku][cfg.COL_SALES].values
        seg = sku_segments.get(sku, "cold_start")
        has_actual = actuals[i].sum() > 0
        mask.append(len(y) >= MIN_TRAIN_WEEKS and seg != "discontinued" and has_actual)
    return np.array(mask)


# ── Local SB segment at fold origin ──────────────────────────────────────────

def local_segment(sku, train, seg_df, config):
    """Recompute segment at fold origin to avoid stale global flags."""
    y = train[train[config.COL_SKU_ID] == sku][config.COL_SALES].values
    if len(y) == 0:
        return "cold_start"
    # Seasonal dormancy: zero last 26w but had sales in prior year -> not discontinued
    tail = y[-26:] if len(y) >= 26 else y
    if float(tail.sum()) == 0:
        prior = y[-78:-26] if len(y) >= 78 else (y[:-26] if len(y) > 26 else np.array([]))
        if len(prior) == 0 or float(prior.sum()) == 0:
            return "discontinued"
        # Had prior year activity: use global SB class (seasonal product)
    row = seg_df[seg_df[config.COL_SKU_ID] == sku]
    return str(row["sb_class"].iloc[0]) if len(row) > 0 else "cold_start"


# ── Model factory ─────────────────────────────────────────────────────────────

def make_models(lgbm_params: dict, feature_cols: list[str] | None = None,
                segment_as_cluster: bool = False) -> dict:
    """Instantiate all candidate models with current hyperparameters.

    segment_as_cluster=True: LightGBM uses SB segment as pooling unit
    instead of K-means clusters (lever 2 from slide 23).
    """
    import forecasting.models.baseline
    import forecasting.models.classical
    import forecasting.models.intermittent
    from forecasting import config
    from forecasting.models.baseline import (
        SeasonalNaive, TrendSeasonalModel, RecentLevelModel, ZeroForecast
    )
    q = np.array(config.QUANTILES)

    models = {
        "seasonal_naive":    SeasonalNaive(q_levels=q),
        "trend_seasonal":    TrendSeasonalModel(q_levels=q),
        "recent_level":      RecentLevelModel(q_levels=q),
        "zero_forecast":     ZeroForecast(q_levels=q),
    }
    try:
        from forecasting.models.classical import AutoETSModel, ThetaModel
        models["auto_ets"] = AutoETSModel(q_levels=q)
        models["theta"]    = ThetaModel(q_levels=q)
    except Exception as e:
        print(f"  [warn] classical models: {e}")
    try:
        from forecasting.models.intermittent import CrostonSBAModel, TSBModel, CompoundBernoulliModel
        models["croston_sba"]       = CrostonSBAModel(q_levels=q)
        models["tsb"]               = TSBModel(q_levels=q)
        models["compound_bernoulli"] = CompoundBernoulliModel(q_levels=q)
    except Exception as e:
        print(f"  [warn] intermittent models: {e}")
    try:
        from forecasting.models.ml_global import ClusterPooledLGBM
        lgbm_inst = ClusterPooledLGBM(q_levels=q, lgbm_params=lgbm_params)
        # segment_as_cluster: store flag so CV harness can override cluster_id
        lgbm_inst._segment_as_cluster = segment_as_cluster
        models["cluster_lgbm"] = lgbm_inst
    except Exception as e:
        print(f"  [warn] LightGBM: {e}")

    # Chronos-T5-tiny — zero-shot foundation model, best for cold_start/short_history
    try:
        from forecasting.models.foundation import ChronosTiny
        models["chronos_tiny"] = ChronosTiny(q_levels=q)
        print("  [ok] Chronos-T5-tiny loaded")
    except Exception as e:
        print(f"  [warn] Chronos skipped (needs torch): {e}")

    # Hurdle model — two-part Bernoulli×Gamma for intermittent/promo_driven
    try:
        from forecasting.models.hurdle import HurdleModel
        models["hurdle"] = HurdleModel(q_levels=q)
        print("  [ok] HurdleModel loaded")
    except Exception as e:
        print(f"  [warn] Hurdle skipped: {e}")

    # Tweedie GLM — compound Poisson-Gamma for lumpy
    try:
        import forecasting.models.tweedie  # noqa: F401 — registers tweedie_glm
        from forecasting.registry import MODEL_REGISTRY
        TweedieGLM = MODEL_REGISTRY.get("tweedie_glm")
        if TweedieGLM:
            models["tweedie_glm"] = TweedieGLM(q_levels=q)
            print("  [ok] TweedieGLM loaded")
    except Exception as e:
        print(f"  [warn] Tweedie skipped: {e}")

    # HierarchyBorrow — sibling-profile borrowing for cold_start
    try:
        from forecasting.models.hierarchy_borrow import HierarchyBorrowModel
        models["hierarchy_borrow"] = HierarchyBorrowModel(q_levels=q)
        print("  [ok] HierarchyBorrowModel loaded")
    except Exception as e:
        print(f"  [warn] HierarchyBorrow skipped: {e}")

    # PatchTST — Transformer for trend extrapolation (erratic/growing SKUs)
    try:
        import os; os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        from forecasting.models.patchtst import PatchTSTModel
        # Fast config for CV (30 epochs, small model)
        models["patchtst"] = PatchTSTModel(
            q_levels=q, epochs=30, d_model=64, n_layers=2,
            batch_size=32, lr=5e-4
        )
        print("  [ok] PatchTST loaded")
    except Exception as e:
        print(f"  [warn] PatchTST skipped: {e}")

    return models


# ── Run one fold ──────────────────────────────────────────────────────────────

def run_one_fold(
    fold_id: int,
    origin: pd.Timestamp,
    dense: pd.DataFrame,
    full_feats: pd.DataFrame,
    segs,
    lgbm_params: dict,
    routing: dict,           # seg -> model_name (current best guess)
    drop_features: set,      # features to exclude this fold
) -> dict:
    """
    Runs ALL candidate models on this fold.
    Returns dict with:
      wape_by_model_seg  -> {model_name: {seg: wape}}
      wape_by_model      -> {model_name: overall_wape}
      feature_importance -> {feature: importance} from LightGBM
      sku_distribution   -> {seg: {n_skus, vol_26w}}
      best_per_seg       -> {seg: model_name}
    """
    from forecasting import config
    from forecasting.validate import _build_fold_data, _run_model_on_fold

    holdout_end = origin + pd.Timedelta(weeks=HORIZON)
    print(f"\n  Fold {fold_id} | {origin.date()} -> {holdout_end.date()}")

    # Slice features to current fold (drop excluded features)
    fold_feats = full_feats.copy()
    if drop_features:
        fold_feats = fold_feats.drop(columns=[c for c in drop_features if c in fold_feats.columns])

    train_dense, train_feats, actuals, sku_order = _build_fold_data(
        dense, fold_feats, segs, origin, HORIZON
    )
    if len(sku_order) == 0:
        print("  No SKUs in holdout — skipping.")
        return {}

    train = dense[dense[config.COL_TIMESTAMP] <= origin]

    # Local segments at this fold origin
    sku_segments = {
        sku: local_segment(sku, train, segs.segments, config)
        for sku in sku_order
    }

    # Eligibility mask
    el_mask = eligible_mask(sku_order, train, actuals, sku_segments, config)
    n_eligible = int(el_mask.sum())
    print(f"  Eligible SKUs: {n_eligible} / {len(sku_order)}")

    # SKU distribution
    seg_dist = {}
    for sku in sku_order:
        seg = sku_segments[sku]
        y = train[train[config.COL_SKU_ID] == sku][config.COL_SALES].values
        vol = float(y[-26:].sum()) if len(y) >= 26 else float(y.sum())
        if seg not in seg_dist:
            seg_dist[seg] = {"n_skus": 0, "vol_26w": 0.0}
        seg_dist[seg]["n_skus"] += 1
        seg_dist[seg]["vol_26w"] += vol

    print(f"  SKU distribution:")
    for seg in ["smooth","erratic","lumpy","intermittent","cold_start","discontinued"]:
        if seg in seg_dist:
            d = seg_dist[seg]
            print(f"    {seg:15s}: {d['n_skus']:3d} SKUs  last-26w-vol={d['vol_26w']:.0f}")

    # Build + run models
    models = make_models(lgbm_params)
    wape_by_model     = {}
    wape_by_model_seg = {}
    feat_importance   = {}
    p50_by_model      = {}

    for model_name, model in models.items():
        try:
            out = _run_model_on_fold(
                model=model,
                model_name=model_name,
                fold=fold_id,
                train_dense=train_dense,
                train_features=train_feats,
                segments=segs,
                actuals=actuals,
                sku_order=sku_order,
                cutoff=origin,
                horizon=HORIZON,
            )
        except Exception as e:
            print(f"    {model_name}: FAILED ({e})")
            continue

        if out is None:
            continue

        fm, pred = out
        p50_idx = pred.quantiles.shape[2] // 2
        p50 = pred.quantiles[:, :, p50_idx]  # (n_sku, H)
        p50_by_model[model_name] = p50

        # Overall WAPE on eligible SKUs
        act_e = actuals[el_mask].flatten()
        p50_e = p50[el_mask].flatten()
        w = plain_wape(act_e, p50_e)
        wape_by_model[model_name] = w

        # Per-segment WAPE
        seg_wapes = {}
        for seg in ["smooth","erratic","lumpy","intermittent","cold_start"]:
            seg_mask = np.array([
                sku_segments.get(sku,"") == seg
                and len(train[train[config.COL_SKU_ID]==sku][config.COL_SALES].values) >= MIN_TRAIN_WEEKS
                and actuals[i].sum() > 0
                for i, sku in enumerate(sku_order)
            ])
            if seg_mask.sum() == 0:
                continue
            a = actuals[seg_mask].flatten()
            p = p50[seg_mask].flatten()
            seg_wapes[seg] = plain_wape(a, p)
        wape_by_model_seg[model_name] = seg_wapes

        # LightGBM feature importance
        if model_name == "cluster_lgbm" and hasattr(model, "_fit_result") and model._fit_result:
            try:
                fi = model._fit_result.feature_importance  # dict or DataFrame
                if isinstance(fi, dict):
                    feat_importance = fi
                elif isinstance(fi, pd.DataFrame):
                    feat_importance = dict(zip(fi.iloc[:,0], fi.iloc[:,1]))
            except Exception:
                pass

    # Print results
    print(f"  Model results (WAPE, eligible SKUs):")
    for m, w in sorted(wape_by_model.items(), key=lambda x: x[1]):
        seg_str = "  ".join(
            f"{s}={v:.3f}" for s, v in wape_by_model_seg.get(m, {}).items()
        )
        status = "PASS" if w < 0.50 else ("OK" if w < 0.60 else "FAIL")
        print(f"    {m:25s}: {w:.4f} [{status}]  {seg_str}")

    # Best model per segment
    best_per_seg = {}
    for seg in ["smooth","erratic","lumpy","intermittent","cold_start"]:
        candidates = SEG_CANDIDATES.get(seg, ["seasonal_naive"])
        best_w = float("inf")
        best_m = "seasonal_naive"
        for m in candidates:
            w = wape_by_model_seg.get(m, {}).get(seg, float("inf"))
            if w < best_w:
                best_w = w
                best_m = m
        best_per_seg[seg] = (best_m, best_w)

    print(f"  Best per segment:")
    for seg, (m, w) in best_per_seg.items():
        print(f"    {seg:15s}: {m}  WAPE={w:.4f}")

    return {
        "wape_by_model": wape_by_model,
        "wape_by_model_seg": wape_by_model_seg,
        "feature_importance": feat_importance,
        "sku_distribution": seg_dist,
        "best_per_seg": best_per_seg,
        "sku_segments": sku_segments,
        "sku_order": sku_order,
        "actuals": actuals,
        "p50_by_model": p50_by_model,
        "eligible_mask": el_mask,
    }


# ── Sequential learning: update params from fold results ─────────────────────

def update_from_fold(
    fold_id: int,
    fold_result: dict,
    lgbm_params: dict,
    routing: dict,
    drop_features: set,
    log: list,
) -> tuple[dict, dict, set]:
    """
    Learn from fold_result and update:
      lgbm_params  - hyperparameter adjustments
      routing      - seg -> best model
      drop_features - features to drop next fold

    Returns updated (lgbm_params, routing, drop_features).
    """
    if not fold_result:
        return lgbm_params, routing, drop_features

    wape_by_model = fold_result["wape_by_model"]
    wape_by_seg   = fold_result["wape_by_model_seg"]
    best_per_seg  = fold_result["best_per_seg"]
    feat_imp      = fold_result["feature_importance"]

    changes = [f"\n--- Fold {fold_id} learnings ---"]

    # 1. Update routing from best-per-segment results
    new_routing = {}
    for seg, (best_m, best_w) in best_per_seg.items():
        old_m = routing.get(seg, "seasonal_naive")
        new_routing[seg] = best_m
        if old_m != best_m:
            changes.append(f"  routing[{seg}]: {old_m} -> {best_m}  (WAPE {best_w:.4f})")
    new_routing["discontinued"] = "zero_forecast"

    # 2. LightGBM tuning based on its WAPE vs SeasonalNaive
    lgbm_w = wape_by_model.get("cluster_lgbm", float("inf"))
    sn_w   = wape_by_model.get("seasonal_naive", float("inf"))
    new_lgbm = dict(lgbm_params)

    if lgbm_w > sn_w:
        # LightGBM lost to SeasonalNaive — increase complexity
        old_leaves = new_lgbm.get("num_leaves", 31)
        old_lr     = new_lgbm.get("learning_rate", 0.05)
        new_leaves = min(int(old_leaves * 1.5), 127)
        new_lr     = round(old_lr * 0.7, 4)
        new_lgbm["num_leaves"]     = new_leaves
        new_lgbm["learning_rate"]  = new_lr
        new_lgbm["n_estimators"]   = min(int(new_lgbm.get("n_estimators", 300) * 1.2), 600)
        changes.append(
            f"  LGBM: num_leaves {old_leaves}->{new_leaves}, "
            f"lr {old_lr}->{new_lr}  (lgbm_wape={lgbm_w:.4f} > sn_wape={sn_w:.4f})"
        )
    else:
        # LightGBM beat SeasonalNaive — slight regularisation increase
        new_lgbm["reg_alpha"]  = round(new_lgbm.get("reg_alpha", 0.1) * 1.2, 4)
        new_lgbm["reg_lambda"] = round(new_lgbm.get("reg_lambda", 0.1) * 1.2, 4)
        changes.append(
            f"  LGBM: LightGBM won (wape={lgbm_w:.4f}). "
            f"Increase regularisation slightly."
        )

    # 3. Feature pruning: drop bottom-20% by importance
    new_drop = set(drop_features)
    if feat_imp:
        fi_series = pd.Series(feat_imp).sort_values()
        n_drop    = max(1, int(len(fi_series) * 0.20))
        bottom    = set(fi_series.head(n_drop).index.tolist())
        # Never drop core lag/rolling/Fourier cols
        protected = {
            "lag_1","lag_2","lag_4","lag_8","lag_13","lag_26","lag_52",
            "roll4_mean","roll13_mean","roll26_mean","roll52_mean",
            "horizon_step","week_of_year","sku_age_weeks",
            "hol_christmas","hol_black_friday","hol_thanksgiving",
        }
        to_drop = bottom - protected
        new_drop = new_drop | to_drop
        if to_drop:
            changes.append(
                f"  Features dropped (bottom {n_drop} by importance): "
                f"{sorted(to_drop)[:8]}{'...' if len(to_drop)>8 else ''}"
            )

    for c in changes:
        log.append(c)
        print(c)

    return new_lgbm, new_routing, new_drop


# ── Final forecast using locked routing + full history ────────────────────────

def make_final_forecast(dense, full_feats, segs, final_routing, lgbm_params):
    from forecasting import config
    from forecasting.registry import MODEL_REGISTRY
    import forecasting.models.baseline
    import forecasting.models.classical
    import forecasting.models.intermittent

    print("\n[Final] 26-week forecast from full history...")
    ts_max = dense[config.COL_TIMESTAMP].max()
    train  = dense[dense[config.COL_TIMESTAMP] <= ts_max]
    q      = np.array(config.QUANTILES)
    p10_i  = int(np.argmin(np.abs(q - 0.10)))
    p50_i  = len(q) // 2
    p90_i  = int(np.argmin(np.abs(q - 0.90)))

    seg_map = dict(zip(
        segs.segments[config.COL_SKU_ID].astype(int),
        segs.segments["sb_class"]
    ))
    all_skus = sorted(seg_map.keys())

    # Group by segment
    by_seg: dict[str, list] = {}
    for sku in all_skus:
        s = seg_map.get(sku, "cold_start")
        by_seg.setdefault(s, []).append(sku)

    horizon_dates = [ts_max + pd.Timedelta(weeks=h+1) for h in range(HORIZON)]
    rows = []

    for seg, skus in by_seg.items():
        model_name = final_routing.get(seg, "seasonal_naive")
        ModelClass = MODEL_REGISTRY.get(model_name)
        if ModelClass is None:
            from forecasting.models.baseline import SeasonalNaive
            ModelClass = SeasonalNaive

        series = {
            str(sku): train[train[config.COL_SKU_ID]==sku][config.COL_SALES].values
            for sku in skus
            if len(train[train[config.COL_SKU_ID]==sku]) > 0
        }
        if not series:
            continue

        try:
            m = ModelClass(q_levels=q)
            if model_name == "cluster_lgbm":
                m = ModelClass(q_levels=q, lgbm_params=lgbm_params)
            m.fit_series(series)
            result = m.predict(np.empty(0), HORIZON)
            uid_order = sorted(series.keys())
            for j, uid in enumerate(uid_order):
                if j >= result.n_sku:
                    continue
                sku = int(uid)
                for h in range(HORIZON):
                    rows.append({
                        "sku_id":        sku,
                        "segment":       seg,
                        "model":         model_name,
                        "horizon_week":  h + 1,
                        "forecast_date": horizon_dates[h].date(),
                        "p10": round(float(result.quantiles[j, h, p10_i]), 2),
                        "p50": round(float(result.quantiles[j, h, p50_i]), 2),
                        "p90": round(float(result.quantiles[j, h, p90_i]), 2),
                    })
        except Exception as e:
            print(f"  WARNING: {seg}/{model_name} failed: {e}")

    df = pd.DataFrame(rows)
    print(f"  Rows: {len(df)}  ({len(all_skus)} SKUs x {HORIZON} weeks)")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("FONTANA CANDLE  --  SEQUENTIAL LEARNING CV + FORECAST")
    print("=" * 65)

    dense, full_feats, lc, segs = load_pipeline_data()
    fold_origins = get_fold_origins(dense)

    # Starting state — V2 8-class routing informed by fold 1-4 evidence
    lgbm_params = {
        "objective": "quantile",
        "metric": "quantile",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 63,       # start higher: fold 3+4 evidence says more leaves better
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "random_state": 42,
    }
    routing = {
        "smooth":          "theta",
        "smooth_growing":  "theta",          # growing SKUs: Theta captures trend
        "smooth_stable":   "seasonal_naive", # stable SKUs: last year is best
        "erratic":         "patchtst",        # trend extrapolation for erratic ramps
        "promo_driven":    "cluster_lgbm",   # price/promo features critical
        "lumpy":           "seasonal_naive", # fold 2+3 winner: 0.787/0.594
        "intermittent":    "seasonal_naive", # fold 3+4 winner: 0.609/0.672
        "cold_start":      "seasonal_naive", # no seasonality — use mean
        "discontinued":    "zero_forecast",
    }
    drop_features: set = set()
    log: list[str] = ["=== SEQUENTIAL LEARNING LOG V2 — 8-class + hurdle + hierarchy ==="]
    segment_as_cluster = True   # Slide 23 lever 2: use SB segment as LightGBM pool

    all_fold_rows = []

    # ── Sequential folds ─────────────────────────────────────────────────────
    for fold_id in [1, 2, 3, 4]:
        origin = fold_origins[fold_id]
        print(f"\n{'='*65}")
        print(f"FOLD {fold_id}  |  LightGBM num_leaves={lgbm_params['num_leaves']}  "
              f"lr={lgbm_params['learning_rate']}  n_est={lgbm_params['n_estimators']}  "
              f"seg_as_cluster={segment_as_cluster}")
        print(f"Routing: {routing}")
        print(f"Dropped features: {len(drop_features)}")

        result = run_one_fold(
            fold_id=fold_id,
            origin=origin,
            dense=dense,
            full_feats=full_feats,
            segs=segs,
            lgbm_params=lgbm_params,
            routing=routing,
            drop_features=drop_features,
        )

        if not result:
            continue

        # Save fold summary
        fold_rows = []
        for model_name, seg_wapes in result["wape_by_model_seg"].items():
            overall = result["wape_by_model"].get(model_name, float("nan"))
            for seg, w in seg_wapes.items():
                fold_rows.append({
                    "fold": fold_id,
                    "in_selection": fold_id in SELECTION_FOLDS,
                    "origin": origin.date(),
                    "model": model_name,
                    "segment": seg,
                    "wape": round(w, 4),
                    "overall_wape": round(overall, 4),
                })
        fold_df = pd.DataFrame(fold_rows)
        fold_df.to_csv(OUT / f"fold_{fold_id}_summary.csv", index=False)
        all_fold_rows.extend(fold_rows)

        # SKU distribution
        dist_rows = [
            {"fold": fold_id, "segment": seg, **vals}
            for seg, vals in result["sku_distribution"].items()
        ]
        pd.DataFrame(dist_rows).to_csv(OUT / f"fold_{fold_id}_distribution.csv", index=False)

        # Learn and update for next fold
        lgbm_params, routing, drop_features = update_from_fold(
            fold_id, result, lgbm_params, routing, drop_features, log
        )

    # ── Consolidated summary ─────────────────────────────────────────────────
    cv_df = pd.DataFrame(all_fold_rows)
    cv_df.to_csv(OUT / "cv_wape_final.csv", index=False)

    print("\n" + "=" * 65)
    print("SELECTION FOLDS (2+3+4) CONSOLIDATED WAPE")
    print("=" * 65)
    sel = cv_df[cv_df["in_selection"]]
    summary = (
        sel.groupby("model")["overall_wape"]
        .agg(["mean", "min", "max"])
        .sort_values("mean")
    )
    print(summary.round(4).to_string())

    print("\nFinal routing (after sequential learning):")
    for seg, m in routing.items():
        print(f"  {seg:15s}: {m}")

    # ── Final forecast ───────────────────────────────────────────────────────
    forecast_df = make_final_forecast(dense, full_feats, segs, routing, lgbm_params)
    forecast_df.to_csv(OUT / "forecast_26w.csv", index=False)

    # ── Write log ────────────────────────────────────────────────────────────
    (OUT / "sequential_log.txt").write_text("\n".join(log), encoding="utf-8")

    print(f"\nAll outputs -> {OUT}/")
    print("  forecast_26w.csv        final 26-week forecast all SKUs")
    print("  fold_N_summary.csv      per-fold results (N=1..4)")
    print("  cv_wape_final.csv       consolidated fold x model x segment WAPE")
    print("  sequential_log.txt      what changed each fold and why")


if __name__ == "__main__":
    main()
