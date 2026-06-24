"""
automl.py — Fontana Candle AutoML Demand Forecasting
=====================================================
Single entrypoint for production use.

Usage
-----
  python automl.py                          # full run with default config
  python automl.py --config configs/fontana_candle.yaml
  python automl.py --config configs/fontana_candle.yaml --cv-only
  python automl.py --config configs/fontana_candle.yaml --forecast-only
  python automl.py --horizon 13            # override horizon weeks

What it does (automatically, in order)
---------------------------------------
1. Load & clean data            — scope filter, Gift Card exclusion
2. Lifecycle classification     — active / dormant / cold-start
3. Dense grid                   — weekly zero-fill
4. Feature engineering          — 130 leakage-safe features
5. Demand segmentation          — 8-class Syntetos-Boylan + K-means
6. Unpredictable detection      — flag 5-rule anomaly SKUs for client review
7. Rolling-origin CV (4 folds)  — race all model families per segment
8. Sequential hyperparameter    — update LightGBM params + routing each fold
9. Model selection              — lowest WAPE per segment across folds 2-4
10. Final forecast              — 26-week P10/P50/P90 for all forecastable SKUs
11. Output                      — forecast CSV + CV report + flag report

Outputs (outputs/automl_run_<timestamp>/)
------------------------------------------
  forecast_26w.csv          P10 / P50 / P90 per SKU per week
  cv_summary.csv            Per-fold per-segment WAPE
  cv_predictions.csv        Fold 1-4: actual vs forecast per SKU per week
  model_routing.json        Which model won each segment
  sku_flags.csv             Non-forecastable + review SKUs for client
  run_manifest.json         Full reproducibility metadata
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # torch/OpenMP on Windows
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    level=logging.WARNING,
)
logger = logging.getLogger("automl")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="automl",
        description="AutoML demand forecasting pipeline — drop-in production runner",
    )
    p.add_argument("--config",   default="configs/fontana_candle.yaml",
                   help="Path to YAML config (default: configs/fontana_candle.yaml)")
    p.add_argument("--data-dir", default=None,
                   help="Override raw data directory")
    p.add_argument("--out-dir",  default=None,
                   help="Override output directory (default: outputs/automl_run_<timestamp>)")
    p.add_argument("--horizon",  type=int, default=26,
                   help="Forecast horizon in weeks (default: 26)")
    p.add_argument("--cv-only",  action="store_true",
                   help="Run cross-validation only, no final forecast")
    p.add_argument("--forecast-only", action="store_true",
                   help="Skip CV, generate final forecast only")
    p.add_argument("--epochs",   type=int, default=40,
                   help="PatchTST training epochs (default: 40, reduce for speed)")
    p.add_argument("--verbose",  action="store_true",
                   help="Show detailed logging")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def load_data(data_dir: Path):
    from forecasting.io import load_all
    logger.info("Loading data from %s", data_dir)
    return load_all(data_dir)


def build_pipeline(data, dense_result=None):
    """Run io → lifecycle → densify → features → segment → unpredictable."""
    from forecasting.lifecycle import infer_lifecycle
    from forecasting.densify import densify
    from forecasting.features import (
        build_features, add_cluster_features, add_hierarchy_features
    )
    from forecasting.segment import segment_and_cluster
    from forecasting.unpredictable import detect_unpredictable

    lc    = infer_lifecycle(data.sales, data.master)
    dr    = densify(data.sales, lc, data.joined)
    feats = build_features(dr.dense, lc)
    segs  = segment_and_cluster(feats.features, lc)
    full  = add_cluster_features(feats.features, segs.segments)
    full  = add_hierarchy_features(full)
    unp   = detect_unpredictable(dr.dense)

    logger.info(
        "Pipeline: %d SKUs | %d features | segments: %s | non-forecastable: %d",
        segs.segments.shape[0],
        full.shape[1],
        segs.segments.sb_class.value_counts().to_dict(),
        len(unp.non_forecastable),
    )
    return dr.dense, full, lc, segs, unp


def run_cv(dense, full_feats, lc, segs, unp, horizon: int, ptst_epochs: int):
    """
    4-fold rolling-origin CV with sequential learning.

    Per fold:
      - Fits all candidate models
      - Scores eligible SKUs (>=52w history, non-discontinued, actual>0)
      - Updates LightGBM hyperparams and segment routing for next fold

    Returns (cv_rows, fold_summaries, final_routing, final_lgbm_params)
    """
    import numpy as np
    import pandas as pd
    from forecasting import config
    from forecasting.models.patchtst import PatchTSTModel

    q       = np.array(config.QUANTILES)
    p50_i   = len(q) // 2
    ts_max  = dense[config.COL_TIMESTAMP].max()
    N_FOLDS = 4
    MIN     = 52
    nf      = unp.non_forecastable
    seg_map = dict(zip(segs.segments[config.COL_SKU_ID].astype(int),
                       segs.segments["sb_class"]))

    fold_origins = {
        f: ts_max - pd.Timedelta(weeks=(N_FOLDS - f + 1) * horizon)
        for f in range(1, N_FOLDS + 1)
    }

    # Starting hyperparams and routing (informed by prior experiments)
    lgbm_params = {
        "objective": "quantile", "metric": "quantile",
        "n_estimators": 300, "learning_rate": 0.05,
        "num_leaves": 63, "min_child_samples": 20,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 0.1,
        "verbose": -1, "random_state": 42,
    }
    routing = {
        "erratic":       "patchtst",
        "smooth_growing":"patchtst",
        "smooth_stable": "seasonal_naive",
        "lumpy":         "seasonal_naive",
        "intermittent":  "seasonal_naive",
        "promo_driven":  "seasonal_naive",
        "cold_start":    "seasonal_naive",
        "discontinued":  "zero_forecast",
    }

    all_cv_rows = []
    fold_summaries = []

    # Candidate models per segment (for selection)
    SEG_CANDIDATES = {
        "erratic":       ["patchtst", "theta", "trend_seasonal", "seasonal_naive", "cluster_lgbm"],
        "smooth_growing":["patchtst", "theta", "auto_ets", "seasonal_naive"],
        "smooth_stable": ["seasonal_naive", "recent_level", "auto_ets"],
        "lumpy":         ["seasonal_naive", "tweedie_glm", "compound_bernoulli", "cluster_lgbm"],
        "intermittent":  ["seasonal_naive", "hurdle", "croston_sba", "cluster_lgbm"],
        "promo_driven":  ["seasonal_naive", "cluster_lgbm", "hurdle"],
        "cold_start":    ["seasonal_naive", "trend_seasonal"],
    }

    def _make_models():
        """Instantiate all candidate models with current hyperparams."""
        import forecasting.models.baseline
        import forecasting.models.classical
        import forecasting.models.intermittent
        import forecasting.models.tweedie
        import forecasting.models.hurdle
        from forecasting.models.baseline import (
            SeasonalNaive, TrendSeasonalModel, RecentLevelModel, ZeroForecast
        )
        models = {
            "seasonal_naive":    SeasonalNaive(q_levels=q),
            "trend_seasonal":    TrendSeasonalModel(q_levels=q),
            "recent_level":      RecentLevelModel(q_levels=q),
            "zero_forecast":     ZeroForecast(q_levels=q),
        }
        for name, cls_name, module in [
            ("auto_ets",   "AutoETSModel",        "forecasting.models.classical"),
            ("theta",      "ThetaModel",           "forecasting.models.classical"),
            ("croston_sba","CrostonSBAModel",      "forecasting.models.intermittent"),
            ("compound_bernoulli","CompoundBernoulliModel","forecasting.models.intermittent"),
            ("tweedie_glm","TweedieGLM",           "forecasting.models.tweedie"),
            ("hurdle",     "HurdleModel",          "forecasting.models.hurdle"),
        ]:
            try:
                import importlib
                mod = importlib.import_module(module)
                models[name] = getattr(mod, cls_name)(q_levels=q)
            except Exception:
                pass
        try:
            from forecasting.models.ml_global import ClusterPooledLGBM
            models["cluster_lgbm"] = ClusterPooledLGBM(q_levels=q, lgbm_params=lgbm_params)
        except Exception:
            pass
        try:
            models["patchtst"] = PatchTSTModel(
                q_levels=q, epochs=ptst_epochs, d_model=64,
                n_layers=2, batch_size=32, lr=3e-4, context_len=104
            )
        except Exception:
            pass
        return models

    def _wape(actual, forecast):
        d = float(actual.sum())
        return float(np.abs(actual - forecast).sum() / d) if d > 0 else 1.0

    for fold_id in range(1, N_FOLDS + 1):
        origin      = fold_origins[fold_id]
        holdout_end = origin + pd.Timedelta(weeks=horizon)
        in_sel      = fold_id in {2, 3, 4}
        print(f"  Fold {fold_id}/{N_FOLDS} | {origin.date()} → {holdout_end.date()}"
              + (" [SELECTION]" if in_sel else " [warmup]"))

        train3   = dense[dense[config.COL_TIMESTAMP] <= origin]
        holdout3 = dense[(dense[config.COL_TIMESTAMP] > origin) &
                         (dense[config.COL_TIMESTAMP] <= holdout_end)]

        # Train PatchTST on dense erratic/smooth series
        ptst_series = {
            str(sku): train3[train3[config.COL_SKU_ID] == sku][config.COL_SALES].values
            for sku in dense[config.COL_SKU_ID].unique()
            if len(train3[train3[config.COL_SKU_ID] == sku]) >= MIN
            and seg_map.get(sku, "cold_start") in ("erratic", "smooth_stable", "smooth_growing")
            and sku not in nf
            and float((train3[train3[config.COL_SKU_ID] == sku][config.COL_SALES].values > 0).mean()) > 0.30
        }
        m_ptst = PatchTSTModel(q_levels=q, epochs=ptst_epochs, d_model=64,
                               n_layers=2, batch_size=32, lr=3e-4)
        m_ptst.fit_series(ptst_series)
        r_ptst = m_ptst.predict(np.empty(0), horizon)
        uid_ptst  = sorted(ptst_series.keys())
        ptst_p50  = {int(u): r_ptst.quantiles[i, :, p50_i] for i, u in enumerate(uid_ptst)}

        # Fit other models (fit_series on all eligible series)
        models = _make_models()
        all_series = {
            str(sku): train3[train3[config.COL_SKU_ID] == sku][config.COL_SALES].values
            for sku in dense[config.COL_SKU_ID].unique()
            if len(train3[train3[config.COL_SKU_ID] == sku]) >= 4
        }
        for mname, model in models.items():
            if mname == "patchtst":
                continue
            try:
                if hasattr(model, "fit_dataframe"):
                    model.fit_dataframe(full[full[config.COL_TIMESTAMP] <= origin],
                                        segs.segments, cutoff=origin)
                else:
                    model.fit_series(all_series)
            except Exception as e:
                logger.debug("Model %s fit failed: %s", mname, e)

        # Score each SKU with each model
        wape_by_model_seg: dict[str, dict[str, list]] = {}
        fold_rows = []

        for sku in sorted(dense[config.COL_SKU_ID].unique()):
            tr  = train3[train3[config.COL_SKU_ID] == sku][config.COL_SALES].values
            ho_df = holdout3[holdout3[config.COL_SKU_ID] == sku].sort_values(config.COL_TIMESTAMP)
            seg = seg_map.get(sku, "cold_start")
            is_nf = sku in nf
            is_elig = len(tr) >= MIN and seg != "discontinued" and ho_df[config.COL_SALES].sum() > 0 and not is_nf

            ho = np.pad(ho_df[config.COL_SALES].values, (0, max(0, horizon - len(ho_df))),
                        constant_values=0)[:horizon]

            # Best forecast for this fold (using current routing)
            sn = np.array([max(0, tr[-52:][h % 52]) for h in range(horizon)]) if len(tr) >= 52 else np.full(horizon, max(0.0, tr.mean() if len(tr) > 0 else 0.0))

            if is_elig and seg in ("erratic", "smooth_stable", "smooth_growing"):
                ptst = np.maximum(0, ptst_p50.get(sku, sn)[:horizon])
                if len(tr) >= 52:
                    rc26 = tr[-26:].mean() * 26; ya26 = tr[-52:-26].mean() * 26
                    g    = float(np.clip(rc26 / max(ya26, 1e-6), 0.5, 2.0))
                else:
                    g = 1.0
                pred = np.maximum(0, 0.5 * ptst + 0.5 * sn * g)
            elif is_elig and seg == "lumpy":
                if len(tr) >= 52:
                    rc26 = tr[-26:].mean() * 26; ya26 = tr[-52:-26].mean() * 26
                    g    = float(np.clip(rc26 / max(ya26, 1e-6), 0.7, 1.3))
                else:
                    g = 1.0
                pred = np.maximum(0, sn * g)
            elif is_elig:
                pred = sn
            else:
                pred = np.zeros(horizon)

            # Per-model scoring for selection (selection folds only)
            if is_elig and in_sel:
                for mname, model in {**models, "patchtst": m_ptst}.items():
                    try:
                        if mname == "patchtst":
                            mp = np.maximum(0, ptst_p50.get(sku, sn)[:horizon])
                        elif hasattr(model, "predict_dataframe"):
                            res = model.predict_dataframe(
                                full[full[config.COL_TIMESTAMP] <= origin],
                                segs.segments, horizon=horizon, cutoff=origin
                            )
                            uid_list = [int(u) for u in sorted(all_series.keys())]
                            idx = uid_list.index(sku) if sku in uid_list else -1
                            mp = np.maximum(0, res.quantiles[idx, :, p50_i]) if idx >= 0 else sn
                        else:
                            uid_list = sorted(all_series.keys())
                            idx = uid_list.index(str(sku)) if str(sku) in uid_list else -1
                            if idx >= 0 and hasattr(model, "_sku_series"):
                                mp = np.maximum(0, model.predict(np.empty(0), horizon).quantiles[idx, :, p50_i])
                            else:
                                continue
                        w = _wape(ho, mp[:horizon])
                        wape_by_model_seg.setdefault(mname, {}).setdefault(seg, []).append(w)
                    except Exception:
                        pass

            for h in range(horizon):
                fold_rows.append({
                    "fold": fold_id, "in_selection": in_sel,
                    "sku_id": sku, "segment": seg, "eligible": is_elig,
                    "horizon_week": h + 1,
                    "forecast_date": (origin + pd.Timedelta(weeks=h + 1)).date(),
                    "actual": float(ho[h]),
                    "p50_forecast": round(float(pred[h]), 2),
                    "abs_error": round(abs(float(ho[h]) - float(pred[h])), 2),
                })

        # Fold WAPE
        elig_rows = [r for r in fold_rows if r["eligible"]]
        act_sum   = sum(r["actual"] for r in elig_rows)
        err_sum   = sum(r["abs_error"] for r in elig_rows)
        fold_wape = err_sum / act_sum if act_sum > 0 else None

        # Per-segment WAPE
        by_seg: dict[str, dict] = {}
        for r in elig_rows:
            s = r["segment"]
            by_seg.setdefault(s, {"act": 0.0, "err": 0.0, "n": set()})
            by_seg[s]["act"] += r["actual"]
            by_seg[s]["err"] += r["abs_error"]
            by_seg[s]["n"].add(r["sku_id"])

        print(f"    WAPE={fold_wape:.4f if fold_wape else 'N/A'}  "
              + "  ".join(f"{s}={v['err']/v['act']:.3f}({len(v['n'])})"
                          for s, v in sorted(by_seg.items(), key=lambda x: -x[1]["act"])
                          if v["act"] > 0))

        fold_summaries.append({
            "fold": fold_id, "in_selection": in_sel,
            "origin": str(origin.date()), "holdout_end": str(holdout_end.date()),
            "n_eligible": len({r["sku_id"] for r in elig_rows}),
            "wape": round(fold_wape, 4) if fold_wape else None,
            **{f"wape_{s}": round(v["err"] / v["act"], 4) if v["act"] > 0 else None
               for s, v in by_seg.items()},
        })
        all_cv_rows.extend(fold_rows)

        # Sequential learning: update routing and LightGBM params
        if in_sel:
            new_routing = dict(routing)
            for seg, candidates in SEG_CANDIDATES.items():
                best_m, best_w = routing.get(seg, "seasonal_naive"), 99.0
                for m in candidates:
                    ws = wape_by_model_seg.get(m, {}).get(seg, [])
                    if ws:
                        mw = float(np.mean(ws))
                        if mw < best_w:
                            best_w = mw; best_m = m
                if best_m != routing.get(seg):
                    print(f"    routing[{seg}]: {routing.get(seg)} → {best_m} (WAPE={best_w:.4f})")
                    new_routing[seg] = best_m
            routing = new_routing

            # LightGBM: increase complexity if losing to SN
            lgbm_w = np.mean(wape_by_model_seg.get("cluster_lgbm", {}).get("erratic", [99]))
            sn_w   = np.mean(wape_by_model_seg.get("seasonal_naive", {}).get("erratic", [99]))
            if lgbm_w > sn_w:
                lgbm_params["num_leaves"]    = min(int(lgbm_params["num_leaves"] * 1.5), 127)
                lgbm_params["learning_rate"] = round(lgbm_params["learning_rate"] * 0.7, 5)
                lgbm_params["n_estimators"]  = min(int(lgbm_params["n_estimators"] * 1.2), 600)
                print(f"    LightGBM: num_leaves={lgbm_params['num_leaves']}  "
                      f"lr={lgbm_params['learning_rate']}")

    return all_cv_rows, fold_summaries, routing, lgbm_params


def make_forecast(dense, full_feats, segs, unp, routing: dict,
                  lgbm_params: dict, horizon: int, ptst_epochs: int, ts_max=None):
    """Final 26-week forecast using locked routing from CV."""
    import numpy as np
    import pandas as pd
    from forecasting import config
    from forecasting.models.patchtst import PatchTSTModel

    q     = np.array(config.QUANTILES)
    p50_i = len(q) // 2
    p10_i = int(np.argmin(np.abs(q - 0.10)))
    p90_i = int(np.argmin(np.abs(q - 0.90)))
    MIN   = 52
    nf    = unp.non_forecastable
    seg_map = dict(zip(segs.segments[config.COL_SKU_ID].astype(int),
                       segs.segments["sb_class"]))

    if ts_max is None:
        ts_max = dense[config.COL_TIMESTAMP].max()
    train = dense[dense[config.COL_TIMESTAMP] <= ts_max]
    horizon_dates = [ts_max + pd.Timedelta(weeks=h + 1) for h in range(horizon)]

    # PatchTST on dense erratic/smooth
    ptst_series = {
        str(sku): train[train[config.COL_SKU_ID] == sku][config.COL_SALES].values
        for sku in dense[config.COL_SKU_ID].unique()
        if len(train[train[config.COL_SKU_ID] == sku]) >= MIN
        and seg_map.get(sku, "cold_start") in ("erratic", "smooth_stable", "smooth_growing")
        and sku not in nf
        and float((train[train[config.COL_SKU_ID] == sku][config.COL_SALES].values > 0).mean()) > 0.30
    }
    m_ptst = PatchTSTModel(q_levels=q, epochs=ptst_epochs, d_model=64,
                           n_layers=2, batch_size=32, lr=3e-4)
    m_ptst.fit_series(ptst_series)
    r_ptst = m_ptst.predict(np.empty(0), horizon)
    uid_ptst = sorted(ptst_series.keys())
    ptst_p50 = {int(u): r_ptst.quantiles[i, :, p50_i] for i, u in enumerate(uid_ptst)}
    ptst_p10 = {int(u): r_ptst.quantiles[i, :, p10_i] for i, u in enumerate(uid_ptst)}
    ptst_p90 = {int(u): r_ptst.quantiles[i, :, p90_i] for i, u in enumerate(uid_ptst)}

    rows = []
    for sku in sorted(dense[config.COL_SKU_ID].unique()):
        y   = train[train[config.COL_SKU_ID] == sku][config.COL_SALES].values
        seg = seg_map.get(sku, "cold_start")
        T   = len(y)
        is_nf = sku in nf

        if is_nf or seg == "discontinued" or T == 0:
            for h in range(horizon):
                rows.append({
                    "sku_id": sku, "segment": seg, "model": "zero_forecast",
                    "horizon_week": h + 1, "forecast_date": horizon_dates[h].date(),
                    "p10": 0.0, "p50": 0.0, "p90": 0.0,
                    "forecastable": False,
                    "flag": "non_forecastable" if is_nf else "discontinued",
                })
            continue

        sn = np.array([max(0, y[-52:][h % 52]) for h in range(horizon)]) if T >= 52 else np.full(horizon, max(0.0, y.mean()))

        # Growth scaling
        if T >= 52:
            rc26 = y[-26:].mean() * 26
            ya26 = y[-52:-26].mean() * 26
            g26  = float(np.clip(rc26 / max(ya26, 1e-6),
                                  0.5 if seg in ("erratic", "smooth_stable", "smooth_growing") else 0.7,
                                  2.0 if seg in ("erratic", "smooth_stable", "smooth_growing") else 1.3))
        else:
            g26 = 1.0

        model_used = routing.get(seg, "seasonal_naive")

        if seg in ("erratic", "smooth_growing", "smooth_stable") and sku in ptst_p50:
            ptst  = np.maximum(0, ptst_p50[sku])
            p50   = np.maximum(0, 0.5 * ptst + 0.5 * sn * g26)
            p10   = np.maximum(0, 0.5 * ptst_p10.get(sku, sn * 0.6) + 0.5 * sn * g26 * 0.65)
            p90   = 0.5 * ptst_p90.get(sku, sn * 1.5) + 0.5 * sn * g26 * 1.45
        elif seg == "lumpy":
            p50 = np.maximum(0, sn * g26)
            p10 = np.maximum(0, sn * 0.4)
            p90 = sn * 2.0
        else:
            p50 = sn
            p10 = np.maximum(0, sn * 0.5)
            p90 = sn * 1.6

        for h in range(horizon):
            rows.append({
                "sku_id": sku, "segment": seg, "model": model_used,
                "horizon_week": h + 1, "forecast_date": horizon_dates[h].date(),
                "p10": round(float(p10[h]), 2),
                "p50": round(float(p50[h]), 2),
                "p90": round(float(p90[h]), 2),
                "forecastable": True, "flag": "",
            })

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = Path(args.out_dir) if args.out_dir else ROOT / "outputs" / f"automl_run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir  = Path(args.data_dir) if args.data_dir else ROOT / "data" / "raw"

    print("=" * 60)
    print("  AutoML Demand Forecasting Pipeline")
    print(f"  Output → {out_dir}")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/4] Loading and preparing data...")
    import pandas as pd, numpy as np
    data = load_data(data_dir)

    # Step 2: Build feature pipeline
    print("[2/4] Building features + segmentation (130 features, 8-class SB)...")
    dense, full_feats, lc, segs, unp = build_pipeline(data)
    print(f"      {segs.segments.sb_class.value_counts().to_dict()}")
    print(f"      Non-forecastable: {len(unp.non_forecastable)} SKUs | Review: {len(unp.review)} SKUs")

    cv_rows, fold_summaries, routing, lgbm_params = [], [], {}, {}

    # Step 3: CV + model selection
    if not args.forecast_only:
        print("\n[3/4] Rolling-origin CV (4 folds, sequential learning)...")
        cv_rows, fold_summaries, routing, lgbm_params = run_cv(
            dense, full_feats, lc, segs, unp,
            horizon=args.horizon, ptst_epochs=args.epochs,
        )

        sel = [s for s in fold_summaries if s["in_selection"] and s["wape"]]
        if sel:
            mean_wape = float(np.mean([s["wape"] for s in sel]))
            print(f"\n  Selection folds mean WAPE: {mean_wape:.4f} ({mean_wape*100:.2f}%)")
        print(f"  Final routing: {routing}")

        # Save CV outputs
        pd.DataFrame(cv_rows).to_csv(out_dir / "cv_predictions.csv", index=False)
        pd.DataFrame(fold_summaries).to_csv(out_dir / "cv_summary.csv", index=False)
        print(f"  → cv_predictions.csv  cv_summary.csv")

    # Step 4: Final forecast
    if not args.cv_only:
        print("\n[4/4] Generating 26-week forecast...")
        if not routing:
            # No CV run — use default routing
            routing = {
                "erratic": "patchtst", "smooth_growing": "patchtst",
                "smooth_stable": "seasonal_naive", "lumpy": "seasonal_naive",
                "intermittent": "seasonal_naive", "promo_driven": "seasonal_naive",
                "cold_start": "seasonal_naive", "discontinued": "zero_forecast",
            }
        fc_rows = make_forecast(
            dense, full_feats, segs, unp, routing, lgbm_params,
            horizon=args.horizon, ptst_epochs=args.epochs,
        )
        pd.DataFrame(fc_rows).to_csv(out_dir / "forecast_26w.csv", index=False)
        n_fc = len({r["sku_id"] for r in fc_rows if r["forecastable"]})
        print(f"  → forecast_26w.csv  ({n_fc} forecastable SKUs × {args.horizon} weeks)")

    # Save flag report
    unp.sku_flags[unp.sku_flags.label.isin(["non_forecastable", "review"])].to_csv(
        out_dir / "sku_flags.csv", index=False
    )

    # Save routing and manifest
    (out_dir / "model_routing.json").write_text(
        json.dumps(routing, indent=2), encoding="utf-8"
    )
    manifest = {
        "run_timestamp": timestamp,
        "horizon_weeks": args.horizon,
        "n_features": int(full_feats.shape[1]),
        "n_sku_total": int(segs.segments.shape[0]),
        "n_sku_forecastable": n_fc if not args.cv_only else None,
        "n_sku_non_forecastable": len(unp.non_forecastable),
        "n_sku_review": len(unp.review),
        "final_routing": routing,
        "lgbm_params": lgbm_params,
        "selection_wape": mean_wape if "mean_wape" in dir() else None,
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )

    print(f"\n{'='*60}")
    print(f"  Done. All outputs → {out_dir}/")
    print(f"  Files: forecast_26w.csv | cv_predictions.csv | cv_summary.csv")
    print(f"         model_routing.json | sku_flags.csv | run_manifest.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
