"""
run_final_forecast.py
=====================
Final 26-week forecast using best-of-all-evidence routing.
Outputs forecast_26w.csv for all 220 SKUs.

Best routing from sequential CV evidence:
  erratic      -> PatchTST (fold3: 0.733 vs SN 0.783) + SN blend
  lumpy        -> SeasonalNaive (fold3: 0.594, fold4: 0.766)
  intermittent -> SeasonalNaive (fold3: 0.609, fold4: 0.672)
  smooth_stable-> SeasonalNaive with YoY growth
  promo_driven -> SeasonalNaive (NOT PatchTST - over-extrapolates)
  cold_start   -> SeasonalNaive mean fallback
  discontinued -> ZeroForecast

Non-forecastable SKUs (9): flagged with zero forecast + client note
"""
import os; os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

OUT = ROOT / "outputs" / "final_forecast"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    from forecasting.io import load_all
    from forecasting.lifecycle import infer_lifecycle
    from forecasting.densify import densify
    from forecasting.features import build_features, add_cluster_features, add_hierarchy_features
    from forecasting.segment import segment_and_cluster
    from forecasting.unpredictable import detect_unpredictable
    from forecasting import config
    from forecasting.models.patchtst import PatchTSTModel

    print("[1] Loading pipeline...")
    data  = load_all(ROOT / "data/raw")
    lc    = infer_lifecycle(data.sales, data.master)
    dr    = densify(data.sales, lc, data.joined)
    feats = build_features(dr.dense, lc)
    segs  = segment_and_cluster(feats.features, lc)
    full  = add_cluster_features(feats.features, segs.segments)
    full  = add_hierarchy_features(full)
    dense = dr.dense

    print(f"    {full.shape[1]} features | {segs.segments.shape[0]} SKUs")
    print(f"    Segments: {segs.segments.sb_class.value_counts().to_dict()}")

    seg_map = dict(zip(segs.segments[config.COL_SKU_ID].astype(int),
                       segs.segments["sb_class"]))
    unp = detect_unpredictable(dr.dense)
    nf  = unp.non_forecastable
    print(f"    Non-forecastable SKUs: {len(nf)} (excluded, flagged for client)")

    ts_max = dense[config.COL_TIMESTAMP].max()
    train  = dense[dense[config.COL_TIMESTAMP] <= ts_max]
    q      = np.array(config.QUANTILES)
    p50_i  = len(q) // 2
    p10_i  = int(np.argmin(np.abs(q - 0.10)))
    p90_i  = int(np.argmin(np.abs(q - 0.90)))
    HORIZON = 26

    horizon_dates = [ts_max + pd.Timedelta(weeks=h+1) for h in range(HORIZON)]

    # Train PatchTST on erratic/smooth SKUs
    print("[2] Training PatchTST on erratic/smooth SKUs...")
    ptst_series = {}
    for sku in dense[config.COL_SKU_ID].unique():
        seg = seg_map.get(sku, "cold_start")
        y   = train[train[config.COL_SKU_ID] == sku][config.COL_SALES].values
        if len(y) >= 52 and seg in ("erratic", "smooth_stable", "smooth_growing"):
            ptst_series[str(sku)] = y

    m_ptst = PatchTSTModel(q_levels=q, epochs=50, d_model=64, n_layers=2,
                           batch_size=32, lr=3e-4, context_len=104)
    m_ptst.fit_series(ptst_series)
    r_ptst = m_ptst.predict(np.empty(0), HORIZON)
    uid_ptst = sorted(ptst_series.keys())
    ptst_p50 = {int(u): r_ptst.quantiles[i, :, p50_i] for i, u in enumerate(uid_ptst)}
    ptst_p10 = {int(u): r_ptst.quantiles[i, :, p10_i] for i, u in enumerate(uid_ptst)}
    ptst_p90 = {int(u): r_ptst.quantiles[i, :, p90_i] for i, u in enumerate(uid_ptst)}
    print(f"    PatchTST trained on {len(ptst_series)} SKUs")

    # Generate forecasts for all SKUs
    print("[3] Generating forecasts...")
    rows = []
    for sku in sorted(dense[config.COL_SKU_ID].unique()):
        y   = train[train[config.COL_SKU_ID] == sku][config.COL_SALES].values
        seg = seg_map.get(sku, "cold_start")
        T   = len(y)

        is_nf = sku in nf

        if is_nf or seg == "discontinued" or T == 0:
            for h in range(HORIZON):
                rows.append({
                    "sku_id": sku, "segment": seg,
                    "horizon_week": h+1,
                    "forecast_date": horizon_dates[h].date(),
                    "p10": 0.0, "p50": 0.0, "p90": 0.0,
                    "forecastable": False,
                    "note": "non_forecastable" if is_nf else "discontinued",
                })
            continue

        # SeasonalNaive base
        if T >= 52:
            sn = np.array([max(0, y[-52:][h % 52]) for h in range(HORIZON)])
        else:
            sn = np.full(HORIZON, max(0.0, float(y.mean())))

        # YoY growth for smoothing
        if T >= 52:
            rc26 = y[-26:].sum()
            ya26 = y[-52:-26].sum()
            g    = float(np.clip(rc26 / max(ya26, 1.0), 0.4, 3.0))
        else:
            g = 1.0

        if seg in ("erratic", "smooth_growing"):
            # 50% PatchTST + 50% SN (PatchTST captures growth trends)
            ptst = ptst_p50.get(sku, sn)
            p50  = np.maximum(0, 0.5 * ptst + 0.5 * sn)
            ptst_lo = ptst_p10.get(sku, sn * 0.6)
            ptst_hi = ptst_p90.get(sku, sn * 1.5)
            p10  = np.maximum(0, 0.5 * ptst_lo + 0.5 * sn * 0.7)
            p90  = 0.5 * ptst_hi + 0.5 * sn * 1.4

        elif seg == "smooth_stable":
            # SN with slight YoY growth
            p50 = np.maximum(0, sn * min(g, 1.5))
            p10 = np.maximum(0, p50 * 0.7)
            p90 = p50 * 1.4

        elif seg == "lumpy":
            # Pure SeasonalNaive (best for lumpy)
            p50 = sn
            p10 = np.maximum(0, sn * 0.5)
            p90 = sn * 1.8

        elif seg == "intermittent":
            p50 = sn
            p10 = np.maximum(0, sn * 0.4)
            p90 = sn * 1.6

        elif seg == "promo_driven":
            # SN is safest for promo_driven (PatchTST over-extrapolates)
            p50 = sn
            p10 = np.maximum(0, sn * 0.5)
            p90 = sn * 2.0

        else:  # cold_start
            p50 = sn
            p10 = np.maximum(0, sn * 0.3)
            p90 = sn * 2.5

        for h in range(HORIZON):
            rows.append({
                "sku_id": sku, "segment": seg,
                "horizon_week": h+1,
                "forecast_date": horizon_dates[h].date(),
                "p10": round(float(p10[h]), 2),
                "p50": round(float(p50[h]), 2),
                "p90": round(float(p90[h]), 2),
                "forecastable": True,
                "note": "",
            })

    fc_df = pd.DataFrame(rows)
    fc_path = OUT / "forecast_26w.csv"
    fc_df.to_csv(fc_path, index=False)

    # Summary
    forecastable = fc_df[fc_df.forecastable]
    n_sku_fc = forecastable.sku_id.nunique()
    print(f"\n[4] Output: {len(fc_df)} rows ({fc_df.sku_id.nunique()} SKUs x {HORIZON} weeks)")
    print(f"    Forecastable: {n_sku_fc} SKUs | Non-forecastable: {len(nf)} SKUs")
    print(f"    -> {fc_path}")

    # Save non-forecastable report
    nf_report = unp.sku_flags[unp.sku_flags.label.isin(["non_forecastable", "review"])]
    nf_report.to_csv(OUT / "sku_flags.csv", index=False)
    print(f"    -> {OUT}/sku_flags.csv ({len(nf_report)} flagged SKUs)")


if __name__ == "__main__":
    main()
