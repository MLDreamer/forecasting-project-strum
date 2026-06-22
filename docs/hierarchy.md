# Hierarchy

Forecasts are produced at the bottom (variant) level and reconciled upward so
any roll-up is coherent and probabilistically honest.

## Levels (in the summing matrix)

| Level | Node(s) | Count |
|---|---|---|
| L0 | `total` (portfolio) | 1 |
| L1 | `segment` | 5 — smooth, erratic, lumpy, intermittent, cold_start |
| L2 | `variant` (`item_id`) | N (bottom) |

The summing matrix `S` has shape `(1 + 5 + N, N)`, is binary, and each upper-row
sum equals that node's bottom-member count.

## Additional groupings (app filters, not in S)

- `status`: active / draft / unknown
- `revenue_tier`: mega / high / mid / low / micro

## Reconciliation

- **Bottom-up (default):** sum P50 bottom→up. For quantiles, draw B=1000
  bootstrap paths per SKU from its quantile forecast, sum across SKUs per week,
  then take node-level quantiles — so intervals are wider at the top (uncertainty
  does not add linearly).
- **MinT-shrink (optional):** `hierarchicalforecast.MinTrace(method='mint_shrink')`.
  Available but not default — needs a positive-definite covariance, which can
  fail on sparse intermittent data.

## App roll-up

- Selection matching a pre-computed node → use `forecast_hierarchy.parquet`.
- Ad-hoc multi-segment selection → on-the-fly bootstrap aggregation (B=500),
  sampling each SKU's piecewise-linear quantile CDF.

## Coherence checks (Phase 6 / 15)

- Sum of bottom P50 at week T equals top P50 at week T (within float tolerance).
- Portfolio P90 < sum of bottom P90s (the point of probabilistic reconciliation).
