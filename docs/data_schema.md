# Data schema

Two input files go in `data/raw/` (auto-detected as `.xlsx`, `.xls`, or `.csv`).
They are never committed.

## `variants_export` (variant master)

393 rows (392 unique after dedup).

| Column | Type | Notes |
|---|---|---|
| `variant_id` | UUID (str) | Platform-internal; **not** used for joining |
| `source_variant_id` | int64 | Shopify variant ID; **the join key** |
| `status` | category | One of `active`, `draft`, `archived` |

## `processed_data_filtered` (weekly sales)

11,291 rows; 229 unique `item_id`.

| Column | Type | Notes |
|---|---|---|
| `item_id` | int64 | Equals `source_variant_id` |
| `timestamp` | date | Week-ending **Sunday**, 2020-12-27 → 2026-05-17 |
| `sales` | float | Units sold (gross — definition unconfirmed; a known limitation) |
| `avg_unit_price` | float | Realized price |
| `discount_pct` | float | 0.0 – 1.0 |

> The sales file is pre-filtered to non-zero weeks, so missing weeks within a
> SKU's active window must be zero-filled (see densification, Phase 3).

## Join & validation rules (Phase 1)

- Drop rows with NaN `source_variant_id` (warn).
- Resolve duplicate `source_variant_id` by keeping the first row (warn).
- Left-join sales onto master on `item_id == source_variant_id`; unmatched →
  `status = 'unknown'`.
- Verify the week boundary is Sunday-ending (warn, don't fail).

Expected post-load counts: variants = 392, sales = 11,291, unknown-status
rows = 493, archived-without-sales = 174.

## Generated artifacts

| File | Stage |
|---|---|
| `data/interim/joined.parquet` | load |
| `data/interim/lifecycle.parquet` | lifecycle |
| `data/interim/dense_weekly.parquet` | densify |
| `data/processed/features.parquet` | features |
| `data/processed/segments.parquet` | segment |
| `data/processed/hierarchy.parquet`, `S_matrix.npz` | hierarchy |
| `outputs/*.parquet`, `outputs/forecast_final.csv` | cv / forecast |
