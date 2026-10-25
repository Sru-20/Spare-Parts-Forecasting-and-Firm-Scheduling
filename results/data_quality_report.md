# Phase 1 Data Quality Report

Generated from the original CSVs via schema-validated readers. Source files were not modified.

## Shapes and join checks

| Dataset/table | Rows | Columns |
|---|---:|---:|
| D1 demand source | 120,861 | 62 |
| D2 scheduling source | 158 | 61 |
| D3 catalogue source | 160 | 29 |
| Modeling table (D1 enriched by D3) | 120,861 | 73 |
| Scheduling table (D2 enriched by D3) | 158 | 84 |

Join row counts are preserved: modeling 120,861/120,861; scheduling 158/158. D1→D3 uses `many_to_one`; D2→D3 uses `one_to_one`; shared overlapping attributes are compared and must agree.

## Temporal split and labels

| split_role | rows  | start      | end        | missing_targets |
| ---------- | ----- | ---------- | ---------- | --------------- |
| final_test | 15804 | 2026-03-30 | 2026-09-21 | 11              |
| future     | 8512  | 2026-09-28 | 2026-12-28 | 8512            |
| train_val  | 66388 | 2024-01-08 | 2026-03-23 | 43              |
| warmup     | 30157 | 2023-01-02 | 2024-01-01 | 18              |

- Missing historical labels excluded: 72.
- Future rows: 8,512; future rows with a nonblank target: 0.
- Genuine observed zero-demand labels: 54,511 of 112,277 non-missing historical observations.
- Eligible training rows after split/missing filters: 66,345; roles present: ['train_val'].
- Final-test rows are not selected for fitting; future rows are not selected for fitting.

## Target decision

**Use raw `demand_qty` as the training target and score against raw `demand_qty`.** It represents demand to fulfill and preserves high-demand observations. The generator computes `demand_qty_capped` using a per-series cap estimated over warm-up plus the entire train-validation span, then applies that cap to earlier observations as well. The supplied capped column therefore is not an origin-fitted target transform for rolling origins. 1,403 observed historical rows differ between raw and capped labels; capped fitting is not the default. Genuine zeros remain zeros; missing rows remain excluded, not imputed as zero.

## Keys, negatives and outliers

- Duplicate keys (must be zero): {'demand': 0, 'scheduling': 0, 'catalogue': 0}.
- Referential integrity: PASS; catalogue/D1/D2 part counts are 160/160/158; D3-only obsolete parts: ['P0046', 'P0087'].
- Weekly continuity: PASS; non-seven-day gaps found: 0.
- Negative numeric values across source tables: 0.
- IQR outliers are a diagnostic, not deletion criteria; sparse/lumpy demand and large legitimate orders can be flagged.

Top IQR counts:

| column              | iqr_lower_fence    | iqr_upper_fence    | iqr_outlier_rows |
| ------------------- | ------------------ | ------------------ | ---------------- |
| out_lost_sales_qty  | 0.0                | 0.0                | 16978            |
| out_backorder_qty   | 0.0                | 0.0                | 16529            |
| adi_trainval        | -4.013000000000001 | 9.523              | 15466            |
| out_consumption_qty | -19.5              | 32.5               | 15023            |
| promotion_intensity | 0.0                | 0.0                | 13850            |
| demand_qty          | -27.0              | 45.0               | 13293            |
| demand_qty_capped   | -27.0              | 45.0               | 13194            |
| vehicle_parc        | 74531.0            | 747731.0           | 11340            |
| demand_lag52        | -28.5              | 47.5               | 10001            |
| unit_cost           | -6008.619999999999 | 11287.619999999999 | 7065             |

Negative values by column:

_None._

## Missing values

Missingness is preserved as nullable values; target filtering is explicit. Detailed per-column type/missing counts and examples are in `reports/data_dictionary.csv`.

| dataset   | column                    | pandas_dtype   | missing_count | missing_pct |
| --------- | ------------------------- | -------------- | ------------- | ----------- |
| catalogue | substitution_group        | string         | 150           | 93.75       |
| catalogue | launch_week               | datetime64[ns] | 148           | 92.5        |
| catalogue | shelf_life_days           | Float64        | 129           | 80.625      |
| demand    | substitution_group        | string         | 112501        | 93.083      |
| demand    | festival_name             | string         | 94236         | 77.9706     |
| demand    | festival_intensity        | string         | 94236         | 77.9706     |
| demand    | demand_lag52              | Float64        | 31830         | 26.336      |
| demand    | weeks_since_last_shutdown | Float64        | 25605         | 21.1855     |
| demand    | demand_qty                | Int64          | 8584          | 7.1024      |
| demand    | demand_qty_capped         | Int64          | 8584          | 7.1024      |
| demand    | out_consumption_qty       | Int64          | 8584          | 7.1024      |
| demand    | out_backorder_qty         | Int64          | 8584          | 7.1024      |
| demand    | out_lost_sales_qty        | Int64          | 8584          | 7.1024      |
| demand    | out_stockout_flag         | Int64          | 8584          | 7.1024      |
| demand    | out_spike_flag            | Int64          | 8584          | 7.1024      |

## Integrity and leakage

- Every source schema is checked against its verified column set before parsing with explicit nullable string/integer/float and datetime dtypes.
- Referential integrity and join row preservation passed; shared D1/D3 and D2/D3 attributes must match.
- No future or final-test rows enter `eligible_training_rows`; only nonmissing `train_val` labels are returned.
- `reports/feature_availability_audit.csv` lists each source column, origin-time status, whether it is an approved default forecast feature, and rationale.
- Feature-origin audit totals: {'Conditional': 74, 'Yes': 54, 'No': 24}; approved default-feature entries across source schemas: 52. Conditional drivers and fields not available at all historical origins are not default features.
- Generator answer keys, `out_*` outcomes, train-validation-derived class summaries, targets and split metadata are barred from default features. Origin-dependent lags and future drivers require origin-time recomputation or proof of advance availability.
- Missing target count, genuine zeros and future-label absence are checked in tests.

## Artifacts

- `artifacts/modelling_table.parquet`
- `artifacts/scheduling_inputs.parquet`
- `reports/data_dictionary.csv`
- `reports/feature_availability_audit.csv`
- `results/demand_iqr_outliers.csv`
- `results/negative_values.csv`
