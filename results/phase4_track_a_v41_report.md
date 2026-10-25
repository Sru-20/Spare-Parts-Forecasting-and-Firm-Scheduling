# Track A selection — SPEC v4.1

**Evidence window:** four held-validation origins only. Final-test and future labels were not scored.  
**Claim scope:** method validated on simulated data; results are not proven factory savings.

## Decision

- Status: champion supported.
- Champion: lightgbm_tweedie::random_1.
- Candidates carried into the Track B forecast-by-scheduler check: lightgbm_tweedie::random_1, lightgbm_quantile::random_1.
- Sensitivity winners: 14 one-at-a-time ±10-point cases; required champion share: 75%.
- The additional critical-bias-overlap-removed scenario is recorded separately in the selection CSV.
- Future output: C:\Users\sruja\OneDrive\Desktop\spare-parts-demand-scheduling\results\forecast_future_14w_v41.csv. It is a recursive train-validation-only projection through the sealed test-date gap, not a directly validated 14-step forecast; the effective recursive step is 40, with 26 unobserved bridge weeks. Refit and evaluate against the final-test window only in Phase 7.

## Required candidate family and scope

SBA versus TSB was selected using pooled lead-time WAPE on the two configured tuning origins, before the four held origins were scored. The comparison is in `track_a_v41_sba_tsb_selection.csv`. Historical average uses the observed train-validation mean available strictly through each origin. Other Phase 3 candidates remain in the forecast artifact and selection matrix; ensembles/routers enter the weighted pool only if they beat the best simple candidate on pooled lead-time WAPE and on at least three held origins.

## Known-at-origin feature decision

The held-validation backtests use date-derived and static fields included in the fold's approved calendar whitelist. Promotions/campaigns, market events/indices, plant shutdown/working-day fields, vehicle parc/age, and substitution groups are excluded because the source does not record publication/availability dates proving they were known at each origin. No `out_*` or label-derived field is used.

## Measured results

- Held-validation forecast rows in the v4.1 artifact: 416,892; lead-time aggregates: 29,196.
- Quantile crossing counts (`P50<=P90`, `P90<=P95`, `P95<=P99`): {'forecast_qty_gt_forecast_p90': 0, 'forecast_p90_gt_forecast_p95': 0, 'forecast_p95_gt_forecast_p99': 0}.
- Lead-time rows using the labelled extension: 624. For lead times over 14 weeks, the final 4 weeks are the “last period”; actual and forecast sums are extended by their respective mean times the extra lead weeks. This is a project assumption.
- Forecast metrics are stored in ratio units in `track_a_v41_metrics.csv`; percentages are display-only. Bootstrap intervals resample part-region series and origins together. Four origins and synthetic data limit inference.
- Screening enforces only beating both seasonal-naive and origin-visible historical-average baselines by point estimate. Bias and calibration screens are reported with intervals and do not reject a candidate when uncertainty crosses the threshold. Bias screens use lead-time aggregation; “critical” means High criticality, and the critical/A error segment is High criticality or ABC A.
- Weighted selection uses rank normalization within the baseline-eligible set. Missing metric scores are zero. Critical/A-class error averages lead-time WAPE and asymmetric-bias ranks; intermittent/lumpy averages lead-time WAPE and weekly MASE ranks; runtime/explainability is split equally. These are explicit scoring conventions.

## Outputs

- `artifacts/track_a_backtest_forecasts_v41.parquet`
- `results/track_a_v41_selection_matrix.csv`
- `results/track_a_v41_weight_sensitivity.csv`
- `results/track_a_v41_metrics.csv`
- `results/track_a_v41_sba_tsb_selection.csv`
- `results/forecast_future_14w_v41.csv`
- Frozen model artifacts under `artifacts/models/`

The Phase 3 artifacts and legacy Phase 4 files were not overwritten. The future output is a projection, not measured performance. Phase 5 must compare the top two forecast candidates on identical demand paths with frozen simulator assumptions; the final test remains sealed.
