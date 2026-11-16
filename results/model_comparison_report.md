# SPEC v4.2 A1 — model comparison at the tuning origins

Dataset: synthetic. Results validate method behaviour on simulated data; they are not factory savings.

## Scope and budget

- Tuning origins: 2024-07-01 and 2024-10-07; fixed 14-week horizon.
- Champion: `lightgbm_tweedie::random_1` with the frozen hyperparameters and seed 42, read from the frozen tuning-forecast artifact.
- New candidates: `moving_average_13`, `exponential_smoothing` (ETS(A,N,N) with fitted alpha, bounded 0.01–0.99), and `xgboost` (one of XGBoost/Random Forest required by the amendment; Random Forest already had a Tier 2 run in Phase 3). XGBoost uses the same three seed-42 configurations as the frozen LightGBM search, mapped as: `num_leaves -> max_depth = ceil(log2(num_leaves))`, `min_child_samples -> min_child_weight`, `feature_fraction -> colsample_bytree`, same `n_estimators` and `learning_rate`. Both baseline candidates are refit at each origin; they do not use future labels.
- Reported references: `naive`, `seasonal_naive_52`, `tsb`, `historical_average`.
- Same features and recursive one-step strategy as the champion. No label after an origin is used.
- MAPE is computed on non-zero actuals only; the excluded-zero count is shown. MAPE is reported, not scored.
- SARIMA/ARIMA deferred: 616 mostly intermittent series, synthetic data and a 14-week horizon make per-series fits costly without decision value. LSTM deferred: no GPU, small synthetic sample, and recursive tabular challengers cover the non-linear comparison.

## Tuning results — weekly level

| candidate_id | candidate_role | wape_pct | bias_pct | mase | mae | rmse | mape_pct | mape_nonzero_n | mape_zero_excluded | runtime_seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| lightgbm_tweedie::random_1 | champion | 63.11 | -18.02 | 0.79 | 13.13 | 55.89 | 90.61 | 8147 | 7775 |  |
| moving_average_13::baseline_default | new_candidate | 69.07 | -10.36 | 0.87 | 14.38 | 55.84 | 101.43 | 8147 | 7775 | 34.46 |
| tsb::baseline_default | reported_reference | 69.32 | -9.12 | 0.87 | 14.43 | 55.84 | 97.71 | 8147 | 7775 | 19.96 |
| exponential_smoothing::baseline_default | new_candidate | 70.32 | -6.63 | 0.88 | 14.63 | 56.38 | 100.03 | 8147 | 7775 | 9.66 |
| historical_average::origin_mean | reported_reference | 71.35 | -12.06 | 0.90 | 14.85 | 61.94 | 91.94 | 8147 | 7775 | 0.44 |
| xgboost::random_1 | new_candidate | 74.59 | -1.46 | 0.94 | 15.52 | 61.30 | 99.61 | 8147 | 7775 | 15.44 |
| xgboost::random_3 | new_candidate | 75.90 | -3.16 | 0.95 | 15.80 | 58.03 | 95.50 | 8147 | 7775 | 22.28 |
| xgboost::random_2 | new_candidate | 77.00 | 1.26 | 0.97 | 16.02 | 59.04 | 101.45 | 8147 | 7775 | 20.91 |
| seasonal_naive_52::baseline_default | reported_reference | 79.67 | -16.06 | 1.00 | 16.58 | 61.77 | 124.94 | 8147 | 7775 | 18.05 |
| naive::baseline_default | reported_reference | 85.00 | 3.08 | 1.07 | 17.69 | 61.51 | 119.43 | 8147 | 7775 | 8.71 |

## Tuning results — lead-time windows

| candidate_id | candidate_role | wape_pct | bias_pct | mase | mae | rmse | mape_pct | mape_nonzero_n | mape_zero_excluded | runtime_seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| lightgbm_tweedie::random_1 | champion | 39.29 | -16.93 | 2.33 | 38.83 | 148.68 | 69.81 | 821 | 301 |  |
| tsb::baseline_default | reported_reference | 44.48 | -12.40 | 2.64 | 43.95 | 152.30 | 101.18 | 821 | 301 | 19.96 |
| moving_average_13::baseline_default | new_candidate | 45.07 | -12.16 | 2.67 | 44.54 | 153.54 | 113.42 | 821 | 301 | 34.46 |
| historical_average::origin_mean | reported_reference | 46.08 | -18.36 | 2.73 | 45.53 | 162.04 | 98.37 | 821 | 301 | 0.44 |
| xgboost::random_1 | new_candidate | 46.12 | -4.89 | 2.73 | 45.58 | 156.51 | 118.58 | 821 | 301 | 15.44 |
| exponential_smoothing::baseline_default | new_candidate | 46.14 | -9.13 | 2.74 | 45.59 | 156.68 | 111.28 | 821 | 301 | 9.66 |
| xgboost::random_2 | new_candidate | 47.55 | -7.23 | 2.82 | 46.99 | 155.30 | 129.84 | 821 | 301 | 20.91 |
| xgboost::random_3 | new_candidate | 47.82 | -10.43 | 2.83 | 47.25 | 155.08 | 142.83 | 821 | 301 | 22.28 |
| seasonal_naive_52::baseline_default | reported_reference | 51.29 | -14.98 | 3.04 | 50.68 | 162.81 | 138.19 | 821 | 301 | 18.05 |
| naive::baseline_default | reported_reference | 61.81 | -1.34 | 3.67 | 61.08 | 182.97 | 159.84 | 821 | 301 | 8.71 |

## Advancement gate

Tuning gate: **NOT MET** — best new candidate `moving_average_13::baseline_default` reached 45.07% lead-time WAPE against the champion's 39.29% (gain -5.78 pp).

The held origins were not opened. The champion stays frozen and no held-forecast look was added to the ledger.

**Decision: `champion_retained_tuning_gate_not_met`.**

## One held evaluation (only if the gate passed)

No held evaluation was opened.

## Limits

- Two tuning origins only; no significance test is claimed from tuning alone.
- XGBoost is one seed and three configurations; its result is conditional on that budget.
- Lead-time WAPE sums a supplier's full window per part-region, so it is easier than weekly WAPE and not comparable across levels.
- The final test remains sealed; nothing here opens it.
