# SPEC v4.2 B — interim 26-week backtest

Dataset: synthetic. Results validate method behaviour on simulated data; they are not factory savings.

## Windows and design

- Origin 2025-03-24: 2025-03-31 to 2025-09-22
- Origin 2025-09-22: 2025-09-29 to 2026-03-23

- Two consecutive 26-week blocks ending at the train-validation cutoff 2026-03-23.
- Each window is fitted only on labels up to its origin; warm-up history initialises lags and seasonal terms. Recursive one-step features are rebuilt inside the origin. Frozen hyperparameters are read from the Phase 3 tuning log and are not re-tuned here.
- Models: champion `lightgbm_tweedie::random_1`, quantile `lightgbm_quantile::random_1`, nine-model equal-weight ensemble, best intermittent baseline (`tsb`), `naive`, `seasonal_naive_52`, `historical_average`.
- Forecast Accuracy % = 100 − WAPE %.
- Complete 26-week aggregate requires 26 observed demand weeks and 26 finite forecasts per part-region; incomplete series are counted, not imputed.

## Weekly results (pooled across both windows)

| model | wape_pct | forecast_accuracy_pct | bias_pct | mase | n_scored | excluded_missing_actual | excluded_missing_forecast |
|---|---|---|---|---|---|---|---|
| lightgbm_quantile | 60.12 | 39.88 | -34.15 | 0.71 | 29950 | 22 | 0 |
| lightgbm_tweedie | 66.15 | 33.85 | -14.03 | 0.78 | 29950 | 22 | 0 |
| simple_average_ensemble | 71.02 | 28.98 | -5.79 | 0.84 | 29950 | 22 | 0 |
| tsb | 73.77 | 26.23 | -1.55 | 0.87 | 29566 | 22 | 384 |
| historical_average | 78.06 | 21.94 | -2.90 | 0.92 | 29566 | 22 | 384 |
| naive | 88.71 | 11.29 | 4.97 | 1.04 | 29566 | 22 | 384 |
| seasonal_naive_52 | 90.88 | 9.12 | -1.71 | 1.07 | 29566 | 22 | 384 |

## Weekly results by origin

| origin | model | wape_pct | forecast_accuracy_pct | bias_pct | mase | n_scored | excluded_missing_actual | excluded_missing_forecast |
|---|---|---|---|---|---|---|---|---|
| 2025-03-24 | lightgbm_quantile | 59.62 | 40.38 | -32.77 | 0.67 | 14785 | 9 | 0 |
| 2025-03-24 | lightgbm_tweedie | 66.27 | 33.73 | -15.10 | 0.75 | 14785 | 9 | 0 |
| 2025-03-24 | simple_average_ensemble | 72.69 | 27.31 | -5.99 | 0.82 | 14785 | 9 | 0 |
| 2025-03-24 | tsb | 78.57 | 21.43 | 1.69 | 0.88 | 14785 | 9 | 0 |
| 2025-03-24 | historical_average | 79.84 | 20.16 | 1.35 | 0.90 | 14785 | 9 | 0 |
| 2025-03-24 | naive | 82.10 | 17.90 | -11.86 | 0.92 | 14785 | 9 | 0 |
| 2025-03-24 | seasonal_naive_52 | 85.82 | 14.18 | -4.54 | 0.97 | 14785 | 9 | 0 |
| 2025-09-22 | lightgbm_quantile | 60.57 | 39.43 | -35.42 | 0.74 | 15165 | 13 | 0 |
| 2025-09-22 | lightgbm_tweedie | 66.04 | 33.96 | -13.04 | 0.81 | 15165 | 13 | 0 |
| 2025-09-22 | tsb | 69.34 | 30.66 | -4.54 | 0.85 | 14781 | 13 | 384 |
| 2025-09-22 | simple_average_ensemble | 69.49 | 30.51 | -5.61 | 0.85 | 15165 | 13 | 0 |
| 2025-09-22 | historical_average | 76.41 | 23.59 | -6.82 | 0.94 | 14781 | 13 | 384 |
| 2025-09-22 | naive | 94.80 | 5.20 | 20.48 | 1.17 | 14781 | 13 | 384 |
| 2025-09-22 | seasonal_naive_52 | 95.55 | 4.45 | 0.91 | 1.18 | 14781 | 13 | 384 |

## Complete 26-week part-region aggregate

| model | wape_pct | forecast_accuracy_pct | bias_pct | n_scored | incomplete_series_excluded |
|---|---|---|---|---|---|
| lightgbm_tweedie | 33.05 | 66.95 | -14.20 | 1116 | 65 |
| simple_average_ensemble | 35.86 | 64.14 | -6.55 | 1116 | 65 |
| tsb | 38.20 | 61.80 | -3.11 | 1116 | 65 |
| lightgbm_quantile | 41.90 | 58.10 | -34.19 | 1116 | 65 |
| historical_average | 42.21 | 57.79 | -4.92 | 1116 | 65 |
| seasonal_naive_52 | 44.76 | 55.24 | -1.79 | 1116 | 65 |
| naive | 72.92 | 27.08 | 5.40 | 1116 | 65 |

## Horizon bands

| segment | model | wape_pct | forecast_accuracy_pct | bias_pct | mase | n_scored | excluded_missing_actual | excluded_missing_forecast |
|---|---|---|---|---|---|---|---|---|
| horizon_14_26 | lightgbm_quantile | 59.88 | 40.12 | -38.44 | 0.73 | 15113 | 0 | 0 |
| horizon_14_26 | lightgbm_tweedie | 65.52 | 34.48 | -18.24 | 0.80 | 15113 | 0 | 0 |
| horizon_14_26 | simple_average_ensemble | 70.11 | 29.89 | -9.67 | 0.85 | 15113 | 0 | 0 |
| horizon_14_26 | tsb | 72.97 | 27.03 | -4.90 | 0.89 | 14782 | 0 | 0 |
| horizon_14_26 | historical_average | 77.66 | 22.34 | -6.19 | 0.95 | 14782 | 0 | 0 |
| horizon_14_26 | naive | 85.88 | 14.12 | 1.30 | 1.05 | 14782 | 0 | 0 |
| horizon_14_26 | seasonal_naive_52 | 88.27 | 11.73 | -6.52 | 1.07 | 14782 | 0 | 0 |
| horizon_1_13 | lightgbm_quantile | 60.37 | 39.63 | -29.53 | 0.69 | 14837 | 0 | 0 |
| horizon_1_13 | lightgbm_tweedie | 66.83 | 33.17 | -9.49 | 0.76 | 14837 | 0 | 0 |
| horizon_1_13 | simple_average_ensemble | 72.01 | 27.99 | -1.61 | 0.82 | 14837 | 0 | 0 |
| horizon_1_13 | tsb | 74.63 | 25.37 | 2.05 | 0.85 | 14784 | 0 | 0 |
| horizon_1_13 | historical_average | 78.48 | 21.52 | 0.64 | 0.89 | 14784 | 0 | 0 |
| horizon_1_13 | naive | 91.75 | 8.25 | 8.90 | 1.04 | 14784 | 0 | 0 |
| horizon_1_13 | seasonal_naive_52 | 93.69 | 6.31 | 3.47 | 1.07 | 14784 | 0 | 0 |

## SBC class

| segment | model | wape_pct | forecast_accuracy_pct | bias_pct | mase | n_scored | excluded_missing_actual | excluded_missing_forecast |
|---|---|---|---|---|---|---|---|---|
| Erratic | lightgbm_quantile | 78.42 | 21.58 | -56.24 | 0.73 | 5065 | 0 | 0 |
| Erratic | lightgbm_tweedie | 86.90 | 13.10 | -12.88 | 0.81 | 5065 | 0 | 0 |
| Erratic | simple_average_ensemble | 88.28 | 11.72 | -9.31 | 0.83 | 5065 | 0 | 0 |
| Erratic | historical_average | 90.02 | 9.98 | -6.20 | 0.84 | 5065 | 0 | 0 |
| Erratic | tsb | 92.32 | 7.68 | -1.22 | 0.86 | 5065 | 0 | 0 |
| Erratic | naive | 112.58 | -12.58 | -1.43 | 1.05 | 5065 | 0 | 0 |
| Erratic | seasonal_naive_52 | 113.62 | -13.62 | -1.00 | 1.06 | 5065 | 0 | 0 |
| Intermittent | lightgbm_quantile | 101.74 | -1.74 | -83.03 | 0.73 | 7119 | 0 | 0 |
| Intermittent | lightgbm_tweedie | 117.40 | -17.40 | -23.77 | 0.84 | 7119 | 0 | 0 |
| Intermittent | simple_average_ensemble | 122.44 | -22.44 | -8.49 | 0.88 | 7119 | 0 | 0 |
| Intermittent | historical_average | 125.21 | -25.21 | -1.02 | 0.90 | 7119 | 0 | 0 |
| Intermittent | tsb | 125.91 | -25.91 | 1.01 | 0.90 | 7119 | 0 | 0 |
| Intermittent | seasonal_naive_52 | 140.35 | -40.35 | 2.92 | 1.01 | 7119 | 0 | 0 |
| Intermittent | naive | 143.07 | -43.07 | 4.06 | 1.03 | 7119 | 0 | 0 |
| Lumpy | lightgbm_quantile | 99.82 | 0.18 | -93.81 | 0.46 | 11225 | 0 | 0 |
| Lumpy | lightgbm_tweedie | 115.42 | -15.42 | -71.70 | 0.53 | 11225 | 0 | 0 |
| Lumpy | simple_average_ensemble | 157.82 | -57.82 | -20.67 | 0.73 | 11225 | 0 | 0 |
| Lumpy | tsb | 169.83 | -69.83 | -8.55 | 0.79 | 11225 | 0 | 0 |
| Lumpy | naive | 173.26 | -73.26 | -18.72 | 0.80 | 11225 | 0 | 0 |
| Lumpy | seasonal_naive_52 | 186.64 | -86.64 | 0.92 | 0.86 | 11225 | 0 | 0 |
| Lumpy | historical_average | 190.96 | -90.96 | 14.80 | 0.88 | 11225 | 0 | 0 |
| Smooth | lightgbm_quantile | 39.31 | 60.69 | -6.62 | 0.93 | 6157 | 0 | 0 |
| Smooth | simple_average_ensemble | 39.95 | 60.05 | -0.60 | 0.95 | 6157 | 0 | 0 |
| Smooth | tsb | 40.16 | 59.84 | -0.44 | 0.95 | 6157 | 0 | 0 |
| Smooth | lightgbm_tweedie | 41.35 | 58.65 | -1.30 | 0.98 | 6157 | 0 | 0 |
| Smooth | historical_average | 44.46 | 55.54 | -5.21 | 1.06 | 6157 | 0 | 0 |
| Smooth | naive | 54.78 | 45.22 | 13.14 | 1.30 | 6157 | 0 | 0 |
| Smooth | seasonal_naive_52 | 55.53 | 44.47 | -2.99 | 1.32 | 6157 | 0 | 0 |
| Unclassified | lightgbm_quantile | 97.92 | 2.08 | -92.36 |  | 384 | 0 | 0 |
| Unclassified | simple_average_ensemble | 98.93 | 1.07 | -44.22 |  | 384 | 0 | 0 |
| Unclassified | lightgbm_tweedie | 99.40 | 0.60 | -89.29 |  | 384 | 0 | 0 |

## Criticality

| segment | model | wape_pct | forecast_accuracy_pct | bias_pct | mase | n_scored | excluded_missing_actual | excluded_missing_forecast |
|---|---|---|---|---|---|---|---|---|
| High | lightgbm_quantile | 62.61 | 37.39 | -34.64 | 0.79 | 10154 | 0 | 0 |
| High | simple_average_ensemble | 68.98 | 31.02 | -8.27 | 0.87 | 10154 | 0 | 0 |
| High | lightgbm_tweedie | 69.18 | 30.82 | -13.13 | 0.87 | 10154 | 0 | 0 |
| High | tsb | 70.95 | 29.05 | -4.14 | 0.89 | 9818 | 0 | 0 |
| High | historical_average | 73.92 | 26.08 | -6.47 | 0.93 | 9818 | 0 | 0 |
| High | naive | 77.61 | 22.39 | -4.44 | 0.98 | 9818 | 0 | 0 |
| High | seasonal_naive_52 | 91.00 | 9.00 | -2.95 | 1.15 | 9818 | 0 | 0 |
| Low | lightgbm_quantile | 59.64 | 40.36 | -37.09 | 0.59 | 7118 | 0 | 0 |
| Low | lightgbm_tweedie | 65.13 | 34.87 | -17.81 | 0.64 | 7118 | 0 | 0 |
| Low | simple_average_ensemble | 76.14 | 23.86 | -5.12 | 0.75 | 7118 | 0 | 0 |
| Low | tsb | 79.84 | 20.16 | -0.02 | 0.79 | 7118 | 0 | 0 |
| Low | historical_average | 86.89 | 13.11 | 2.51 | 0.86 | 7118 | 0 | 0 |
| Low | seasonal_naive_52 | 88.53 | 11.47 | -7.66 | 0.88 | 7118 | 0 | 0 |
| Low | naive | 108.37 | -8.37 | 19.01 | 1.07 | 7118 | 0 | 0 |
| Medium | lightgbm_quantile | 58.26 | 41.74 | -32.04 | 0.72 | 12678 | 0 | 0 |
| Medium | lightgbm_tweedie | 64.14 | 35.86 | -12.62 | 0.80 | 12678 | 0 | 0 |
| Medium | simple_average_ensemble | 69.81 | 30.19 | -4.06 | 0.87 | 12678 | 0 | 0 |
| Medium | tsb | 72.67 | 27.33 | -0.23 | 0.90 | 12630 | 0 | 0 |
| Medium | historical_average | 76.48 | 23.52 | -2.98 | 0.95 | 12630 | 0 | 0 |
| Medium | naive | 86.82 | 13.18 | 4.88 | 1.08 | 12630 | 0 | 0 |
| Medium | seasonal_naive_52 | 92.14 | 7.86 | 2.79 | 1.15 | 12630 | 0 | 0 |

## Reading the tables

- `lightgbm_quantile` has the lowest weekly absolute error at the median but strongly understates totals (bias −34.15%); it is a benchmark line, not a replacement for the champion.
- `lightgbm_tweedie` (the frozen champion) has the lowest complete 26-week aggregate WAPE.
- Baselines lack forecasts for 384 part-region rows where the series has no usable history; `excluded_missing_forecast` records this and the model sample counts differ.
- These results are not used for champion selection and do not change the frozen champion.

## Caveats

- Only two windows and no significance test is claimed.
- Summed 26-week aggregate WAPE is easier than weekly WAPE and is **not comparable** to the 14-week supplier lead-time metric.
- The windows lie inside held validation and overlap forecast horizons already used in Phase 3; they are not independent evidence.
- The final test remains sealed.
- Data are synthetic; accuracy here supports method validation, not a factory forecast guarantee.
