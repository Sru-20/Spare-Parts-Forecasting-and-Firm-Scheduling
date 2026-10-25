# Phase 4 — Track A selection

This evaluation uses simulated data and train-validation evidence only. The final-test labels remain sealed.

- Tier 2 Random Forest fitting rows logged successfully: 6.
- Sensitivity-stable winner: no.
- Frozen model artifact: not created.
- Future output rows saved: 0.
- Phase 3 lead-time correction rows: 12.

## Selection

| Rank | Candidate | Score | Lead-time WAPE % | Weekly WAPE % | Calibration score |
|---:|---|---:|---:|---:|---:|
| 1 | simple_average_ensemble::equal_weight_nine_model_mean | 71.458 | 37.904 | 69.258 | 0.00 |
| 2 | lightgbm_quantile::random_1 | 69.792 | 36.202 | 57.714 | 50.00 |
| 3 | lightgbm_tweedie::random_1 | 68.958 | 34.306 | 63.253 | 0.00 |
| 4 | sbc_segment_router::fixed_sbc_model_map | 63.125 | 38.798 | 71.022 | 0.00 |
| 5 | sba::baseline_default | 54.583 | 40.643 | 72.033 | 0.00 |
| 6 | tsb::baseline_default | 52.917 | 40.959 | 72.284 | 0.00 |
| 7 | croston::baseline_default | 42.083 | 41.858 | 73.343 | 0.00 |
| 8 | moving_average_13::baseline_default | 31.250 | 44.950 | 75.641 | 0.00 |
| 9 | random_forest::tier2_random_2 | 29.792 | 41.719 | 77.114 | 0.00 |
| 10 | naive::baseline_default | 27.917 | 60.012 | 78.873 | 0.00 |

## Conditional Tier 2 result

Random Forest ranked 9 of 13 candidates (selection score 29.792; lead-time WAPE 41.719%; weekly WAPE 77.114%; mixed intermittent/lumpy WAPE 126.646%; runtime 2052.7 seconds). It did not improve on the best Tier 1 intermittent/lumpy result.

## Weight sensitivity

Winner changed in 6 of 16 one-at-a-time ±10-point perturbations.

## Statistical comparisons

Pairwise comparisons among the top five produced 10 rows. DM uses weekly WAPE by target week, Bartlett HAC through lag 13, and the 14-week HLN correction. Bootstrap intervals resample part-region clusters with 2,000 paired replicates.

| Candidate A | Candidate B | DM mean WAPE diff. (B−A) | Holm p | Weekly WAPE diff. 95% CI | Lead-time WAPE diff. 95% CI |
|---|---|---:|---:|---:|---:|
| simple_average_ensemble::equal_weight_nine_model_mean | lightgbm_quantile::random_1 | -12.350 | 2.423e-12 | [-15.524, -8.360] | [-3.653, 2.103] |
| simple_average_ensemble::equal_weight_nine_model_mean | lightgbm_tweedie::random_1 | -6.288 | 0.0003391 | [-9.763, -3.261] | [-4.487, -1.217] |
| simple_average_ensemble::equal_weight_nine_model_mean | sbc_segment_router::fixed_sbc_model_map | 1.976 | 0.1681 | [0.232, 3.788] | [-0.843, 2.565] |
| simple_average_ensemble::equal_weight_nine_model_mean | sba::baseline_default | 3.121 | 0.1127 | [1.443, 4.621] | [1.143, 4.143] |
| lightgbm_quantile::random_1 | lightgbm_tweedie::random_1 | 6.062 | 1.546e-05 | [4.706, 6.521] | [-4.284, 0.020] |
| lightgbm_quantile::random_1 | sbc_segment_router::fixed_sbc_model_map | 14.326 | 7.114e-12 | [9.642, 18.641] | [-1.523, 4.616] |
| lightgbm_quantile::random_1 | sba::baseline_default | 15.467 | 1.575e-06 | [10.160, 19.810] | [0.085, 6.929] |
| lightgbm_tweedie::random_1 | sbc_segment_router::fixed_sbc_model_map | 8.264 | 9.652e-14 | [4.409, 12.831] | [1.950, 5.730] |
| lightgbm_tweedie::random_1 | sba::baseline_default | 9.405 | 0.001382 | [4.902, 14.074] | [2.880, 8.187] |
| sbc_segment_router::fixed_sbc_model_map | sba::baseline_default | 1.244 | 0.4776 | [-0.337, 2.542] | [0.136, 3.737] |

## Calibration

| Candidate | Weekly P90/P95/P99 | Weekly coverage error | Weekly pinball | Lead-time P90/P95/P99 | Lead-time coverage error | Lead-time pinball |
|---|---:|---:|---:|---:|---:|---:|
| croston::baseline_default | n/a / n/a / n/a | n/a | n/a | n/a / n/a / n/a | n/a | n/a |
| lightgbm_point::random_3 | n/a / n/a / n/a | n/a | n/a | n/a / n/a / n/a | n/a | n/a |
| lightgbm_quantile::random_1 | 0.8829 / 0.9265 / 0.9737 | 0.0190 | 4.1206 | 0.9309 / 0.9741 / 0.9964 | 0.0205 | 6.9182 |
| lightgbm_tweedie::random_1 | n/a / n/a / n/a | n/a | n/a | n/a / n/a / n/a | n/a | n/a |
| moving_average_13::baseline_default | n/a / n/a / n/a | n/a | n/a | n/a / n/a / n/a | n/a | n/a |
| naive::baseline_default | n/a / n/a / n/a | n/a | n/a | n/a / n/a / n/a | n/a | n/a |
| random_forest::tier2_random_2 | n/a / n/a / n/a | n/a | n/a | n/a / n/a / n/a | n/a | n/a |
| same_period_last_year::baseline_default | n/a / n/a / n/a | n/a | n/a | n/a / n/a / n/a | n/a | n/a |
| sba::baseline_default | n/a / n/a / n/a | n/a | n/a | n/a / n/a / n/a | n/a | n/a |
| sbc_segment_router::fixed_sbc_model_map | n/a / n/a / n/a | n/a | n/a | n/a / n/a / n/a | n/a | n/a |
| seasonal_naive_52::baseline_default | n/a / n/a / n/a | n/a | n/a | n/a / n/a / n/a | n/a | n/a |
| simple_average_ensemble::equal_weight_nine_model_mean | n/a / n/a / n/a | n/a | n/a | n/a / n/a / n/a | n/a | n/a |
| tsb::baseline_default | n/a / n/a / n/a | n/a | n/a | n/a / n/a / n/a | n/a | n/a |

## Exit-gate blocker

No stable winner: the top weighted candidate changes in the one-at-a-time weight-sensitivity analysis. Top contenders are retained in the selection matrix; a champion was not forced.

Tier 3 is deferred: the source is synthetic, only four held origins are available, and more complex models have no demonstrated decision value on this validation sample.

These results are method validation on simulated data, not evidence of factory savings. Four held origins and dependent part-region observations limit inferential power.
