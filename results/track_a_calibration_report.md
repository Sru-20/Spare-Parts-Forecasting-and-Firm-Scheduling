# Track A calibration report

Calibration is scored on mean absolute P90/P95/P99 coverage error and mean pinball loss. A candidate without all required quantile outputs receives the minimum calibration component score. The calibration component combines lead-time score at 70% and weekly score at 30%.

| Candidate | Weekly P90/P95/P99 coverage | Weekly coverage error | Weekly pinball | Lead-time P90/P95/P99 coverage | Lead-time coverage error | Lead-time pinball | Calibration score |
|---|---:|---:|---:|---:|---:|---:|---:|
| croston::baseline_default | nan / nan / nan | nan | nan | nan / nan / nan | nan | nan | 0.0 |
| lightgbm_point::random_3 | nan / nan / nan | nan | nan | nan / nan / nan | nan | nan | 0.0 |
| lightgbm_quantile::random_1 | 0.8829295704295704 / 0.926510989010989 / 0.9736513486513486 | 0.018969363969363957 | 4.1205804892778755 | 0.9309090909090909 / 0.9740909090909091 / 0.9963636363636363 | 0.020454545454545465 | 6.918225303048125 | 50.0 |
| lightgbm_tweedie::random_1 | nan / nan / nan | nan | nan | nan / nan / nan | nan | nan | 0.0 |
| moving_average_13::baseline_default | nan / nan / nan | nan | nan | nan / nan / nan | nan | nan | 0.0 |
| naive::baseline_default | nan / nan / nan | nan | nan | nan / nan / nan | nan | nan | 0.0 |
| random_forest::tier2_random_2 | nan / nan / nan | nan | nan | nan / nan / nan | nan | nan | 0.0 |
| same_period_last_year::baseline_default | nan / nan / nan | nan | nan | nan / nan / nan | nan | nan | 0.0 |
| sba::baseline_default | nan / nan / nan | nan | nan | nan / nan / nan | nan | nan | 0.0 |
| sbc_segment_router::fixed_sbc_model_map | nan / nan / nan | nan | nan | nan / nan / nan | nan | nan | 0.0 |
| seasonal_naive_52::baseline_default | nan / nan / nan | nan | nan | nan / nan / nan | nan | nan | 0.0 |
| simple_average_ensemble::equal_weight_nine_model_mean | nan / nan / nan | nan | nan | nan / nan / nan | nan | nan | 0.0 |
| tsb::baseline_default | nan / nan / nan | nan | nan | nan / nan / nan | nan | nan | 0.0 |
