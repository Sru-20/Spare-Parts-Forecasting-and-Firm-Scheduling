# Phase 4 v4.1 maintenance addendum

This addendum summarizes the existing held-validation artifacts. No model was
retrained, no selection or frozen forecast artifact was changed, and final-test
labels were not opened. The results are method-validation evidence on synthetic
data, not proven factory savings.

## Paired lead-time WAPE differences

Difference is champion minus comparator in WAPE ratio units; negative favors
the champion. The 95% intervals use the saved lead-time aggregates and the
project's two-way paired bootstrap, resampling part-region series and forecast
origins together (2,000 replicates, seed 42 and 43).

| Comparison | Difference | 95% interval | Interpretation |
|---|---:|---:|---|
| Tweedie champion − equal-weight ensemble | -0.0320 | [-0.0629, -0.0041] | Interval excludes zero; lower champion WAPE under this bootstrap |
| Tweedie champion − quantile LightGBM | -0.0211 | [-0.0610, 0.0209] | **Not statistically separated**; interval includes zero |

There are only four held origins, and temporal dependence plus synthetic data
limit inference. These intervals are not a claim of factory-level benefit.

## Critical and A-class performance

The combined segment is the union of High-criticality and ABC A lead-time
aggregates; rows satisfying both are counted once. Asymmetric bias uses the
configured under-forecast penalty factor of 2.

| Segment / check | n lead-time rows | Lead-time WAPE | Signed bias | Asymmetric bias | Under-forecast rule |
|---|---:|---:|---:|---:|---|
| High criticality OR ABC A | 1,334 | 0.3210 | -0.0464 | 0.0929 | Not the standalone High-criticality screen |
| High criticality | 744 | — | -0.0704 | — | Point estimate is above -0.10; 95% two-way bootstrap CI [-0.2054, 0.0733] crosses -0.10, so the screen is **inconclusive** |

The signed-bias convention is forecast minus actual divided by actual, so a
negative value denotes under-forecasting. WAPE and bias are ratios, not
percentages.

## Quantile coverage by criticality and horizon

Coverage is the share of scored weekly forecasts with actual demand no greater
than the corresponding quantile. Counts are scored rows from the quantile
LightGBM model across the four held-validation origins. For reporting, groups
with fewer than 30 observations would be labeled “insufficient data”; all groups
below meet that minimum. This count threshold is a reporting convention, not a
model-selection gate.

| Criticality | Horizon week | n | P90 coverage | P95 coverage | P99 coverage |
|---|---:|---:|---:|---:|---:|
| High | 1 | 756 | 0.9061 | 0.9643 | 0.9881 |
| High | 2 | 756 | 0.9048 | 0.9444 | 0.9841 |
| High | 3 | 763 | 0.8912 | 0.9240 | 0.9712 |
| High | 4 | 764 | 0.8966 | 0.9241 | 0.9725 |
| High | 5 | 766 | 0.8486 | 0.9060 | 0.9634 |
| High | 6 | 767 | 0.8540 | 0.9074 | 0.9583 |
| High | 7 | 771 | 0.8625 | 0.9287 | 0.9754 |
| High | 8 | 774 | 0.8734 | 0.9212 | 0.9755 |
| High | 9 | 775 | 0.8916 | 0.9303 | 0.9729 |
| High | 10 | 773 | 0.8823 | 0.9224 | 0.9715 |
| High | 11 | 775 | 0.8723 | 0.9148 | 0.9768 |
| High | 12 | 775 | 0.8981 | 0.9381 | 0.9781 |
| High | 13 | 772 | 0.8640 | 0.9249 | 0.9715 |
| High | 14 | 775 | 0.8684 | 0.9200 | 0.9652 |
| Medium | 1 | 972 | 0.9095 | 0.9455 | 0.9877 |
| Medium | 2 | 971 | 0.8950 | 0.9485 | 0.9815 |
| Medium | 3 | 972 | 0.9043 | 0.9486 | 0.9856 |
| Medium | 4 | 972 | 0.8981 | 0.9372 | 0.9794 |
| Medium | 5 | 972 | 0.8477 | 0.8920 | 0.9609 |
| Medium | 6 | 972 | 0.8560 | 0.9105 | 0.9640 |
| Medium | 7 | 972 | 0.8879 | 0.9218 | 0.9702 |
| Medium | 8 | 970 | 0.8784 | 0.9196 | 0.9670 |
| Medium | 9 | 972 | 0.8951 | 0.9249 | 0.9712 |
| Medium | 10 | 971 | 0.9053 | 0.9341 | 0.9732 |
| Medium | 11 | 972 | 0.8796 | 0.9218 | 0.9702 |
| Medium | 12 | 972 | 0.9033 | 0.9414 | 0.9794 |
| Medium | 13 | 970 | 0.8598 | 0.9093 | 0.9567 |
| Medium | 14 | 974 | 0.8686 | 0.9148 | 0.9754 |
| Low | 1 | 548 | 0.8978 | 0.9361 | 0.9818 |
| Low | 2 | 548 | 0.8923 | 0.9343 | 0.9818 |
| Low | 3 | 548 | 0.8869 | 0.9288 | 0.9872 |
| Low | 4 | 548 | 0.8978 | 0.9416 | 0.9818 |
| Low | 5 | 547 | 0.8665 | 0.8995 | 0.9561 |
| Low | 6 | 547 | 0.8812 | 0.9159 | 0.9671 |
| Low | 7 | 548 | 0.9069 | 0.9580 | 0.9799 |
| Low | 8 | 548 | 0.8814 | 0.9270 | 0.9745 |
| Low | 9 | 547 | 0.8885 | 0.9232 | 0.9799 |
| Low | 10 | 548 | 0.8905 | 0.9325 | 0.9799 |
| Low | 11 | 548 | 0.8887 | 0.9380 | 0.9708 |
| Low | 12 | 548 | 0.8996 | 0.9325 | 0.9854 |
| Low | 13 | 547 | 0.8629 | 0.9068 | 0.9653 |
| Low | 14 | 546 | 0.8388 | 0.9011 | 0.9652 |

## Selection context and preserved scenario

- Three candidates passed the enforced gates and were eligible for weighted
  selection: `lightgbm_tweedie::random_1`,
  `lightgbm_quantile::random_1`, and `tsb::baseline_default`.
- The champion won 14/14 v4.1 sensitivity cases. This denominator is the
  eligible candidate set after gates, not the broader Phase 3 candidate list.
- The prior “frozen baseline” weights-only scenario was run and remains
  preserved in `track_a_weight_scenario_rankings.csv`. Its winner was
  `simple_average_ensemble::equal_weight_nine_model_mean`, with score 71.458
  in that scenario. This is an extra historical scenario, not the v4.1
  decision. Its associated sensitivity outcomes (baseline 6/16; proposed
  business weights 8/16) remain preserved in the existing scenario report.
- The v4.1 decision follows screening gates, then weighted ranking, then
  downstream validation/tie-break; it does not select by weights alone.
