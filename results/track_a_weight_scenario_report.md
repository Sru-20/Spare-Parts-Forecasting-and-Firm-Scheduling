# Track A weight-scenario comparison

## Status and evaluation boundary

The original Phase 4 scoring configuration and its results remain frozen. The alternative below is a business-proposed scenario only; it has **not** been adopted as project policy. The comparison re-scores the same 13 Phase 4 candidates using the saved four-origin held-validation metrics. It does not refit any model, tune on held validation, use final-test/future labels, select a champion, or start Phase 5.

Reproduce the comparison with:

```powershell
python -m src.forecasting.compare_weight_scenarios
```

The command uses the existing `_weighted_selection` and `_sensitivity` routines. It verifies that baseline scores and all 16 saved baseline sensitivity winners reproduce before writing comparison artifacts. `configs/project.yaml`, the original selection matrix, and the original baseline sensitivity results are not modified.

## Weights

| Component | Frozen baseline | Proposed business scenario | Change (percentage points) |
|---|---:|---:|---:|
| WAPE | 35% | 20% | -15 |
| Absolute bias | 15% | 15% | 0 |
| A-class / critical parts | 15% | 20% | +5 |
| Intermittent / lumpy demand | 10% | 20% | +10 |
| Calibration | 10% | 8% | -2 |
| Stability across origins | 5% | 7% | +2 |
| Runtime | 5% | 3% | -2 |
| Explainability | 5% | 7% | +2 |
| **Total** | **100%** | **100%** | |

The proposed emphasis on A-class/critical parts and intermittent/lumpy demand is a plausible business priority, not evidence that these exact percentages are optimal. Relative to the baseline it gives up 15 points of general WAPE and 2 points of calibration for the subgroup metrics, stability, and explainability. Business owners should confirm that trade-off against service targets, stock-out consequences, and inventory costs.

## Scoring and direction checks

- Each component is normalized independently by average rank over the same 13 eligible candidates. The best raw value receives 100, the worst 0, ties use average ranks, and higher component scores always improve the weighted score.
- Lower is better for mixed WAPE, absolute mixed bias, A-class/critical WAPE, intermittent/lumpy WAPE, lead-time WAPE standard deviation, and measured runtime. Higher is better for the configured explainability tier. The explainability tiers are qualitative project assumptions, not empirical quality measurements.
- Calibration combines the lead-time score (70%) and weekly score (30%). Each level averages the rank scores for absolute P90/P95/P99 coverage error and pinball loss. Missing probabilistic output scores 0; the point/ensemble candidates are not assigned synthetic quantiles.
- The total is the sum of the component rank scores multiplied by the scenario weights; no missing weight is redistributed. Ties follow the existing simplicity, explainability, runtime, and candidate-id ordering.
- Sensitivity perturbs each weight separately by +10 and -10 percentage points, floors at zero, renormalizes to 100%, and reranks all 13 candidates. All 16 cases per scenario are retained in the sensitivity CSV.

## Candidate ranking comparison

Scores are rank-normalized weighted scores, not forecast-error percentages or business savings.

| Baseline rank | Proposed rank | Candidate | Baseline score | Proposed score |
|---:|---:|---|---:|---:|
| 1 | 1 | Equal-weight ensemble | 71.458 | 71.042 |
| 2 | 3 | LightGBM quantile | 69.792 | 69.625 |
| 3 | 2 | LightGBM Tweedie | 68.958 | 70.958 |
| 4 | 4 | SBC segment router | 63.125 | 61.458 |
| 5 | 5 | SBA | 54.583 | 55.833 |
| 6 | 6 | TSB | 52.917 | 54.917 |
| 7 | 7 | Croston | 42.083 | 41.583 |
| 8 | 8 | 13-week moving average | 31.250 | 32.000 |
| 9 | 9 | Tier 2 Random Forest | 29.792 | 31.708 |
| 10 | 10 | Naive | 27.917 | 31.250 |
| 11 | 11 | Same period last year | 27.083 | 29.917 |
| 12 | 12 | Seasonal naive 52 | 26.250 | 29.417 |
| 13 | 13 | LightGBM point | 24.792 | 22.292 |

The nominal leader remains the equal-weight ensemble, but its proposed score is only **0.083** points above Tweedie. Tweedie moves from third to second. The quantile model moves from second to third.

## Full one-at-a-time sensitivity results

“Changed” indicates that the case's winner differs from that scenario's nominal winner. The baseline and proposed results below include every perturbation.

| Component perturbed | Direction | Baseline winner | Changed? | Proposed winner | Changed? |
|---|---|---|:---:|---|:---:|
| WAPE | +10 points | Quantile | Yes | Tweedie | Yes |
| WAPE | -10 points | Ensemble | No | Ensemble | No |
| Absolute bias | +10 points | Ensemble | No | Ensemble | No |
| Absolute bias | -10 points | Quantile | Yes | Quantile | Yes |
| A-class / critical | +10 points | Ensemble | No | Tweedie | Yes |
| A-class / critical | -10 points | Ensemble | No | Ensemble | No |
| Intermittent / lumpy | +10 points | Ensemble | No | Tweedie | Yes |
| Intermittent / lumpy | -10 points | Ensemble | No | Ensemble | No |
| Calibration | +10 points | Quantile | Yes | Quantile | Yes |
| Calibration | -10 points | Ensemble | No | Ensemble | No |
| Stability | +10 points | Quantile | Yes | Tweedie | Yes |
| Stability | -10 points | Ensemble | No | Ensemble | No |
| Runtime | +10 points | Ensemble | No | Ensemble | No |
| Runtime | -10 points | Quantile | Yes | Quantile | Yes |
| Explainability | +10 points | Quantile | Yes | Tweedie | Yes |
| Explainability | -10 points | Ensemble | No | Ensemble | No |

- Frozen baseline: winner changes in **6/16** cases.
- Proposed scenario: winner changes in **8/16** cases.
- The proposed scenario therefore remains unstable and is less stable under this defined sensitivity test, despite retaining the ensemble at nominal weights.

## Relation to business objectives

**Measured directly on simulated held-validation data:** weekly and lead-time forecast WAPE, mixed absolute bias, WAPE for the A-class/critical subgroup, WAPE for intermittent/lumpy series, quantile calibration for the quantile model, runtime, and lead-time-WAPE variation across four origins. For the leading candidates:

| Candidate | Weekly WAPE | Lead-time WAPE | A-class/critical WAPE | Intermittent/lumpy WAPE | Mixed absolute bias |
|---|---:|---:|---:|---:|---:|
| Equal-weight ensemble | 69.258% | 37.904% | 43.548% | 114.716% | 2.425% |
| LightGBM Tweedie | 63.253% | 34.306% | 41.326% | 92.146% | 8.812% |
| LightGBM quantile | 57.714% | 36.202% | 42.866% | 95.303% | 26.775% |

These show forecast-error trade-offs: the ensemble has much lower absolute bias than the two LightGBM alternatives; Tweedie has lower lead-time, critical-subgroup, and intermittent/lumpy WAPE than the ensemble; quantile has the lowest weekly WAPE among these three and is the only one with probabilistic calibration output. These measures do **not** establish inventory or production outcomes.

**Not directly measured in this Track A comparison:** critical-part availability, stock-out reduction, service/fill rate against criticality targets, emergency procurement count/cost, excess inventory, carrying cost, inventory turnover, schedule adherence, supplier/production feasibility, and reliable production/procurement scheduling. A-class/critical forecast WAPE is only a demand-prediction proxy for critical availability. The unmeasured outcomes require the paired, fixed-assumption simulator and scheduler evaluation; no such Phase 5 work is performed here. Because the inputs are synthetic, results are method validation on simulated data, not proven factory savings.

## Recommendation and gate

**Recommendation: retain the frozen baseline policy and keep Phase 4 blocked pending further evidence.** The proposed weights are a reasonable business-priority scenario to discuss, but they are not validated policy: the nominal score gap is negligible and the winner changes in 8/16 perturbations. Do not adopt them just because they leave the same nominal leader, and do not declare a champion. Better evidence would include business-owner-approved service/cost priorities and, later, measured paired scheduling outcomes under Phase 5's frozen simulator assumptions. No Phase 5 work was started.

## Validation and artifacts

- Baseline reproduction: 13 candidates; saved baseline scores and all 16 baseline sensitivity winners matched.
- Alternative evaluation: 13 candidates; 16/16 sensitivity cases evaluated.
- Full test suite: 39 passed under Python 3.10 and 39 passed under Python 3.13.
- Dataset validator: 32 PASS, 0 FAIL.
- Outputs: [`track_a_weight_scenario_weights.csv`](./track_a_weight_scenario_weights.csv), [`track_a_weight_scenario_comparison.csv`](./track_a_weight_scenario_comparison.csv), [`track_a_weight_scenario_rankings.csv`](./track_a_weight_scenario_rankings.csv), and [`track_a_weight_scenario_sensitivity.csv`](./track_a_weight_scenario_sensitivity.csv).
