# Spare Parts Demand Forecasting and Firm Scheduling
## Project Progress Report

**Prepared for:** Project Mentor  
**Report date:** 23 October 2026

## Project objective

This project is developing a system to forecast spare-parts demand over a 14-week horizon, translate demand forecasts into feasible procurement or production schedules, and provide a multilingual assistant for approved user requests.

The evaluation prioritizes service, inventory, and scheduling outcomes. The supplied datasets are synthetic, so results to date demonstrate methods tested on simulated data. They do not establish real factory savings or operational impact.

## Tasks completed

### Requirements, data, and evaluation setup

- Reviewed the final project guide, updated problem statement, README, data-generation code, validation code, and the three supplied CSV datasets.
- Documented a missing source document: the README refers to an original problem statement PDF that was not present. The available updated statement and final project guide were used.
- Recorded dataset definitions, date ranges, environment constraints, generator assumptions, requirements traceability, forecast metrics, leakage safeguards, and evaluation rules.
- Confirmed data validation passed: **32 checks passed, 0 failed**.
- Preserved the original data files; derived data and evaluation outputs are stored separately.

### Data preparation and quality checks

| Dataset | Rows × columns | Description |
|---|---:|---|
| Demand forecasting | 120,861 × 62 | Part-region-week demand history and future calendar rows |
| Scheduling inputs | 158 × 61 | Part-level inventory and supplier snapshot, dated 26 September 2026 |
| Spare-parts catalogue | 160 × 29 | Part and catalogue attributes |

- Created a consolidated modelling table (**120,861 × 73**) and scheduling-input table enriched with catalogue attributes (**158 × 84**). Join row counts were preserved.
- Audited missing values, duplicate keys, negative quantities, weekly continuity, and cross-file references.
- Chose raw `demand_qty` as the training and scoring target. The supplied capped target is calculated using information from a broader time period and is not suitable for earlier rolling forecast origins.
- Kept missing-week targets blank and excluded them from fitting/scoring; genuine zero-demand weeks remain zero.
- Created a data dictionary and a feature-availability/leakage audit. Outcome fields and generator answer keys are excluded from model features.
- Recorded that 72 historical target values are missing and 54,511 observed historical rows have genuine zero demand.

### Demand exploration and forecasting features

- Completed descriptive analysis using **66,345** observed development-period demand rows through 23 March 2026. Final-test and future labels were not used.
- Classified 616 part-region demand series: Smooth 119, Erratic 97, Intermittent 133, Lumpy 220, and Unclassified/short 47.
- Implemented origin-safe lag, rolling, intermittent-demand, calendar, and recursive forecasting features.
- Checked the supplied demand-class labels against recomputed development-period classes; they matched for all **569 comparable series**.
- Produced data summaries, feature documentation, segmentation artifacts, and charts. Event patterns are reported as descriptive associations; they are not treated as causal evidence.

### Forecast model evaluation

- Evaluated forecasting baselines and machine-learning candidates over six rolling, non-overlapping 14-week origins. Two origins were used for tuning and four later origins for held validation.
- Selected model configurations using tuning data only. Final-test labels have not been opened.
- Evaluated an additional Random Forest candidate after the initial results showed a gap on intermittent and lumpy demand.
- Corrected lead-time scoring so incomplete forecast windows are excluded rather than inadvertently treated as zero forecasts. The largest resulting WAPE adjustment was approximately **0.0024 percentage points**.
- Ran paired model comparisons and bootstrap intervals for leading candidates. These are limited by four validation origins, dependent observations, and the synthetic dataset.

## Results to date

### Forecast accuracy on held validation

| Candidate | Weekly WAPE | Lead-time WAPE | Mixed absolute bias |
|---|---:|---:|---:|
| Equal-weight ensemble | 69.26% | 37.90% | 2.43% |
| LightGBM Tweedie | 63.25% | 34.31% | 8.81% |
| LightGBM quantile | 57.71% | 36.20% | 26.78% |

The table shows different trade-offs, not a definitive best model. The ensemble has lower absolute bias than the two LightGBM models. Tweedie has lower lead-time WAPE in this sample. Quantile LightGBM has the lowest weekly WAPE among these three and is the only one producing probabilistic quantiles.

The quantile model's measured lead-time P90/P95/P99 coverage was **93.09% / 97.41% / 99.64%**. This measures forecast calibration; it does not by itself demonstrate that parts will be available or stock-outs will decrease.

### Scoring-policy comparison

The existing baseline weights remain unchanged:

| Metric | Baseline weights | Proposed business scenario |
|---|---:|---:|
| WAPE | 35% | 20% |
| Absolute bias | 15% | 15% |
| A-class / critical-parts forecast error | 15% | 20% |
| Intermittent / lumpy forecast error | 10% | 20% |
| Calibration | 10% | 8% |
| Stability across validation origins | 5% | 7% |
| Runtime | 5% | 3% |
| Explainability | 5% | 7% |
| **Total** | **100%** | **100%** |

The proposed weights reflect a possible emphasis on critical parts and intermittent/lumpy demand. They are a business proposal, not a research-proven or approved optimum.

| Scenario | Nominal top candidate | Weighted score | Sensitivity cases where top candidate changes |
|---|---|---:|---:|
| Existing baseline weights | Equal-weight ensemble | 71.458 | 6 of 16 |
| Proposed business weights | Equal-weight ensemble | 71.042 | 8 of 16 |

The proposed scenario moves LightGBM Tweedie from third to second place; its score is only **0.083** points behind the ensemble. Under the required one-at-a-time ±10 percentage-point tests, the proposed weights are less stable than the baseline. Neither scenario meets the current stability condition for selecting a champion. Scores are rank-normalized selection scores, not forecast percentages or savings.

## Business outcomes not yet established

The current forecasting evaluation directly measures weekly and lead-time demand error, bias, A-class/critical and intermittent/lumpy forecast error, probabilistic calibration where available, runtime, and error variation across origins.

It does **not** yet measure:

- Critical-part availability, stock-out reduction, or fill/service rate versus targets.
- Emergency procurement count or cost.
- Excess inventory, carrying cost, or inventory turnover.
- Schedule adherence or supplier/production capacity feasibility.
- End-to-end results comparing the forecast-and-scheduler chain with a traditional baseline.
- Real-plant savings or causal operational impact.

These outcomes require a tested simulator and scheduling comparison using identical demand paths and fixed assumptions. A-class/critical forecast error is a forecasting measure, not a direct measure of critical-part availability.

## Tasks remaining

1. **Resolve forecast selection evidence.** Keep existing weights frozen. Review the sensitivity findings with business stakeholders and obtain evidence-based priorities or additional validation evidence. Do not choose a model merely because it is the nominal leader.
2. **Select and freeze a forecast candidate only if justified.** The current score-sensitivity results do not support declaring a champion. No final 14-week forecast has been generated.
3. **Define and freeze simulation assumptions.** Confirm inventory initialization, backorder/lost-sales handling, supplier delay and rejection behavior, emergency procurement rules, cost proxies, and plant-calendar treatment.
4. **Implement and validate the scheduling engine.** Check supplier/production capacity, lead time, MOQ, order multiples, and any contractual share-of-business rules. Infeasible schedules must be reported as infeasible.
5. **Evaluate scheduling business KPIs.** Compare scheduling approaches on identical demand paths, including stock-outs, fill/service level, emergency procurement, excess and carrying cost, turnover, schedule adherence, critical-part availability, stability, runtime, and hard-constraint violations.
6. **Evaluate the full forecast-to-schedule chain** against a traditional baseline, using the same simulation assumptions and demand path.
7. **Build and test the controlled assistant and application.** The selected design is React/TypeScript → one Django/DRF backend → MySQL, with forecasting and scheduling imported in-process. The assistant should interpret requests and call approved functions; deterministic scheduling logic must calculate quantities. Authentication, authorization, confirmation, cancellation, and audit behavior still require implementation and testing.
8. **Run the final evaluation once** only after prior decisions and changes are complete. Final-test labels remain sealed until that evaluation.

## Current quality status and recommendation

- Full automated tests: **39 passed** under Python 3.10 and **39 passed** under Python 3.13.
- Dataset validation: **32 passed, 0 failed**.
- Original source datasets remain unchanged.
- Final-test labels remain unopened.
- No forecast champion, future forecast file, scheduler result, or assistant outcome is being claimed.

**Recommendation:** retain the existing scoring weights as the audit baseline, but do not declare a winning forecast model yet. The proposed weighting is not more stable in the tested scenarios. Continue only after the model-selection evidence and business priorities are reviewed; scheduler and operational KPIs remain to be measured.

## Supporting project documents

- [`SPEC.md`](../SPEC.md) — project requirements, evaluation rules, and safeguards.
- [`STATE.md`](../STATE.md) — detailed execution record and current checkpoint.
- [`phase0_environment.md`](./phase0_environment.md) — environment and dataset facts.
- [`phase0_traceability.md`](./phase0_traceability.md) — requirements traceability and discrepancies.
- [`data_quality_report.md`](../results/data_quality_report.md) — data checks and joins.
- [`phase2_eda_report.md`](../results/phase2_eda_report.md) — demand exploration and feature safeguards.
- [`phase3_track_a_report.md`](../results/phase3_track_a_report.md) — rolling forecast evaluation.
- [`phase4_track_a_report.md`](../results/phase4_track_a_report.md) — forecast scoring and selection blocker.
- [`track_a_weight_scenario_report.md`](../results/track_a_weight_scenario_report.md) — baseline and proposed weight comparison.
