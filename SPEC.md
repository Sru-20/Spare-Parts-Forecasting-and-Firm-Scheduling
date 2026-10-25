# Spare Parts Demand Forecasting & Firm Scheduling — Project Contract

**Contract version:** aligned to `docs/Spec.md` SPEC v4.1 (2026-10-25 update).  
**Governing documents:** `docs/Spec.md`, `docs/SPEC.docx`, the available problem-statement material, verified repository data/code, and `reports/project_progress_report.md`. If sources conflict, record the discrepancy and follow the verified files.  
**Claim scope:** the supplied data are synthetic; describe results as **method validated on simulated data**, never as proven savings.

This file records project-specific verified decisions and implementation status. The fuller feature, scheduling, assistant, API, security, and release contract is in `docs/Spec.md` and remains binding. Changes to the governing contract require a reason and evidence in `STATE.md`.

## 1. Verified project facts and discrepancies

- D1 `data/01_demand_forecasting.csv`: 120,861 rows × 62 columns; part-region-week demand; 2023-01-02 to 2026-12-28.
- D2 `data/02_scheduling_inputs.csv`: 158 × 61; one part-level inventory/supplier snapshot as of 2026-09-26.
- D3 `data/03_spare_parts_catalogue.csv`: 160 × 29.
- D1 splits are defined by `split_role`: warm-up through 2024-01-01; train/validation through 2026-03-23; final test 2026-03-30 to 2026-09-21; 14 future weeks 2026-09-28 to 2026-12-28.
- D1 contains 616 part-region series; 72 historical missing labels; 54,511 genuine zero-demand historical observations. D3-only obsolete parts P0046 and P0087 have no D2 snapshot.
- The supplied training-period SBC classification leaves 47 series short/unclassified; Phase 2 uses an analogue fallback for prediction without fabricating their class.
- The source README references `docs/Spare_Parts_Forecasting_Problem_Statement.pdf`, which is absent. The latest `docs/` inventory contains `Spec.md` and `SPEC.docx`; both are SPEC v4.1, not copies of that missing problem-statement PDF. Previously referenced PDFs are not present in the current document directory. This discrepancy is retained; no source document is invented.
- `docs/Spec.md` and `docs/SPEC.docx` specify the updated v4.1 contract. Its progress-report reference is `reports/project_progress_report.md` (submitted as a review document); business priorities and cost assumptions remain subject to stakeholder confirmation.
- Dataset validator result: **32 PASS, 0 FAIL**. Original data are not modified.

## 2. Architecture and scope

- Target architecture: React + TypeScript → one Django + DRF backend → one relational database (SQLite for development; MySQL or PostgreSQL for production).
- Forecasting, scheduling, simulation, and evaluation are importable Python packages called in-process by Django. No second backend, queue, or additional service unless measured need is documented.
- Work in order: planning core, web application, controlled multilingual assistant. Each scope needs its own tests and gate.
- Batch forecasts/schedules/KPIs are persisted and versioned by `run_id`; read dashboards do not run inference. Retraining is offline and champion promotion is gated.

## 3. Data, leakage, and forecast contract

- Fit and score on raw `demand_qty`. The precomputed `demand_qty_capped` uses a broad train-validation distribution and is not fold-safe. Any capped-target comparison must be validation-only and fit its cap within each training fold.
- `is_missing_week = 1` with a blank label is excluded from fitting/scoring, never filled with zero. Genuine zeros remain zero. Future rows are prediction-only.
- Never use `out_*`, `intended_sbc_class`, or `demand_driver_tags` as predictors. Exclude `lifecycle_pattern` unless origin-time availability is proven. Recompute lags, rolling features, segment labels, encoders, and scalers from origin-visible data.
- Use only known-at-origin covariates. Festival calendars, plant calendars/shutdowns, and promotions/campaigns are allowed only when published by the origin; unplanned market events and realized future indices are unavailable.
- Warm-up is history/lag initialization only. Training labels are observed `train_val` only. Use rolling-origin validation and seed 42. Keep final-test labels unopened until their one-time final evaluation.
- Required forecast family: naïve, seasonal naïve, the better validation performer of SBA/TSB, global LightGBM point (Tweedie or Poisson selected on validation), and LightGBM quantile P50/P90/P95/P99. Preserve all completed Phase 3 results, including additional candidates; ensemble/router/other candidates require repeatable validation benefit to remain in scope.
- Point output supports netting/MRP/KPIs; quantiles support safety stock and warnings. Ensure `P50 <= P90 <= P95 <= P99`; any correction must be documented and evaluated on validation.
- Report required forecast metrics weekly and lead-time aggregated, overall and by SBC, ABC, criticality, region, category, and horizon step. Report coverage by criticality and horizon with counts; small samples are labelled insufficient.

## 4. Forecast metrics and units

The v4.1 contract supersedes legacy percent-valued metric files for new comparisons:

- Store error ratios internally. For example, WAPE ratio `0.6926` displays as `69.26%`.
- WAPE = `sum(abs(actual - forecast)) / sum(actual)`; undefined when actual sum is zero and excluded from averages.
- Signed bias ratio = `sum(forecast - actual) / sum(actual)`; positive is over-forecast; undefined for zero actual total.
- Asymmetric bias = `abs(bias)` for nonnegative bias; `under_weight * abs(bias)` for negative bias. Default `under_weight=2` is a configurable scenario, not an observed cost.
- Forecast Accuracy % = `100 * (1 - WAPE ratio)`, shown beside WAPE. It can be negative and is not a probability.
- MAE is `mean(abs(actual - forecast))`; RMSE is `sqrt(mean((actual - forecast)^2))`.
- sMAPE is the mean of `200 * abs(actual - forecast) / (abs(actual) + abs(forecast))`; a zero/zero term contributes zero.
- MASE is MAE divided by training-period in-sample one-step naïve MAE; an all-zero/undefined training scale is undefined and counted.
- Tracking signal is `sum(forecast - actual) / MAE`; it is zero when both numerator and MAE are zero, otherwise undefined when MAE is zero.
- Lead-time demand aggregates forecast and actual over `ceil(lead_time_days/7)` weeks from the origin. For lead time >14 weeks, use the 14-week forecast plus the explicitly labelled last-period-average extension.
- For the three parts with lead times beyond 14 weeks, define “last period” as the final four weeks of the 14-week validation horizon. Extend both actual and forecast sums by that period’s mean times the over-horizon weeks; this is a project assumption, not a measured forecast.
- Forecast-versus-actual variance reports signed/absolute differences and the population variance of `forecast - actual` errors.
- Quantile coverage is `mean(actual <= quantile forecast)`; pinball loss is `mean(max(q * (actual - forecast_q), (q - 1) * (actual - forecast_q)))`. Coverage and pinball loss are reported at weekly and lead-time levels; quantile crossing is checked.
- WAPE, bias, and sMAPE are stored as ratios; accuracy is displayed in percent. MAE, RMSE, pinball loss are in demand units; variance is in squared demand units; tracking signal and MASE are dimensionless.
- Every KPI/API value must identify its basis (`measured`, `simulated`, `proxy`, or `not_measurable`).

## 5. Selection and current Track A assumptions

Selection order is mandatory: (1) configurable screening rules with uncertainty, (2) weighted rank score among eligible candidates, (3) sensitivity and tie-break/downstream simulation.

Default screening rules:

1. Beat both seasonal-naïve and historical-average baselines on lead-time WAPE.
2. Overall absolute signed bias no greater than 10%.
3. Critical-part under-forecast bias no worse than -10%.
4. For quantile candidates, P95 coverage on critical parts within ±5 percentage points.

Report bootstrap intervals by resampling series and origins together. With four origins these are screening checks with limited evidence. `gates.enforce` defaults to enforcing only baseline-beating; other rules are flagged, not rejected, when the bootstrap interval includes the threshold. If evidence is inconclusive, prefer the simpler model; do not claim significance without interval support. Diebold–Mariano is optional.

Track A weights are editable project assumptions from SPEC v4.1:

| Component | Weight | Scoring definition |
|---|---:|---|
| Overall error | 20% | 70% lead-time WAPE, 30% weekly WAPE |
| Critical/A-class error | 20% | Lead-time WAPE plus asymmetric bias |
| Intermittent/lumpy | 15% | Lead-time WAPE and MASE, 50/50 |
| Asymmetric absolute bias | 15% | Under-forecast penalty uses configured factor |
| Calibration | 15% | Coverage error plus pinball; no quantiles get minimum score |
| Stability across origins | 10% | Lower variation is better |
| Runtime and explainability | 5% | Combined as configured |
| **Total** | **100%** | |

Critical asymmetric bias deliberately overlaps the general asymmetric-bias component. Run the required one-at-a-time ±10-point sensitivities and one additional sensitivity scenario with the overlap removed. A champion is supported when it wins at least 75% of perturbations; otherwise report contenders with uncertainty and use downstream validation simulation for the tie-break. Forecast error alone does not determine the best planning model.

Track B weights: critical availability 25%, overall fill/stock-out 15%, emergency procurement 15%, excess inventory/carrying cost 20%, schedule adherence/stability 15%, inventory turnover 10%. Any hard-constraint violation disqualifies a scheduler.

Business defaults in `configs/business.yaml` are assumptions pending stakeholder input: emergency cost = unit cost × 1.5 (test 1.25–2.0); stock-out penalty = unit cost × 10/5/2 for High/Medium/Low (vary ±50%); carrying rate uses supplied value; downtime cost is not measurable. Report whether the winner changes over these ranges. Service targets come from D2.

## 6. Scheduling and assistant safety

- Build deterministic MRP first; add an (s,S)/quantile variant if useful. Add CP-SAT only if MRP results show a measured need.
- The scheduler and simulator share the frozen weekly inventory order-of-operations and conservative order-arrival rules in `docs/Spec.md`.
- Independently validate MOQ, multiples, capacity, lead time, inventory conservation, obsolete status, and production/import restrictions. Label plans `firm` only when every hard check passes; otherwise `unresolved` with reasons. Never invent replacement quantities.
- Compare forecast/scheduler pairs on identical validation demand paths. Call outputs simulated firm schedules, not purchase orders or operational commitments.
- Ask AI starts read-only, then typed proposals only. AI interprets and calls approved functions; deterministic server code validates and computes all quantities. Explicit authorized confirmation, stale-proposal rejection, audit, cancellation, and safe revert are mandatory. Incorrect auto-apply and invented-quantity rates must be zero.
- Initial assistant language set: English, Hindi, Hinglish, Marathi; Tamil optional later.

## 7. Execution gates and status

- Read this contract and `STATE.md` at every session start. Work one gate at a time; run `pytest` at each gate.
- At each checkpoint record completed work, changed files, commands/results, assumptions, discrepancies, open issues, and next executable command in `STATE.md`.
- Phases 0–3 have been reverified against v4.1. Phase 4 Track A selection has a sensitivity-stable champion and its exit gate is complete. A train-validation-only recursive projection to the future dates is available for planning context, but it bridges the sealed test-date gap and is not a directly validated 14-week forecast; the operational forecast requires the Phase 7 final-test evaluation/refit. The final test remains sealed. Phase 5 is next.
- Never fabricate metrics or claim real-world savings. Missing evidence is explicitly reported as not measured, proxy, or blocked.

## 8. Change log

- **2026-10-25:** Reconciled this project contract to `docs/Spec.md` SPEC v4.1 and `docs/SPEC.docx`. New v4.1 selection rules, Track A/B weights, metric units, gate behavior, forecast scope, simulator contract, and business cost scenarios supersede the previous v3 contract. Existing Phase 0–3 artifacts and prior Phase 4 results are retained as historical evidence; they are not silently relabelled as v4.1-compliant. Reverification and v4.1 Phase 4 results are recorded in `STATE.md`.
- **2026-10-25:** Defined the lead-time extension period as the final four forecast weeks and recorded its use as an assumption. Phase 4's v4.1 ranking, bootstrap screening, sensitivity, origin-availability decision, and recursive projection limitation are recorded in `STATE.md`.
