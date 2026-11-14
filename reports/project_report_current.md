# Current project report — SPEC v4.2

**Project:** AI-based spare-parts demand forecasting and firm scheduling
**Data:** synthetic, seed 42
**Claim scope:** method validated on simulated data. This report does **not** claim factory savings, a production forecast guarantee, or historical schedule accuracy.
**Prepared:** 2026-11-14
**Status:** Phases 0–5 complete; SPEC v4.2 items A1–A4 and B complete; A5 deferred by its own gate; Phase 6 comparison and Phases 7–9 not started. Final test remains sealed.

---

## 1. Governing documents and document control

| Item | Current state |
|---|---|
| Governing specification | [`docs/SPEC.md`](../docs/SPEC.md), titled **SPEC v4.2 (FINAL, supersedes all earlier versions)** |
| Config version | `configs/project.yaml` records `specification.version: "4.2"` and `specification.document: docs/SPEC.md` |
| Root `SPEC.md` | Not present. The spec's working instructions say to keep `SPEC.md` in the repository root; the active copy is in `docs/`. This is a document-control discrepancy, not a modelling result. Decision required: move it to root or accept `docs/SPEC.md` as the controlled location. |
| `docs/SPEC_AMENDMENT.md` | A separate **v4.3** amendment (planner explainability, operational clarity, execution safeguards). It states that it supplements `SPEC_FINAL_v4_3.md`, but **no v4.3 baseline exists in this project**. The file says to merge it into the single authoritative spec only after review. It is not accepted or implemented. Decision required: accept (which also requires the v4.3 baseline), reject, or defer. |
| `docs/phase5_assumptions.md` | Restored from repository history because the Phase 5 test requires it. Its recorded config SHA-256 hashes were refreshed after the v4.2 config edits. |
| Old six-month benchmark | The earlier 3-origin outputs and the previous mentor report/plan were moved out of the project to `../spare-parts-archive-2026-11-14/`. They are superseded by the v4.2 interim backtest. |

No other specification is treated as active. There is no assumption that a missing document exists.

---

## 2. Status at a glance

| Area | Status | Evidence |
|---|---|---|
| Phase 0 contract and environment | Complete | `reports/phase0_traceability.md`, `reports/phase0_environment.md`, contract tests |
| Phase 1–2 data and EDA | Complete | `reports/data_dictionary.csv`, `results/phase2_*` |
| Phase 3 rolling-origin backtest | Complete | `results/phase3_track_a_report.md`, `artifacts/track_a_backtest_forecasts_v41.parquet` |
| Phase 4 v4.1 champion selection | Complete; champion frozen | `results/phase4_track_a_v41_report.md`, `results/track_a_v41_*` |
| Phase 5 MRP and simulator | Complete | `results/phase5_*`, `docs/phase5_assumptions.md` |
| Bias calibration experiment | Closed; not adopted | `results/bias_calibration_report.md` |
| Structural model experiments | Closed; no candidate admitted | `results/model_experiments_report.md` |
| **A1** moving average, ETS, XGBoost comparison | Complete; gate not met | §5; `results/model_comparison_*` |
| **A2** part-category classifier | Complete; weak out-of-component evidence | §6; `results/part_category_*` |
| **A3/A4** supplier risk and reorder points | Complete; decision needed on policy | §7; `results/phase6_supplier_risk_*` |
| **A5** optional small language model | Deferred by its gate | §8 |
| **B** interim 26-week backtest | Complete | §9; `results/interim_26w_backtest.csv`, `results/interim_26w_report.md` |
| Phase 6 scheduler comparison | Not started | — |
| Phase 7 final test | Sealed | Not opened |
| Phase 8–9 Ask AI and delivery hardening | Not started | — |

---

## 3. Forecasting champion and prior evidence

The frozen champion is **`lightgbm_tweedie::random_1`** with:

`num_leaves=63`, `learning_rate=0.03`, `n_estimators=300`, `min_child_samples=20`, `feature_fraction=1.0`, seed 42.

| Evidence | Weekly WAPE / bias | Lead-time WAPE / bias |
|---|---|---|
| Tuning origins (2024-07-01, 2024-10-07) | 63.11% / −18.02% | **39.29% / −16.93%** |
| Held origins (2025-01-13, 2025-04-21, 2025-07-28, 2025-11-03) | 63.25% / −15.11% | **34.90% / −5.06%** |

The champion under-forecasts. Held lead-time bias is −5.06%; weekly bias is −15.11%. The bias-calibration experiment tried seven multiplicative variants and all failed the pre-declared gates; the structural experiment tried a blend, a Poisson objective and trend features, and none cleared the 0.50 pp tuning gate. The champion remains uncorrected.

The four held origins have now been looked at three times: Phase 4 selection, bias calibration, and Experiment 2. They are **not independent** for further exploratory variants. A genuine improvement claim needs a new independent period.

---

## 4. A1 — broader model comparison

### Design

- Tuning origins only: 2024-07-01 and 2024-10-07; fixed 14-week horizon.
- New candidates, same features, same recursive one-step strategy, seed 42:
  - `moving_average_13`;
  - `exponential_smoothing` (ETS(A,N,N), α fitted at each origin, bounded 0.01–0.99);
  - `xgboost` with the same three seed-42 configurations as the frozen LightGBM search, mapped as `num_leaves → max_depth = ceil(log2(num_leaves))`, `min_child_samples → min_child_weight`, `feature_fraction → colsample_bytree`.
- Reported references: `naive`, `seasonal_naive_52`, `tsb`, `historical_average`.
- The champion was read from the frozen tuning-forecast artifact; it was not refitted or changed.
- No held origin was opened because the tuning gate did not pass.

### Tuning results — lead-time windows

| Candidate | WAPE % | Bias % | MASE | MAE | RMSE | MAPE % (non-zero) | Non-zero n | Zero actuals excluded |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **lightgbm_tweedie::random_1 (champion)** | **39.29** | **−16.93** | 2.329 | 38.83 | 148.68 | 69.81 | 821 | 301 |
| tsb (reference) | 44.48 | −12.40 | 2.637 | 43.95 | 152.30 | 101.18 | 821 | 301 |
| moving_average_13 | 45.07 | −12.16 | 2.672 | 44.54 | 153.54 | 113.42 | 821 | 301 |
| historical_average (reference) | 46.08 | −18.36 | 2.732 | 45.53 | 162.04 | 98.37 | 821 | 301 |
| xgboost::random_1 (best XGBoost) | 46.12 | −4.89 | 2.734 | 45.58 | 156.51 | 118.58 | 821 | 301 |
| exponential_smoothing | 46.14 | −9.13 | 2.735 | 45.59 | 156.68 | 111.28 | 821 | 301 |
| seasonal_naive_52 (reference) | 51.29 | −14.98 | 3.039 | 50.68 | 162.81 | 138.19 | 821 | 301 |
| naive (reference) | 61.81 | −1.34 | 3.665 | 61.08 | 182.97 | 159.84 | 821 | 301 |

Weekly results are in `results/model_comparison_tuning.csv`. The champion also leads weekly WAPE (63.11% vs 69.07% for moving average, 70.32% for ETS, 74.59% for the best XGBoost).

**Gate outcome:** best new candidate is moving average at 45.07%, i.e. **5.78 pp worse** than the champion. The required gain is +0.50 pp. **No held evaluation was opened and no held-origin look was added.** Decision: `champion_retained_tuning_gate_not_met`.

**Deferred models and reasons:**

- SARIMA/ARIMA: 616 mostly intermittent series, synthetic data and a 14-week horizon make per-series fits costly without decision value.
- LSTM: no GPU, small synthetic sample, and the recursive tabular challengers already cover the non-linear comparison at lower cost.

MAPE is reported with its zero-exclusion count and is not scored. RMSE is dominated by a small number of lumpy spikes; MAE and WAPE are the more stable comparisons here.

---

## 5. A2 — part-category classifier

### Design

- Target: catalogue `part_category`.
- Text fields: part name, description, component, material, vehicle system, vehicle model, vehicle type.
- Models: TF-IDF character 2–5-grams with logistic regression, and TF-IDF with a linear SVM (calibrated).
- Validation: stratified 5-fold cross-validation and a component-group hold-out (25%), so no component appears in both train and test.
- Outputs add `predicted_category`, `confidence` and `needs_review`, but **never overwrite the catalogue category**. `resolve_category()` always prefers the catalogue value.

### Measured results

| Model | Stratified CV macro-F1 | Component hold-out macro-F1 | Hold-out weighted F1 | Hold-out accuracy |
|---|---:|---:|---:|---:|
| TF-IDF + logistic regression | 1.000 | 0.154 | 0.435 | 48.78% |
| TF-IDF + linear SVM | 1.000 | **0.210** | **0.469** | **56.10%** |

- Hold-out size: 41 parts from 11 unseen components; component overlap = 0.
- Only 5 of 8 categories appear in the hold-out. The SVM predicts Ferrous and Electrical perfectly (recall 1.0), Mechanical at 0.33 recall, and does not predict Body, Electronics, Non-ferrous, Polymer or Rubber.
- `needs_review` at threshold 0.60: **0 parts**. The threshold is provisional and editable; a higher threshold would flag more rows.
- The CV score of 1.000 is **optimistic** because the generator likely derives category from names/components. It is not evidence of generalization. The component hold-out is the honest number.
- The catalogue has no `material_group` field. Under the "if the catalogue has it" rule, **Material group was not inferred** and is marked `not_available_in_catalogue`.
- Confidence is not calibrated for decision-making; this is a review aid, not an automatic classifier.

Outputs: `results/part_category_predictions.csv`, `_metrics.csv`, `_per_class.csv`, `_confusion_matrix.csv`, `_report.md`, `_summary.json`.

---

## 6. A3/A4 — supplier risk and reorder points

### Method

- Supplier risk score = `1 − supN_reliability_pct/100`; risk level = population tertiles of the SOB-weighted part risk score (High/Medium/Low).
- Risk-adjusted planning lead time = `lead_time_days + (1 − OTD%) × average_delay_days`.
- Required quantity is grossed up by `1 / (1 − reject_rate%)`, then MOQ/order-multiple rounding.
- Reorder point = lead-time point demand + safety stock; trigger = first week with projected inventory at or below the reorder point.
- The traditional baseline ignores supplier risk **by design**.
- Both variants ran on identical demand paths and the same D2 snapshot; the simulator was not rerun for the risk variant.

### Risk levels

High: 53 parts; Medium: 52; Low: 53.

### Ablation (563 part-origin paths)

| Metric | Traditional baseline | Risk-adjusted | Change |
|---|---:|---:|---:|
| Unresolved part-origins | 121 | 130 | +9 |
| Order lines | 6,718 | 6,342 | −376 |
| Order quantity | 766,550 | 788,666 | **+2.89%** |
| Expected rejected quantity | 23,022 | 23,825 | +803 |
| Mean safety stock | 776.6 | 801.8 | +25.2 |
| Projected unmet units | 225,978 | 243,181 | **+17,203** |
| High-risk supplier large-share warnings | 1,582 | 1,258 | −324 |

**Interpretation, stated plainly:** risk-adjusted planning orders more and later. It does **not** improve projected service in this run; projected unmet units rise because expected lead times are longer. It also does not reduce expected rejection loss. This is a planning-exposure comparison, not a savings claim and not a service improvement. A policy decision is required before using risk-adjusted lead times in the firm schedule.

The 50% high-risk large-share warning threshold is provisional and editable.

### Reorder points versus D2 current policy

| Criticality | Median computed ROP | Median D2 current ROP | Above current | Below current |
|---|---:|---:|---:|---:|
| High | 587.7 | 54.0 | 187 | 2 |
| Medium | 267.1 | 34.0 | 233 | 7 |
| Low | 195.8 | 28.0 | 128 | 6 |

Computed reorder points are far above D2's `current_reorder_point_qty`. Either D2 represents a much leaner current policy or its values are placeholder/snapshot values; this is **not established** from the available data. The comparison is reported, not reconciled.

Outputs: `results/phase6_supplier_risk_weekly.csv`, `_orders.csv`, `_plans.csv`, `_comparison.csv`, `_summary.json`, `_report.md`.

---

## 7. A5 — optional small language model

Deferred under the amendment's own rule. Phase 8 Ask AI does not exist yet, no approved ≤3B instruct model is installed in the environment, and there is no GPU. The safety gate cannot be evaluated before Ask AI safety tests exist. No model was downloaded, installed or run. This is a **deferral with reason**, not a failed experiment.

---

## 8. B — interim 26-week backtest

### Windows and design

| Origin | Scored window |
|---|---|
| 2025-03-24 | 2025-03-31 to 2025-09-22 |
| 2025-09-22 | 2025-09-29 to 2026-03-23 |

- Two consecutive 26-week blocks; the second ends at the train-validation cutoff.
- Origin-safe: each window is fitted only on labels up to its origin; recursive one-step features are rebuilt inside the origin.
- Frozen hyperparameters from the Phase 3 tuning log; no re-tuning.
- Final test excluded; it was not read or scored.
- Models: champion Tweedie, quantile, nine-model equal-weight ensemble, best intermittent baseline (`tsb`), naive, seasonal naive (52), historical average.
- Forecast Accuracy % = 100 − WAPE %.

### Weekly results (pooled across both windows)

| Model | WAPE % | Accuracy % | Bias % | MASE | n scored |
|---|---:|---:|---:|---:|---:|
| lightgbm_quantile | 60.12 | 39.88 | −34.15 | 0.707 | 29,950 |
| **lightgbm_tweedie (champion)** | **66.15** | **33.85** | **−14.03** | **0.778** | 29,950 |
| simple_average_ensemble | 71.02 | 28.98 | −5.79 | 0.836 | 29,950 |
| tsb | 73.77 | 26.23 | −1.55 | 0.869 | 29,566 |
| historical_average | 78.06 | 21.94 | −2.90 | 0.919 | 29,566 |
| naive | 88.71 | 11.29 | +4.97 | 1.045 | 29,566 |
| seasonal_naive_52 | 90.88 | 9.12 | −1.71 | 1.070 | 29,566 |

### Complete 26-week part-region aggregate

| Model | WAPE % | Accuracy % | Bias % | Aggregate series | Incomplete series excluded |
|---|---:|---:|---:|---:|---:|
| **lightgbm_tweedie (champion)** | **33.05** | **66.95** | **−14.20** | 1,116 | 65 |
| simple_average_ensemble | 35.86 | 64.14 | −6.55 | 1,116 | 65 |
| tsb | 38.20 | 61.80 | −3.11 | 1,116 | 65 |
| lightgbm_quantile | 41.90 | 58.10 | −34.19 | 1,116 | 65 |
| historical_average | 42.21 | 57.79 | −4.92 | 1,116 | 65 |
| seasonal_naive_52 | 44.76 | 55.24 | −1.79 | 1,116 | 65 |
| naive | 72.92 | 27.08 | +5.40 | 1,116 | 65 |

### Horizon bands (pooled weekly)

| Model | Step 1–13 WAPE % | Step 14–26 WAPE % |
|---|---:|---:|
| lightgbm_quantile | 60.37 | 59.88 |
| lightgbm_tweedie | 66.83 | 65.52 |
| simple_average_ensemble | 72.01 | 70.11 |
| tsb | 74.63 | 72.97 |
| historical_average | 78.48 | 77.66 |
| naive | 91.75 | 85.88 |
| seasonal_naive_52 | 93.69 | 88.27 |

SBC-class and criticality breakdowns are in the CSV and report. Sample counts and exclusions:

- 29,950 weekly rows scored for the three global models; 29,566 for baselines.
- 22 rows excluded for missing actuals.
- 384 rows excluded from baselines because the series has no usable history; the global models still produced forecasts.
- 65 part-region series excluded from the complete 26-week aggregate because they lack 26 observed weeks and 26 finite forecasts.

### Reading the result

- `lightgbm_quantile` has the lowest weekly absolute error at the median but **understates totals by 34.15%**. It is not a replacement for the champion.
- The champion has the lowest complete 26-week aggregate WAPE and a smaller bias than the quantile model.
- The interim benchmark is **not used for champion selection**. The champion stays frozen.

### Caveats

- Only two windows; no significance test is claimed.
- The summed 26-week aggregate is easier than weekly and is **not comparable** to the 14-week supplier lead-time metric.
- These windows lie inside held validation and overlap forecast horizons already used in Phase 3; they are **not independent** evidence.
- The final test remains sealed.
- Data are synthetic; accuracy here supports method validation, not a factory guarantee.

Outputs: `results/interim_26w_backtest.csv`, `results/interim_26w_report.md`, `results/interim_26w_summary.json`.

---

## 9. Scheduling and simulation evidence (Phase 5)

| Metric | Value |
|---|---:|
| Complete part-origin paths | 563 |
| Supplier order lines | 6,718 |
| MRP plan weeks / simulation weeks | 7,882 / 7,882 |
| Feasible schedules | 442 |
| Unresolved schedules | 121 |
| Independent hard-constraint violations | 0 |
| Service fill rate | 71.55% |
| Stockout part-week share | 20.38% |
| Emergency orders / quantity | 534 / 45,371 |
| Emergency cost proxy | 104,895,525 |
| Carrying cost proxy | 22,253,439 |
| MRP / simulator inventory-ledger failures | 0 / 0 |

Limits that must be stated with these numbers:

- D2 is **one snapshot** dated 2026-09-26, not a ledger. There is no historical receipt or order-date series.
- The same snapshot initialises every origin; there is no schedule warm-up.
- D2 has no individual purchase-order IDs or receipt dates.
- Costs are proxies, not company accounts.
- **Historical schedule accuracy cannot be claimed** because there are no company schedule logs. Only forecast accuracy can be measured.

---

## 10. Decisions required

1. **Specification location:** move `docs/SPEC.md` to the root, or accept `docs/SPEC.md` as the controlled copy.
2. **v4.3 amendment:** accept, reject, or defer `docs/SPEC_AMENDMENT.md`. It is not implemented, and it names a `SPEC_FINAL_v4_3.md` baseline that is not in the project, so accepting it requires that baseline first.
3. **A5 small language model:** keep deferred, or approve hardware/model access after Phase 8 safety tests exist.
4. **A2 classifier use:** keep predictions as reviewer-only (current behaviour) or invest in more labelled components. The component hold-out macro-F1 is 0.21; it is not safe to auto-apply.
5. **Supplier-risk policy:** accept risk-adjusted lead times and rejection gross-up despite higher projected unmet units, or keep the traditional baseline. This is a policy choice, not a modelling result.
6. **Reorder-point policy:** decide whether D2 `current_reorder_point_qty` is a real current policy. Computed ROPs are several times higher; the data cannot establish which is correct.
7. **Final test:** Phase 7 opens only after sign-off. It remains sealed now.

---

## 11. Limits and claim scope

- All data are synthetic. Every result means "method validated on simulated data".
- D2 is a single snapshot; supplier reliability, reject rates and delays are synthetic inputs, not calibrated operating rules.
- Four held origins have been reused three times. They are no longer independent for new variants.
- The interim 26-week backtest has two windows and no significance test.
- MAPE is unstable on intermittent demand; zero-actual rows are excluded and counted.
- RMSE is spike-sensitive; WAPE, MAE and MASE are the primary comparisons.
- No production cost, service-level or savings claim is supported.
- The final test has not been read.

---

## 12. Verification and reproducibility

| Check | Result |
|---|---|
| `python -m pytest -q` | **119 passed** |
| `python data/validate_datasets.py data` | **32 PASS, 0 FAIL** |
| `git diff --check` | **clean** |
| Final-test isolation | Confirmed by data split filters and tests |

Reproduce the new work:

```bash
python -m src.forecasting.model_comparison
python -m src.classification.part_category
python -m src.scheduling.risk_reorder
python -m src.forecasting.interim_backtest
```

Supporting tests:

```bash
python -m pytest -q tests/test_model_comparison.py tests/test_part_category.py \
  tests/test_risk_reorder.py tests/test_interim_backtest.py
```

---

## 13. Deliverables added or updated in this pass

| Path | Purpose |
|---|---|
| `src/forecasting/model_comparison.py` | A1 tuning comparison and advancement gate |
| `results/model_comparison_tuning.csv` | A1 metrics with MAE/RMSE/MAPE and exclusions |
| `results/model_comparison_report.md` | A1 report and decision |
| `results/model_comparison_summary.json` | A1 machine-readable decision |
| `src/classification/part_category.py` | A2 classifier |
| `results/part_category_*` | A2 predictions, metrics, per-class, confusion matrix, report |
| `src/scheduling/engine.py` | A3/A4 risk-adjusted planning and reorder points |
| `src/scheduling/risk_reorder.py` | A3/A4 ablation runner |
| `results/phase6_supplier_risk_*` | A3/A4 outputs |
| `src/forecasting/interim_backtest.py` | B interim 26-week backtest |
| `results/interim_26w_backtest.csv` | B metric table |
| `results/interim_26w_report.md` | B report with required caveats |
| `tests/test_model_comparison.py`, `test_part_category.py`, `test_risk_reorder.py`, `test_interim_backtest.py` | Regression tests |
| `docs/phase5_assumptions.md` | Restored Phase 5 evidence; hashes refreshed |
| `requirements.txt` | Added the XGBoost dependency used by A1 |

Removed from the project (archived outside it at `../spare-parts-archive-2026-11-14/`): the previous mentor reports (`project_report_remaining_work.md/.docx` and the 23 Oct 2026 `project_progress_report.md`), the superseded completion plan, and the old 3-origin six-month outputs. Generated caches (`__pycache__`, `.pytest_cache`, docx lock files) were deleted. Data, frozen artifacts, Phase 3–5 results and evidence documents were retained.
