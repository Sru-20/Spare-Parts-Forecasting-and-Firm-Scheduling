# Execution State

## Current status

- **Phase 0: COMPLETE.** Final guide `docs/Spec.pdf` has been read and reconciled; deliverables and exit evidence are recorded below.
- **Phase 1: COMPLETE.** Dataset validation, schema-checked consolidation, quality and leakage audits, derived Parquet tables, and Phase 1 tests are complete. No forecasting or final-test evaluation has been started.
- **Phase 2: COMPLETE.** Train-validation EDA, development-window segmentation, origin-safe features, leakage tests, and derived artifacts passed the gate. Final test remains unopened.
- **Phase 3: COMPLETE.** Track A rolling-origin candidates, evaluation artifacts, report, and gate checks are complete. Final-test rows remain sealed.
- **Phase 4: COMPLETE under governing SPEC v4.1.** Reverified Track A selection on held validation and identified a sensitivity-stable Tweedie champion. Generated 14-week planning projections for the top two candidates using train-validation labels only; these recursively bridge the sealed test-date gap and are not directly validated 14-week forecasts. Final-test labels remain sealed. Phase 5 is next.
- Tier 1 is mandatory. Tier 2 requires a measured Tier 1 gap; Tier 3 is deferred unless requested.
- **Architecture decision (2026-10-23; maintenance update 2026-10-25):** use React + TypeScript → one Django + Django REST Framework backend → SQLite for development; production database remains undecided. Forecasting and scheduling are imported `src/` Python packages, called in-process; do not build a Spring Boot API or separate FastAPI/ML service. Django management commands are the MVP batch entry points; no task queue or second backend service.

## Phase 0 completed work

- Read the final 17-page `docs/Spec.pdf`, the available updated problem statement, `README.md`, `data/generate_datasets.py`, `data/validate_datasets.py`, and all three source CSV headers and rows.
- Confirmed the updated problem statement is already in `docs/`; did not duplicate or modify source PDFs or datasets.
- Ran the specified validator: **32 PASS, 0 FAIL**.
- Verified actual dimensions, date ranges, split counts, missing/genuine-zero targets, series classes, catalogue/scheduling coverage, and the single D2 snapshot; details are in `reports/phase0_environment.md`.
- Read generator logic and documented its seeded synthetic patterns, outcome-generation process, and known leakage/answer-key risks in `reports/phase0_traceability.md`.
- Created the 16-factor and 10-KPI traceability tables plus updated assistant/controls/audit requirements and gated execution plan in `reports/phase0_traceability.md`.
- Recorded the frozen metric/selection/safety contract in `SPEC.md`; created baseline project configuration in `configs/project.yaml`, dependency manifest in `requirements.txt`, and the repository directory skeleton.
- Completed the Phase 0 deliverables with explicit **provisional** proxy-cost formulas (stock-out criticality multipliers, emergency cost multiplier, carrying-rate fallback), service-level targets, and zero hard-violation tolerance. These are assumptions only and must be reviewed/frozen in Phase 5.
- Phase 1 finalized the target as raw `demand_qty` for fitting and scoring after verifying the generator's global train-validation cap; the decision and evidence are recorded below and in `SPEC.md`.
- No model was trained, no scheduler or assistant was implemented, and no business KPI result is claimed.

## Phase 1 completed work

- Implemented `src/data/loader.py` with strict verified-header checks and explicit nullable string, integer, float, and datetime types for every D1/D2/D3 column.
- Implemented `src/data/build_phase1_artifacts.py` to enforce primary-key uniqueness, referential integrity, target/missing/future safeguards, weekly continuity, nonnegative numeric values, shared-attribute equality, and row-preserving join cardinality before writing outputs.
- Built D1→D3 `many_to_one` modelling table (120,861 rows × 73 columns) and D2→D3 `one_to_one` scheduling table (158 rows × 84 columns). Join row counts match their original left tables exactly.
- Phase 1 decision: train on raw `demand_qty` and score on raw `demand_qty`. The generator estimates the `demand_qty_capped` threshold over the full warm-up + train-validation span and applies it to earlier observations. **1,403** observed historical rows differ from the raw target; that precomputed cap is not a fold-local transform and is excluded from model features. Decision documented in `SPEC.md` and `configs/project.yaml`.
- The training selector returns **66,345** rows, all `train_val`, observed, non-missing labels. It excludes warm-up (lags only), final test and future. All **72** historical missing labels remain blank/excluded; **54,511** genuine zero labels remain zero; all 8,512 future labels remain blank.
- Generated per-field data dictionary (152 source-field entries) and feature availability audit (152 entries: 54 Yes, 74 Conditional, 24 No). Approved default feature set has 41 unique columns in the consolidated modelling table; excludes all `out_*`, targets, generator answer keys, train-validation-derived summaries, metadata and features not known at every forecast origin.
- The validator and report show zero primary-key duplicates, zero non-weekly date gaps, zero negative numeric values, and referential integrity with D3-only obsolete parts P0046/P0087. IQR outlier counts are retained as diagnostics, not filtered rows.
- Source datasets and PDFs were not modified. No features were fit on final-test/future labels, no forecasting model was trained, and the final test remains unopened.

## Phase 2 completed work

- Implemented `src/features/time_features.py` with origin-safe lags (1/4/13/52), shifted rolling statistics (4/13/52), same-period-last-year, intermittent-demand features, approved calendar/static features, conditional covariate gating, and recursive 14-week prediction feedback.
- Implemented `src/features/segmentation.py` with generator-compatible SBC, project-assumption ABC/XYZ, and same-component/vehicle-type analogue metadata for unclassified series.
- Implemented `src/features/phase2_eda.py` to create train-validation-only descriptive EDA and charts, development-window segment outputs, and a feature dictionary.
- EDA/event diagnostics used **66,345** observed train-validation rows, ending **2026-03-23**. No final-test or future labels were used.
- Verified a repository discrepancy: the README describes `demand_class_trainval` as train-validation-only, while the generator computes it from observed warm-up plus train-validation history through the train-validation cutoff. The implemented development-window recomputation matches **all 569 comparable supplied classes (100%)**. The supplied label remains comparison evidence, not a model predictor.
- Created **616** series segments: Smooth 119, Erratic 97, Intermittent 133, Lumpy 220, Unclassified 47 (43 have fewer than 26 observed development weeks; 4 have no development observations). All 47 unclassified series have analogue component and vehicle-type metadata; no SBC class is fabricated.
- Development-window ABC parts: A 66, B 47, C 46, Unclassified 1. XYZ: Z 70, Y 59, X 28, Unclassified 3. These are descriptive/project segmentation conventions, not performance results.
- Generated `artifacts/series_segments.parquet` (616 × 29), `reports/feature_dictionary_phase2.csv` (61 entries), `results/phase2_eda_report.md`, train-validation summary CSVs, and five charts under `results/phase2_charts/`. Parquet readback and required report/chart outputs were verified.
- Fixed two pandas concatenation warnings in recursive feature generation by normalizing numeric target dtype and excluding non-feature/outcome columns from recursive feature history. Future labels remain rejected.
- Added `tests/test_features_phase2.py`; `tests/test_leakage.py` covers truncation invariance, prior-only features, covariate approval, forbidden fields, recursive feedback, and SBC conventions.

## Phase 3 completed work and decision

- Ran all six configured, non-overlapping 14-week origins: two tuning origins and four held-validation origins. Models fit only on observed `train_val` labels available by each origin; scored dates ended 2026-02-09. Final-test and future labels were not used.
- Evaluated 12 forecast candidates: seven baselines (including the separately reported, equivalent 52-week mappings), three LightGBM families, a nine-member equal-weight ensemble, and a fixed SBC router. The separate analogue fallback remains diagnostic-only. The ensemble omits the duplicate same-period-last-year label. Router assumptions are in `configs/project.yaml`: LightGBM Tweedie for Smooth/Erratic, SBA for Intermittent/Lumpy, and the analogue fallback for Unclassified.
- Seed-42 search completed **18/18** LightGBM configurations with **0 tuning failures**. Selected configurations: point `random_3`, Tweedie `random_1`, quantile `random_1`. The full sweep produced **0 candidate execution failures** and **5,322 successful metric rows** (fold, pooled, segment and horizon-step results).
- Pooled held-validation lead-time WAPE: LightGBM Tweedie **34.31%**, quantile **36.20%**, equal-weight ensemble **37.90%**, SBC router **38.80%**; SBA **40.65%**. Pooled weekly WAPE: quantile **57.71%**, Tweedie **63.25%**, ensemble **69.26%**, router **71.02%**. This is a descriptive WAPE ordering, not a champion selection; weighted scoring, sensitivity, DM and bootstrap comparisons belong to Phase 4.
- Quantile lead-time P90/P95/P99 coverage was **93.09% / 97.41% / 99.64%**, with mean pinball loss **6.918**. Two tuning origins selected candidates; held results were not used to tune them.
- Supplier lead time was available for **158/160** demand parts; **2** lacked a mapping and **3** required more than the 14-week horizon. Those parts remain in weekly metrics and are omitted only from lead-time metrics.
- The strict same-`component`/`vehicle_type` donor rule was infeasible because all 160 catalogue combinations are unique. The broader `part_category` + `vehicle_type` fallback prefers same-region donors, excludes the target part, and uses origin-visible trailing 13-calendar-week observed means. It produced **190 finite fallback predictions** for the same 190 series-week-origin keys where each of the seven baseline forecasts is unavailable at the final held origin because no observed train-validation history exists; it is not ranked as a full candidate.
- Seasonality and festival ablations completed. Promotion/campaign, market and shutdown ablations are explicitly blocked because the data does not establish which future values were known at each origin.
- Generated `results/track_a_candidates.csv` (5,322 metric rows), `results/track_a_tuning_log.csv` (18 rows), `results/track_a_ablations.csv` (10 outcomes), `results/track_a_feature_importance.csv`, `results/phase3_track_a_report.md`, and three Track A charts. `artifacts/track_a_backtest_forecasts.parquet` contains **384,838** read-back-verified rows for **13** forecast labels, only the four held-validation origins, 2025-01-20–2026-02-09; unavailable baseline values remain missing, not zero. `origin_date` is stored as ISO text for fastparquet timestamp portability.
- The first report attempt failed because optional `tabulate` was absent; an internal Markdown table renderer replaced that optional dependency. Initial full artifact readback exposed a fastparquet millisecond timestamp conversion issue; storing `origin_date` as ISO text fixed it. A first test of the composite helper also caught duplicate merge-key columns; corrected and covered by a unit test.
- Final Phase 3 gate: `python -m pytest -q` — **33 passed** under Python 3.10 and **33 passed** under Python 3.13; `python data/validate_datasets.py data` — **32 PASS, 0 FAIL**; forecast Parquet full readback — **384,838 rows verified**. LightGBM 4.7.0 is now installed in both Python environments; `requirements.txt` already declared it and required no change.
- Architecture update after Phase 3: the user directed replacing the earlier Spring Boot + FastAPI proposal with one Django/DRF backend and in-process imports of `src/`. Updated `SPEC.md`, `README.md`, `configs/project.yaml`, `requirements.txt`, the assistant/security traceability references, and the environment note. Removed the unused `services/ml/.gitkeep` placeholder. Full `python -m pytest -q`: **34 passed**; YAML and `git diff --check` validation passed. Framework dependencies are declared but not installed, and no Django runtime exists yet; this architecture-only change does not alter completed phase results.

## Legacy Phase 4 evaluation (pre-v4.1; superseded)

The following evaluation records the earlier contract and is retained as historical evidence. The governing SPEC v4.1 selection and current gate result are recorded in the later “Phase 4 v4.1 completion” section.

- Completed Track A scoring from the four held-validation origins, conditional Tier 2 Random Forest evaluation, 16 one-at-a-time ±10-point weight perturbations, ten top-five pairwise comparisons, 2,000-replicate paired bootstrap intervals, and calibration reporting. No final-test or future labels were used.
- The 13-candidate weighted ranking was led by the simple average ensemble (**71.458**), quantile LightGBM (**69.792**), and Tweedie LightGBM (**68.958**). The top candidate changes in **6/16** sensitivity cases; therefore no sensitivity-stable champion was forced.
- Quantile LightGBM calibration: weekly P90/P95/P99 coverage **88.29% / 92.65% / 97.37%**, lead-time coverage **93.09% / 97.41% / 99.64%**; coverage errors **0.0190 / 0.0205** and pinball losses **4.1206 / 6.9182**, respectively. All non-quantile candidates receive the minimum calibration score because they do not provide the required quantiles.
- Tier 2 Random Forest selected `tier2_random_2`; **6/6** tuning fits succeeded. It ranked **9/13**, with lead-time WAPE **41.719%**, weekly WAPE **77.114%**, mixed intermittent/lumpy WAPE **126.646%**, and held-validation runtime **2,052.7 seconds**. It did not improve the best Tier 1 intermittent/lumpy outcome.
- The top-five statistical report contains **10** comparisons. For ensemble versus quantile LightGBM, quantile-minus-ensemble weekly WAPE difference was **−11.538 percentage points** (95% paired-bootstrap CI **[−15.524, −8.360]**); lead-time difference was **−0.709** (CI **[−3.653, 2.103]**). The DM weekly-WAPE difference was **−12.350**, with Holm-adjusted p-value **2.42e−12**. Interpret cautiously: four held origins, temporal dependence, and synthetic data limit inference.
- Corrected Phase 3 lead-time aggregation to exclude incomplete windows rather than treating missing forecasts as zero. The correction report covers **12** models; affected baseline observations changed from **2,200 to 2,196**, with maximum absolute WAPE change approximately **0.0024 percentage points**. Weekly metrics and stored forecasts did not change.
- Fixed an ordered-duplicate-column defect in `_future_calendar()` discovered by the Phase 4 test; the future-calendar test now passes. Phase 4 full test gates: **39 passed** under Python 3.10 and **39 passed** under Python 3.13; `python data/validate_datasets.py data`: **32 PASS, 0 FAIL**; score-weight/config assertions and `git diff --check` passed.
- Outputs include `src/forecasting/phase4.py`, the corrected `src/forecasting/backtest.py`, `src/forecasting/global_models.py`, `tests/test_phase4_selection.py`, `configs/project.yaml`, `results/phase4_track_a_report.md`, `results/track_a_selection_matrix.csv`, `results/track_a_weight_sensitivity.csv`, `results/track_a_statistical_comparisons.csv`, `results/track_a_phase4_metric_correction.csv`, `results/track_a_tier2_tuning.csv`, `results/track_a_tier2_candidates.csv`, calibration reports/metrics, and `artifacts/track_a_phase4_validation_forecasts.parquet`.
- `artifacts/models/track_a_champion.joblib` and `results/forecast_future_14w.csv` were verified absent by design. Phase 4's exit gate is not passed; do not start Phase 5 or imply a winner. To continue, the scoring assumptions need an evidence-based business decision and any approved revision must be documented and reevaluated using validation only. Do not alter weights solely to select a preferred model.

## Proposed Track A weight scenario — evaluated, not adopted

- Preserved `configs/project.yaml` and all frozen baseline scoring outputs unchanged. Added `src/forecasting/compare_weight_scenarios.py` to apply a proposed weight set to the saved Phase 4 metric matrix using the existing scoring and sensitivity functions. It asserts that frozen scores and all 16 baseline sensitivity winners reproduce before writing scenario outputs.
- Frozen weights total **100%**; proposed weights (WAPE 20%, absolute bias 15%, A-class/critical 20%, intermittent/lumpy 20%, calibration 8%, stability 7%, runtime 3%, explainability 7%) also total **100%**. Every metric direction remains unchanged: lower forecast errors, variability and runtime are better; higher explainability tier is better. Average-rank normalization and missing-calibration score 0 remain as defined in `SPEC.md`.
- On the same 13 held-validation candidates, nominal winner remains the equal-weight ensemble: baseline **71.458**, proposed **71.042**. Proposed Tweedie moves from rank 3 to rank 2 (**70.958**); quantile moves rank 2 to rank 3 (**69.625**). The proposed leader's margin over Tweedie is only **0.083** normalized score points.
- Ran every one-at-a-time ±10-point perturbation for both policies: baseline winner changed in **6/16** cases; proposed winner changed in **8/16** cases. The proposed policy therefore does not satisfy the sensitivity stability condition. No champion was selected, the baseline weights were not changed, and Phase 5 was not started.
- Business interpretation: Track A directly measures weekly/lead-time forecast error, bias, critical-subgroup and intermittent/lumpy forecast error, quantile calibration where supplied, runtime, and cross-origin lead-time WAPE variation. Critical-subgroup WAPE is not critical-part availability; stock-out/fill rate, emergency procurement, inventory/excess/carrying cost, supplier/production feasibility and schedule adherence are not measured by this comparison and require the later fixed-assumption simulator/scheduling evaluation. No factory savings or operating-policy improvement is claimed.
- `python -m src.forecasting.compare_weight_scenarios` completed and verified **2 × 13** ranking rows and **2 × 16** sensitivity rows. Full test suites: **39 passed** under Python 3.10 and **39 passed** under Python 3.13; dataset validation: **32 PASS, 0 FAIL**; output shape, weight sum, and `git diff --check` checks passed.
- Added outputs: `results/track_a_weight_scenario_report.md`, `results/track_a_weight_scenario_weights.csv`, `results/track_a_weight_scenario_comparison.csv`, `results/track_a_weight_scenario_rankings.csv`, and `results/track_a_weight_scenario_sensitivity.csv`.
- Historical recommendation under the pre-v4.1 policy: keep the baseline policy frozen and Phase 4 blocked pending stronger business evidence. This result was not adopted; it is superseded by the governing v4.1 evaluation below.

## Phase 4 v4.1 completion

- Reverified Phases 0–3 against the governing SPEC v4.1 documents. Phase 4 uses the four held-validation origins only; no final-test or future target labels were scored.
- Selected TSB over SBA using the two tuning origins (lead-time WAPE ratio **0.4448** versus **0.4738**). Three candidates passed the enforced point-estimate baseline screen.
- Stable weighted champion: `lightgbm_tweedie::random_1`, score **63.875**, winning **14/14** one-at-a-time ±10-point weight sensitivity cases. It also won the overlap-removed scenario.
- Champion metrics: lead-time WAPE ratio **0.3490** (95% two-way bootstrap CI **0.2932–0.4120**); weekly WAPE ratio **0.6325**. Tweedie lead-time WAPE deltas versus seasonal naïve and historical average were **−0.2053** (CI **−0.3034 to −0.1364**) and **−0.1221** (CI **−0.2035 to −0.0551**). These results are method validation on simulated data, not proven factory savings.
- Generated `results/forecast_future_14w_v41.csv` for the top two candidates: **8,512 rows each**, covering 14 dates from **2026-09-28 to 2026-12-28**. The projection is fitted with train-validation labels only and recursively spans effective steps 27–40, including **26 unobserved bridge weeks** across the sealed test-date interval. It contains no target or `out_*` columns and is not evidence of measured forecast accuracy. Quantile ordering violations were **0** for P50/P90/P95/P99.
- Created frozen model artifacts for the candidates in `artifacts/models/`, plus updated `results/phase4_track_a_v41_report.md`, `results/track_a_v41_selection_matrix.csv`, `results/track_a_v41_weight_sensitivity.csv`, `results/track_a_v41_sba_tsb_selection.csv`, and the v4.1 forecast/metric artifacts. Legacy Phase 4 files were preserved.
- Gate verification after the integration: `python -m pytest -q` — **47 passed** under Python 3.10 and **47 passed** under Python 3.13; `python data/validate_datasets.py data` — **32 PASS, 0 FAIL**; future-output shape/date/label/quantile checks passed; `git diff --check` passed.
- Limitations/open items: only four held origins support forecast-performance comparisons; the future projections are a long recursive bridge, not a validated operational horizon; final-test evaluation remains reserved for Phase 7. Phase 5 must freeze simulator assumptions and compare the top two forecasts on identical simulated demand paths before any scheduling claims.
- Next executable command to reproduce the completed gate: `python -m src.forecasting.phase4_v41`. Phase 5 has no implemented entry point yet; implement its simulator and scheduling workflow only after finalizing the Phase 5 assumptions in the governing spec.

## Verified project facts

- D1: 120,861 × 62; date range 2023-01-02–2026-12-28. Splits: warm-up 30,157; train/validation 66,388; final test 15,804; future 8,512 across 14 weeks.
- D2: 158 × 61, all as-of 2026-09-26. D3: 160 × 29.
- D1: 616 part-region series: Smooth 119, Erratic 97, Intermittent 133, Lumpy 220, short/unclassified 47. There are 72 missing historical targets. Observed history has 54,511 genuine zeros of 112,277 non-missing rows (48.55%).
- D2 excludes obsolete P0046 and P0087; the validator confirms the catalogue-to-schedule exception is consistent.
- Original problem-statement PDF named in README is absent. `docs/Spec.pdf` is the final guide, not that original. The updated problem statement is present and carries the original factors/KPIs plus assistant requirements. Discrepancy recorded in `SPEC.md` and Phase 0 traceability.

## Phase 0 files and artifacts

- `SPEC.md` — project-specific evaluation and safety contract.
- `STATE.md` — phase record and handoff.
- `configs/project.yaml` — project paths, seed, splits, service targets, score weights, assistant defaults and gates.
- `requirements.txt` — project Python/ML/backend/test dependencies; Django/DRF packages are declared but were not installed or runtime-verified as part of the architecture decision.
- `reports/phase0_traceability.md` — requirements mapping, discrepancies, generator findings, and execution plan.
- `reports/phase0_environment.md` — runtime/tool/package availability, dataset dimensions and validator result.
- `tests/test_phase0_contract.py` — contract and Phase 0 deliverable checks.
- `src/` — shared importable data, feature, forecasting and future scheduling packages; `backend/` is the planned single Django application. `frontend/`, `artifacts/`, and `results/` remain the UI/output areas. No separate `services/ml/` application is planned.

## Phase 1 files and artifacts

- `src/data/loader.py` — schema-validated source loaders, safe join builders, default-feature selector, and training-row selector.
- `src/data/build_phase1_artifacts.py` — repeatable integrity audit and derived artifact/report builder; run with `python -m src.data.build_phase1_artifacts`.
- `artifacts/modelling_table.parquet` — 120,861 × 73 joined demand table.
- `artifacts/scheduling_inputs.parquet` — 158 × 84 enriched scheduling table.
- `results/data_quality_report.md` — measured quality, split, target and join results.
- `results/demand_iqr_outliers.csv`, `results/negative_values.csv`.
- `reports/data_dictionary.csv`, `reports/feature_availability_audit.csv`.
- `tests/test_data_phase1.py` — loader, schema, joins, leakage, missing-label and derived-artifact tests.

## Gate results and failures

- Dataset validator at the Phase 1 gate: `python data/validate_datasets.py data` — **32/32 PASS, 0 FAIL**.
- Full Python 3.10 test gate: `python -m pytest -q` — **12 passed**.
- Full Pylance-selected Python 3.13 test gate: `C:\Users\sruja\AppData\Local\Programs\Python\Python313\python.exe -m pytest -q` — **12 passed**.
- `configs/project.yaml` parsed and verified; forecast weights sum to 1.00, supplied scheduling weights to 0.95, and assistant weights to 1.00.
- YAML assertions for provisional proxy multipliers and zero hard-constraint threshold passed in Phase 0.
- Current provisional, editable simulator proxies in `configs/business.yaml`: stock-out penalty `unit_cost × {High: 10, Medium: 5, Low: 2}` per unmet unit, varied ±50%; emergency cost `quantity × unit_cost × 1.5`, sensitivity range 1.25–2.0; prefer D2 `carrying_cost_rate`, using a 25% annual fallback only when that source rate is absent. These are assumptions, not measured costs; Phase 5 must sensitivity-test them before freezing.
- Environment at Phase 0 inventory: Python 3.10.9, Java 26.0.2.1, Maven 3.9.16, Node 22.20.0, npm 10.9.3; 14 logical CPUs; no `nvidia-smi`; `mysql` client not found. Java and Maven availability is historical environment evidence, not a requirement for the selected backend.
- LightGBM, OR-Tools/PuLP, Optuna, PyTorch/Transformers, and sentence-transformers were absent at Phase 0; LightGBM was subsequently installed for Phase 3. Core data/science packages, XGBoost, FastAPI, pytest, and PyYAML were present in the Phase 0 snapshot. FastAPI is no longer a project runtime dependency. The selected Django, DRF and SimpleJWT versions and import verification are recorded in the maintenance checkpoint below.
- Outbound PyPI HTTPS returned HTTP 200; this does not verify authenticated LLM provider access. No recognized LLM API credential variable was present.
- `pypdf` was installed at user scope solely to read the supplied PDFs; it is not in the project dependency manifest.
- `fastparquet` was added to `requirements.txt` and installed in Python 3.10 and 3.13 to write/read the required Parquet outputs. An attempted PyArrow wheel download timed out; it was not retained as a dependency.
- During implementation, runtime checks caught and corrected two dtype-map omissions (`festival_intensity` is categorical; supplier id/type/country fields are strings). The first generic `runTests` attempt did not discover these pytest files, so the repository's `python -m pytest` runner was used directly and passed.
- Phase 2 validator: `python data/validate_datasets.py data` — **32/32 PASS, 0 FAIL**.
- Phase 2 focused leakage/artifact tests with pandas `FutureWarning` treated as an error — **9 passed**.
- Phase 2 full pytest gates: default Python environment — **21 passed**; Python 3.13 — **21 passed**.
- No final-test evaluation, forecast training, scheduling, or assistant execution occurred in Phase 2.
- An initial focused test run exposed a duplicate-column construction in recursive features and an incorrect test path; both were corrected. No outstanding Phase 2 failures or warnings remain.

## Assumptions and deferred decisions

- Weights and the 5-second proposal latency budget are project assumptions, not business-validated thresholds.
- Scoring weights and the 5-second proposal latency budget are project assumptions, not business-validated thresholds. Track B weights total 100%.
- Assistant languages are English, Hindi, Hinglish and Marathi; Tamil remains optional for later.
- Simulator initialization, backorder/lost-sales, delay/rejection, emergency procurement, proxy costs, and plant calendar treatment must be finalized and frozen in Phase 5.
- The final test remains unopened and must be consumed once in Phase 7.

## Project open issues

1. The original problem statement PDF remains absent. Use the supplied updated statement plus verified data definitions; report any later conflict rather than guessing.
2. Snapshot-only fields and synthetic drivers remain conditional: any use in forecasting must be verified as available at the corresponding forecast origin.
3. The Phase 5 simulator must review and freeze provisional cost proxies; no such proxy is used in Phases 1 or 2.
4. Segment labels computed over the full development window must not be used as predictor inputs for earlier rolling origins; recompute from each fold's origin-visible history.

## Next phase and executable validation command

- **Phase 4 complete under v4.1.** Champion: `lightgbm_tweedie::random_1`; quantile LightGBM remains the safety-stock/calibration contender.
- **Scoring-policy decision:** SPEC v4.1 screening gates, then weighted ranking, then downstream tie-break replaces weights-only selection. The legacy baseline **6/16** and proposed **8/16** sensitivity results remain preserved as historical scenario evidence, not deleted or used as the Phase 4 decision.
- **Known-in-advance rule:** festival calendar, plant calendar/shutdowns, and promotions/campaigns published by the forecast origin may be used. Unplanned market events and any information observed after the origin may not. Existing Phase 4 results are unchanged; covariates without origin-publication evidence were excluded from that evaluation.
- `results/forecast_future_14w_v41.csv` is a provisional bridge projection, recursively spanning 26 unobserved weeks; it is not a validated forecast.
- Reproduce Phase 4: `python -m src.forecasting.phase4_v41`. This command regenerates Phase 4 evidence; maintenance validation should not invoke it.
- Next phase: Phase 5 scheduling and simulator work, after freezing the simulator assumptions. Final-test labels remain sealed.

## SPEC v4.1 maintenance checkpoint (2026-10-25)

- SPEC contract text was not changed. Configuration and report changes below implement the frozen contract and requested provisional business settings; the historical Phase 4 forecasts, selection, model artifacts and sensitivity outputs were not recomputed.
- Verified Track A weights in `configs/project.yaml` are exactly SPEC v4.1 section 6 values (overall error 20%, critical/A 20%, intermittent/lumpy 15%, asymmetric bias 15%, calibration 15%, stability 10%, runtime/explainability 5%). `run_phase4_v41` reads `scoring.tracks.forecasting.weights` directly. The tracked champion and saved Phase 4 scores are unchanged.
- Track B weights are configured to 25/15/15/20/15/10 (100%); assistant weights to 25/25/20/15/10/5 (100%). Weight-total and exact-value tests cover all three tracks.
- Removed the obsolete Track B weight-total statement and replaced the old provisional proxy amounts with current editable values. The config prefers D2 carrying rate and uses its configured 25% annual proxy only if absent.
- Development database config is SQLite. `mysqlclient` was removed; production database selection is pending. Django **5.2.18**, Django REST Framework **3.18.3**, and SimpleJWT **5.5.1** were installed in Python 3.10; all three imports and exact versions were verified. Exact pins are in `requirements.txt`.
- The future v4.1 CSV is labeled `forecast_status=provisional_bridge`; the data-loader docstring instructs that these projections must retain that display label.
- Original datasets and the final-test split were not changed or opened. Historical Phase 4 forecasts, result rankings, and frozen model artifacts were not changed; only the future projection received its required status column.
- Existing old-weights scenario evidence was verified without rerunning or overwriting it: the saved frozen-baseline winner is `simple_average_ensemble::equal_weight_nine_model_mean` (score 71.458); baseline/proposed sensitivity records remain in the prior scenario outputs.
- The report addendum was calculated from the saved Phase 4 Parquet/CSV outputs. The champion-minus-ensemble lead-time WAPE difference was -0.0320 (95% CI [-0.0629, -0.0041]); champion-minus-quantile difference was -0.0211 (95% CI [-0.0610, 0.0209], not statistically separated). Three candidates passed the gates.
- Gate commands/results: `python -m pytest -q` under Python 3.10 — **49 passed**; Python 3.13 — **49 passed**; `python data/validate_datasets.py data` — **32 PASS, 0 FAIL**; `git diff --check` — **passed**.
- SPEC.md contract text and status were left unchanged because its v4.1 status already reflected Phase 4 completion. The current maintenance log and next-phase handoff are recorded here.
