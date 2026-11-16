# Execution State

## Current status

- **Phases 0–4: COMPLETE** according to the phase evidence below. The frozen Track A point champion is `lightgbm_tweedie::random_1`; its saved future projection remains a provisional bridge.
- **Phase 5: COMPLETE WITH APPROVED DATA-LIMITATION EXCEPTION** The exact same-component/same-vehicle-type rule is retained. Where no donor exists, the approved status is `UNFORECASTED_NO_DEFENSIBLE_ANALOGUE`; no forecast quantity is fabricated or replaced by zero. Exit-gate evidence recorded in `results/phase5_exit_gate_evidence_v1.json`.
- **Phase 6: COMPLETE** against its v4.2 gate under the user's explicit sequencing deviation. The primary validation winner is `lightgbm_quantile::random_1` + `risk_adjusted_mrp`; this does not change the frozen Track A point champion.
- **Phase 7: Runner implemented; preflight gate UNBLOCKED (Phase 5 closure evidence recorded); final-test evaluation NOT STARTED.** Do not consume the final test without separate explicit authorization.
- Tier 1 is mandatory. Tier 2 requires a measured Tier 1 gap; Tier 3 is deferred unless requested.
- **Specification authority:** `docs/SPEC.md` v4.2 FINAL governs technical requirements; this root `STATE.md` records execution status. There is no root-level `SPEC.md` in the current repository. The v4.2 status table is a specification snapshot; the later status entries here record subsequent execution and the approved Phase 6 sequencing deviation.
- **Final-test access note:** a prior validator invocation read `demand_qty` from sealed final-test rows for integrity checks, not prediction evaluation. Do not claim those labels were never accessed. Phase 7 evaluation has not been run.

## Current repository structure

- `data/` contains the three source CSVs and dataset-generation/validation scripts. Original datasets are not to be modified.
- `src/` contains the Python implementation, organized into `assistant/`, `classification/`, `data/`, `evaluation/`, `features/`, `forecasting/`, `scheduling/`, and `simulation/`.
- `tests/` contains focused pytest modules for the implemented Python packages and phase workflows.
- `configs/` contains project configuration. `artifacts/`, `results/`, and `reports/` contain generated intermediate data, evaluation/scheduling outputs, and documentation, respectively.
- `backend/` and `frontend/` currently contain only `.gitkeep` placeholders. There is no `manage.py`, Django application, frontend `package.json`, or React application scaffold. Django/DRF dependencies are declared in root `requirements.txt`, but the application architecture described in the documents is not yet implemented.
- `docs/SPEC.md` is the current specification location; the required root `SPEC.md` mentioned by the specification's workflow section is absent. `README.md` describes MySQL persistence, while v4.2 specifies SQLite for development and MySQL or Postgres by environment. Treat v4.2 as authoritative; reconcile the README discrepancy when documentation work is authorized.

- **Target architecture decision (2026-10-23; maintenance update 2026-10-25):** React + TypeScript → one Django + Django REST Framework backend → SQLite for development, MySQL or Postgres by environment. Forecasting and scheduling are imported `src/` Python packages called in-process; no Spring Boot, separate FastAPI service, queue, or second backend. This is the selected target architecture, not a claim that the currently placeholder `backend/` and `frontend/` directories are implemented.

## Phase 0 completed work

- Read the final 17-page `docs/Spec.pdf`, the available updated problem statement, `README.md`, `data/generate_datasets.py`, `data/validate_datasets.py`, and all three source CSV headers and rows.
- Confirmed the updated problem statement is already in `docs/`; did not duplicate or modify source PDFs or datasets.
- Ran the specified validator: **32 PASS, 0 FAIL**.
- Verified actual dimensions, date ranges, split counts, missing/genuine-zero targets, series classes, catalogue/scheduling coverage, and the single D2 snapshot; details are in `reports/phase0_environment.md`.
- Read generator logic and documented its seeded synthetic patterns, outcome-generation process, and known leakage/answer-key risks in `reports/phase0_traceability.md`.
- Created the 16-factor and 10-KPI traceability tables plus updated assistant/controls/audit requirements and gated execution plan in `reports/phase0_traceability.md`.
- Recorded the frozen metric/selection/safety contract in `SPEC.md`; created baseline project configuration in `configs/project.yaml`, dependency manifest in `requirements.txt`, and the repository directory skeleton.
- Completed the Phase 0 deliverables with **provisional** proxy-cost formulas, service-level targets, and zero hard-violation tolerance. Phase 5 reviewed and froze the simulator assumptions in `docs/phase5_assumptions.md`.
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
- The current `configs/project.yaml` has been re-verified: forecasting, scheduling and assistant weight sets each sum to 100%; see the SPEC v4.1 maintenance checkpoint for exact weights.
- YAML assertions for provisional proxy multipliers and zero hard-constraint threshold passed in Phase 0.
- Current provisional, editable simulator proxies in `configs/business.yaml`: stock-out penalty `unit_cost × {High: 10, Medium: 5, Low: 2}` per unmet unit, varied ±50%; emergency cost `quantity × unit_cost × 1.5`, sensitivity range 1.25–2.0; prefer D2 `carrying_cost_rate`, using a 25% annual fallback only when that source rate is absent. These assumptions are not measured costs; they were frozen for the Phase 5 simulator, and sensitivity testing remains for scheduler comparison.
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
- Simulator initialization, backorder/lost-sales, delay/rejection, emergency procurement, proxy costs, and plant calendar treatment are frozen in `docs/phase5_assumptions.md`; they remain synthetic assumptions, not calibrated operating rules.
- The final test remains unopened and must be consumed once in Phase 7.

## Project open issues

1. The original problem statement PDF remains absent. Use the supplied updated statement plus verified data definitions; report any later conflict rather than guessing.
2. Snapshot-only fields and synthetic drivers remain conditional: any use in forecasting must be verified as available at the corresponding forecast origin.
3. The Phase 5 simulator must review and freeze provisional cost proxies; no such proxy is used in Phases 1 or 2.
4. Segment labels computed over the full development window must not be used as predictor inputs for earlier rolling origins; recompute from each fold's origin-visible history.

## Phase 4 reproduction reference

- **Phase 4 complete under v4.1.** Champion: `lightgbm_tweedie::random_1`; quantile LightGBM remains the safety-stock/calibration contender.
- **Scoring-policy decision:** SPEC v4.1 screening gates, then weighted ranking, then downstream tie-break replaces weights-only selection. The legacy baseline **6/16** and proposed **8/16** sensitivity results remain preserved as historical scenario evidence, not deleted or used as the Phase 4 decision.
- **Known-in-advance rule:** festival calendar, plant calendar/shutdowns, and promotions/campaigns published by the forecast origin may be used. Unplanned market events and any information observed after the origin may not. Existing Phase 4 results are unchanged; covariates without origin-publication evidence were excluded from that evaluation.
- `results/forecast_future_14w_v41.csv` is a provisional bridge projection, recursively spanning 26 unobserved weeks; it is not a validated forecast.
- Reproduce Phase 4: `python -m src.forecasting.phase4_v41`. This command regenerates Phase 4 evidence; maintenance validation should not invoke it.
- Phase 5 status and its reproducibility command are recorded below. Final-test labels remain sealed.

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

## Phase 5 completed — scheduling core and simulator

- Read `SPEC.md`, `docs/Spec.md` section 7 and Appendix A, and `docs/SPEC.docx` before implementation. No contract text or final-test data was changed or opened.
- Verified D2's 158/158 supplier-share totals equal 100%. `generate_datasets.py` assigns SOB patterns as synthetic load-allocation weights and does not establish a binding contract; SOB is therefore a target, and realized deviations are reported rather than treated as violations.
- Completed Appendix A in `reports/phase5_appendix_a.md`. D2 is one snapshot dated 2026-09-26, reused only as identical synthetic initialization. `backorder_qty` is the labelled pending-requirement assumption. In-transit and open-PO due buckets are rebased to weeks 1 and 4 without double counting `open_po_qty`. Arrivals use `release_week + ceil(lead_time_days / 7)`; planning lead demand beyond 14 weeks extends using the last-four-week mean, and the actual demand path is not extended.
- Wrote the hand-calculated scheduling/simulation tests before implementing the engine. The 27 Phase 5 tests cover the ten section 7 cases, inventory conservation, determinism, missing-input errors, per-release requirements, and independent rejection of invalid schedules.
- Implemented deterministic 14-week MRP netting, criticality/service-level quantile safety stock, MOQ/multiple rounding, SOB-target supplier allocation, monthly shared supplier capacity, explicit firm/unresolved status, an independent checker, and a weekly simulator with critical backlog, ordinary lost sales, seeded supplier delay/rejection, emergency supply and cost proxies.
- Frozen assumptions and config hashes are recorded in `docs/phase5_assumptions.md`. Reused only saved validation forecasts and actuals. Of 570 complete part-origin paths, 563 were schedulable; the seven paths excluded from scheduling were P0046 (four origins) and P0087 (three), which are obsolete and absent from D2.
- Executed `python -m src.scheduling.phase5`: **6,718** supplier order lines, **7,882** MRP plan weeks, **7,882** simulation weeks; **442** schedules feasible, **121** unresolved for unfilled requirements under capacity limits, **0** independent hard-constraint violations. All MRP and simulator weekly rows passed inventory conservation with no negative closing stock. No forecast artifact was modified.
- Exit gates: full `python -m pytest -q` — **76 passed** on Python 3.10 and **76 passed** on Python 3.13; `python data/validate_datasets.py data` — **32 PASS, 0 FAIL**; `git diff --check` — **passed**. New Phase 5 files also passed a trailing-whitespace check.
- No existing project files were removed: original data, frozen forecast artifacts and historical reports are required evidence; Phase 5 output files are reproducibility/audit deliverables, not temporary files.
- **Next phase:** Phase 6 scheduler comparison is not started, per the requested scope. Reproduce this Phase 5 run with `python -m src.scheduling.phase5`. Final-test labels remain sealed.

## Phase 5 closure addendum — assumptions v2

- Updated `docs/phase5_assumptions.md` with the requested evidence, sensitivity results, pull-ahead diagnosis, quantile coverage, critical/A-class metrics, and the current SHA-256 hashes. The config version is `phase5-assumptions-v2`. Fixed a duplicate YAML key by merging the emergency rule and cost metadata; the parsed configuration retains the same values and runtime behavior. Added a regression test for the merged configuration.
- Changed only the status sentence in `SPEC.md` section 7 to record Phase 5 completion and Phase 6 as next; the v4.1 contract text was not changed.
- SOB is a target, not a proven contract limit: all 158 D2 rows sum to 100%, and the generator uses shares as synthetic load-allocation weights without contract enforcement.
- Of 640 possible part-origin paths, 570 have complete 14-week forecast/actual data; 42 have no forecast rows, 22 have missing regional labels, and 6 have incomplete horizons. All 48 unclassified paths are excluded. Seven complete paths are obsolete parts absent from D2, leaving 563 scheduled paths. Exclusions are not representative; rates by criticality and SBC segment are documented in the assumptions report.
- Initial inventory cover median is 0.519 weeks High, 0.614 Medium, and 0.519 Low; 61–69% of retained paths have less than one week of cover. Lead-time coverage is 99.47% High (189), 100% Medium (240), and 94.03% Low (134), against targets of 99%, 95%, and 90%. No target exceeds P99. Calibration remains off because no tuning-origin quantile forecasts are saved.
- Lead-time WAPE/bias: High 25.00% / -9.13%; ABC A 20.99% / -3.93%; High∩A 26.01% / -9.44%. Each passes the -10% under-forecast rule; asymmetric biases and P90/P95/P99 coverage by horizon are in `docs/phase5_assumptions.md`.
- Pull-ahead left unresolved schedules unchanged at 121/121 and proposed no orders. All unresolved schedules are supplier-capacity/unfilled-requirement cases. The eight apparent earlier-month slack cases were P0133/S006 with 1.39 units available against a 5-unit order multiple; the independent allocator correctly returned zero.
- Sensitivities were executed for stock multiplier 0.5, due-later weeks 3 and 6, and per-unit-binomial rejection. The frozen baseline was unchanged. Exact results are in `results/phase5_sensitivity.csv` and summarized in the assumptions document. Whole-lot rejection remains the default because the generator provides no unit-level rejection evidence.
- Repeated the default core twice with identical returned summaries. The frozen forecast artifact SHA-256 remained `b589605d55051bace9844d25dd9371f11f31a2a46fd080996425462f04211cb8`. Both runs report 563 paths, 6,718 orders, 442 feasible / 121 unresolved schedules, zero independent hard-constraint violations, and zero MRP/simulator ledger failures. The final test remains sealed.
- Closure gates: focused Phase 5 tests — **41 passed** on Python 3.10 and Python 3.13; full `python -m pytest -q` — **90 passed** on each runtime; `python data/validate_datasets.py data` — **32 PASS, 0 FAIL**; final `git diff --check` — **passed**.
- **Next command:** `python -m src.scheduling.phase5` to reproduce the frozen default core. Phase 6 scheduler comparison remains not started, per scope.

## Reporting checkpoint (2026-11-14)

- Created `reports/project_report_remaining_work.md` for the mentor's 14 Nov 2026 requirements: (a) more data and at least three years of past data for schedule and forecast, (b) forecast percentage from six-month backtesting. The report records measured evidence only and marks unverified business inputs as decisions required.
- Verified forecast history is **195 weeks (3.72 years)**, 2023-01-02 to 2026-09-21; **569/616** series (92.4%) carry the full history; 47 launch series start 2025-11-24 to 2026-04-06.
- Verified schedule history is **absent**: D2 has one `as_of_date` (2026-09-26), 158 rows, no order IDs or receipt dates. The three-year requirement is met for forecast and not met for schedule.
- Verified six-month backtest outputs: 3 origins (2024-09-23, 2025-03-24, 2025-09-22), 26-week horizon, 3 models, 2 evaluation levels. LightGBM Tweedie mean accuracy is **35.10% weekly** and **66.98% complete 26-week part-region**; mean aggregate bias is **−16.75%**.
- Document-control findings recorded: `SPEC.md` is deleted from the working tree; `src/forecasting/six_month_backtest.py` is absent (only `.pyc` remains) and untracked; six-month outputs and all Phase 5 code/results are uncommitted; `docs/phase5_assumptions.md` is absent.
- Current gate could not be reproduced: `python -m pytest -q` fails during collection with `FileNotFoundError: SPEC.md`. The last recorded green gate remains 90 passed on Python 3.10 and 3.13 (Phase 5 closure). No phase status is changed by this checkpoint.

## Bias-calibration experiment (2026-11-14) — closed, not adopted

- Implemented `src/forecasting/bias_calibration.py` and ran it: refit the frozen `lightgbm_tweedie::random_1` at the two tuning origins (2024-07-01, 2024-10-07), fitted multiplicative factors on tuning lead-time evidence only, applied them unchanged to the saved four-origin held artifact, and scored weekly and lead-time WAPE/bias. Held labels were used for scoring only; the final test remained sealed.
- Evidence sizes: 15,922 observed tuning weekly rows; held 32,032 weekly rows and 2,248 lead-time windows over 565 series, including 48 over-14-week extension windows.
- Acceptance rules: improve held lead-time WAPE by at least 0.5 pp, keep |held lead-time bias| at or below 10%, and not reduce P95 coverage by more than 1 pp where quantiles exist. The champion emits no quantiles, so the coverage rule does not apply.
- Measured held baseline: weekly WAPE 63.25% / bias -15.11%; lead-time WAPE 34.90% / bias -5.06%; High-criticality lead bias -7.04%. Per-origin lead WAPE 36.15%, 38.32%, 33.80%, 31.31%.
- All seven variants failed. lead_pooled (factor 1.2038): lead WAPE 41.83%, bias +14.29%. weekly_pooled (1.2198): 42.73%, +15.81%. recent_origin (1.1611): 39.64%, +10.24%. lead_criticality: 42.12%, +14.16% (High +21.90%). Exploratory diagnostics: lead_sbc 39.02%, +7.47%; lead_abc 38.95%, +9.05%; volume quartile 42.32%, +14.13%. No variant improved more than one of the four held origins; `lead_abc` improved 2025-01-13 by 1.30 pp and worsened the other three by 2.37–9.12 pp.
- Transfer diagnosis: tuning bias does not carry to held evidence. SBC Smooth -3.71% → +0.27%; Erratic -23.89% → -1.52%; ABC B -2.63% → +5.45%; lowest volume quartile +348.90% → -74.59%. Weekly held bias (-15.11%) and lead-time held bias (-5.06%) disagree because of window composition, and per-horizon bias changes sign by origin (2025-04-21: +25.2% at step 4, -60.4% at step 6).
- Decision: keep the frozen champion uncorrected. A constant or segment multiplier is rejected under the stated gates; a structural model change would need separate approval and independent validation. The four held origins are no longer independent for further exploratory variants.
- Outputs: `results/bias_calibration_candidates.csv`, `results/bias_calibration_segment_transfer.csv`, `results/bias_calibration_summary.json`, `results/bias_calibration_report.md`, `artifacts/bias_calibration_tuning_forecasts.csv.gz`. Parquet was rejected for the tuning audit artifact because the installed fastparquet misread its own datetime units; the gzip CSV round-trips dated rows correctly and is deterministic (`mtime=0`, no runtime column).
- Tests: `python -m pytest -q tests/test_bias_calibration.py` — **10 passed** on Python 3.10. The full suite remains blocked at collection by the missing `SPEC.md`; no phase status or Phase 4/5 artifact was changed by this experiment.

## Structural model experiments (2026-11-14) — closed, no candidate admitted

- Implemented `src/forecasting/model_experiments.py` and trend features in `src/forecasting/global_models.py`: origin-safe 4/13 and 13/52 trend ratios (capped at 10), 4-week growth, 13-week CV, 13-week zero rate, and observed history length. Added a `lightgbm_poisson` objective family. Evaluated four candidates against the tuning origins first; only candidates with at least a 0.5 pp tuning lead-WAPE gain went to held origins.
- Tuning (2024-07-01, 2024-10-07): frozen Tweedie lead WAPE 39.29% / bias -16.93%; blend 75% Tweedie + 25% quantile P50 38.90% / -20.23% (gain 0.40 pp, below the 0.5 pp rule); Tweedie with trend features 40.03% / -17.35%; Poisson 41.28% / -12.01%; Poisson with trend features 42.31% / -10.29%.
- Held (2025-01-13, 2025-04-21, 2025-07-28, 2025-11-03): frozen weekly 63.25% / -15.11%, lead 34.90% / -5.06%. Blend (exploratory only, not admissible) weekly 61.25% / -20.03%, lead 34.15% / -9.91%, paired lead WAPE difference -0.76 pp with 95% CI [-2.17, +0.46] pp, per-origin lead WAPE 35.86 / 37.24 / 31.55 / 31.87. Three origins improved, 2025-11-03 worsened by 0.56 pp.
- Decision: `keep_frozen_champion`. The blend fails the pre-declared tuning advancement rule, its held gain is not statistically separable from zero, and it worsens weekly and lead-time bias. Poisson and the added trend features reduced bias but increased WAPE. No Phase 4/5 artifact or model was changed; the final test remains sealed.
- Error concentration unchanged and recorded in the report: Smooth 36.5% of absolute error, Erratic 34.9%, Lumpy 20.8%; ABC A 66.1%. Two one-off Lumpy spikes (P0032 West, P0098 West) account for 6.9% of all absolute error and are not forecastable from history.
- Outputs: `results/model_experiments_tuning.csv` (SHA-256 `5f6d7c42acb99b1324efef83afa05c8b4b6368d3ca5cf9cf0e3660b2a3902f31`), `results/model_experiments_held.csv` (`b8ad7e68523d56b421f79c716ae701a02e54c0ce69ddbe5942373fc769cafbde`), `results/model_experiments_summary.json`, `results/model_experiments_report.md`. The summary JSON carries a runtime field, so its hash changes per run; metrics are deterministic.
- Tests: `python -m pytest -q tests/test_model_experiments.py` — **5 passed**; combined experiments, Phase 4 v4.1 and bias-calibration tests — **22 passed** on Python 3.10. Full suite still blocked at collection by the missing `SPEC.md`. The four held origins are now reuse-limited for any further variant; a real improvement claim needs a new independent period.

## SPEC v4.2 amendment logged (2026-11-14) — implementation in progress

- **Logged reason (verbatim):** the original MVP document lists parts the spec did not cover (AI part classification, supplier risk in scheduling, reorder point output, broader model comparison, optional SLM), and the mentor requested an interim six-month backtest. This amendment adds only what is needed. Everything else in v4.1 is unchanged. Phases 0 to 5 are not redone; the items below attach to the phase named.
- The amendment text is stored verbatim in `docs/SPEC_v4.2_amendment.md`. `docs/SPEC.pdf` (v4.1) is unchanged; its SHA-256 is `71a45d1f70d7e0464d87c1a04e0a7a8bb88e24de9803566f84e4efa656097668`. `configs/project.yaml` records version 4.2 and the amendment path.
- Contract test repair: `tests/test_phase0_contract.py` no longer reads the deleted root `SPEC.md`, `docs/Spec.md` or `docs/SPEC.docx`. It now verifies the governing PDF hash and the v4.2 amendment, which removes the collection blocker recorded earlier.
- Items A1–A4 and B are implemented and measured in the sections below once their runs complete. A5 is deferred under the amendment's own deferral rule because Phase 8 Ask AI does not exist yet, no approved small instruct model is installed, and no GPU is available; the gate requires Ask AI safety tests first.
- Held-origin look ledger before this amendment: Phase 4 v4.1 selection looked once; bias-calibration scoring looked once; Experiment 2 blend scoring looked once. A1 does not add a look unless a new candidate clears the 0.50 pp tuning gate.

## SPEC v4.2 consolidation, A1/B results, and current project report (2026-11-14)

- **Document control reconciled.** The working tree carries `docs/SPEC.md` (33,346 bytes, SHA-256 `aa47dfedb673de304f2a5ea7338cefa3bf20c74c33acbcd0dd99a37d11947344`), titled "SPEC v4.2 (FINAL, supersedes all earlier versions)". The earlier entries above that named `docs/SPEC_v4.2_amendment.md` and `docs/SPEC.pdf` no longer describe the tree: neither file is present. `docs/SPEC_AMENDMENT.md` is a separate v4.3 amendment that names `SPEC_FINAL_v4_3.md` as its baseline; that baseline is not in the project, and the amendment is neither accepted nor implemented. `configs/project.yaml` now records `specification.version: "4.2"`, `document: docs/SPEC.md`, `document_status: final_supersedes_earlier_versions`, and a logged `change_control` line. `tests/test_phase0_contract.py` was rewritten against those strings, so the collection blocker recorded in the bias-calibration and structural-experiment sections is closed. `docs/phase5_assumptions.md` was restored from history and its config/business hashes refreshed (config SHA-256 `91cdec6d98727cc57092b89212556d0792c135b93607ac9cd39766af5b1fb9c4`) to satisfy the Phase 5 evidence test.
- **A1 — broader model comparison (tuning only).** Implemented `src/forecasting/model_comparison.py`: `moving_average_13`, `exponential_smoothing` (ETS(A,N,N)), and `xgboost` on the frozen feature set with the same three seed-42 configurations mapped from the LightGBM budget. Tuning-origin lead-time WAPE: champion `lightgbm_tweedie::random_1` **39.29% / bias -16.93%**; tsb 44.48%; moving average 45.07%; historical average 46.08%; xgboost r1 46.12% (-4.89%); ETS 46.14%; seasonal naive 51.29%; naive 61.81%. Best new candidate is 5.78 pp worse than the champion, against the pre-declared 0.50 pp gain needed to advance. Decision `champion_retained_tuning_gate_not_met`; **no held origin was opened**, so the held-look ledger stays at three (Phase 4 selection, bias calibration, Experiment 2). MAPE is reported only on non-zero actuals with 301 excluded rows and is not scored. SARIMA/ARIMA and LSTM are deferred for cost and hardware reasons stated in the report. Outputs: `results/model_comparison_tuning.csv`, `results/model_comparison_summary.json`, `results/model_comparison_report.md`.
- **A2 — part-category classifier (historical result, superseded for Phase 8A).** The earlier row-stratified CV and candidate comparison are retained here as historical evidence only. They are not the Phase 8A gate result; see the component-grouped evaluation recorded below.
- **A3/A4 — supplier risk and reorder points.** Risk levels High 53 / Medium 52 / Low 53. Ablation over 563 part-origin paths: risk-adjusted planning raised order quantity 2.89% (766,550 → 788,666), raised projected unmet units 17,203 (225,978 → 243,181), cut high-risk large-share warnings 324, and did not reduce expected rejection loss. Computed reorder points are several times D2 `current_reorder_point_qty` (High median 587.7 vs 54.0); whether D2 is a real current policy is **not established**. Both are reported as exposure comparisons, not savings.
- **A5 — deferred under its own rule.** Phase 8 Ask AI does not exist, no approved <=3B instruct model is installed, and there is no GPU. No model was downloaded or run.
- **B — interim 26-week backtest.** `src/forecasting/interim_backtest.py`, origins 2025-03-24 and 2025-09-22, windows 2025-03-31..2025-09-22 and 2025-09-29..2026-03-23. The amendment's calendar anchoring would have placed the first window at 2025-03-31..2025-09-29, but the 2025-09-29 origin would then have scored weeks later than its own fit point; the origin was moved back one week to keep every recursive feature origin-safe. That deviation is documented and does not open the final test. Pooled weekly WAPE: quantile 60.12% (bias -34.15%), **tweedie 66.15% (-14.03%)**, ensemble 71.02%, tsb 73.77%, historical average 78.06%, naive 88.71%, seasonal naive 90.88%. Complete 26-week part-region aggregate: **tweedie 33.05% (-14.20%)**, ensemble 35.86%, tsb 38.20%, quantile 41.90%, historical 42.21%, seasonal naive 44.76%, naive 72.92%. 1,116 complete series-origins and 65 incomplete; 29,950 global-model weekly rows, 22 excluded for missing actuals and 384 for missing baseline forecasts. Horizon bands 1-13 / 14-26, SBC and criticality breakdowns are in the CSV. The benchmark is **not used for champion selection** and its two windows are inside held validation, so they are not independent.
- **Cleanup.** Moved out of the repository to `C:\Users\sruja\OneDrive\Desktop\spare-parts-archive-2026-11-14\`: `final_project_completion_plan.md`, `reports/project_report_remaining_work.md`, `reports/project_report_remaining_work.docx`, `reports/project_progress_report.md` (23 Oct 2026, superseded), `results/forecast_six_month_backtest.csv`, `results/forecast_six_month_backtest_summary.json`, and `artifacts/forecast_six_month_backtest.parquet`. Deleted generated caches (`__pycache__`, `.pytest_cache`, docx lock files). Kept data, frozen artifacts, Phase 3-5 results and evidence. Updated the two result reports that pointed at the archived `project_report_remaining_work.md`.
- **Current report.** Wrote `reports/project_report_current.md` with measured numbers for every phase, the A1 and B tables, A2/A3/A4/A5 status, the seven open decisions, limits, and reproduction commands.
- **Gates.** `python -m pytest -q` — **119 passed**; `python data/validate_datasets.py data` — **32 PASS, 0 FAIL**; `git diff --check` — clean. Subsequent integrity review established that this historical validator execution read `demand_qty` from sealed final-test rows for integrity checks, not prediction evaluation; do not describe the labels as never accessed.

## Session 3 update — Phase 5 v4.2 closure verification (2026-11-15)

- **Current Phase 5 status: PARTIAL.** This entry supersedes the older
  assumptions-v2 “Phase 5 complete” status above. The governing
  `docs/SPEC.md` v4.2 closure gate is not passed; do not begin or claim Phase 6
  completion from this work.
- Implemented origin-visible primary initialization in
  `src/scheduling/phase5.py`: trailing 13 complete observed weekly totals,
  at least 8 complete weeks, `ceil(mean) * (ceil(SOB-weighted lead days / 7) + 1)`,
  zero opening POs/backlog. D2 is retained as `d2_snapshot_stress`. Origin SBC
  labels used for exclusion reporting are recalculated from history available
  before each origin, not from the later frozen segment artifact.
- Added exact unresolved reasons for first-week lead-time infeasibility and
  MOQ/order-multiple rounding; firm status requires the per-part independent
  checker and shared supplier-capacity check to pass. Origin-wide failures are
  not copied onto unrelated parts. MRP and simulator ledgers reconciled in
  every executed path.
- `python -m src.scheduling.phase5 --demo` completed. Authoritative evidence is
  in `results/phase5_primary_v5_*`,
  `results/phase5_d2_snapshot_stress_v5_*`,
  `results/phase5_primary_sensitivity_v5.csv`,
  `results/phase5_scenario_comparison_v5.csv`,
  `results/phase5_repeat_run_evidence_v5.json`, and
  `results/phase5_mvp_planner_summary_v5.md`. Superseded intermediate outputs
  from this session were removed; pre-existing Phase 5 outputs were preserved.
- Primary initialization: **467 firm**, **93 unresolved**, **3 not ordered**;
  0 hard-constraint failures on firm schedules; 8 unresolved first-week
  lead-time failures and 85 unresolved supplier-capacity/MOQ-rounding and
  unfilled-requirement paths. There were 0 MRP and 0 simulator ledger failures.
  D2 stress: **256 firm**, **305 unresolved**, **2 not ordered**. Initial-cover
  medians by criticality, exclusions, monthly unresolved counts, sensitivity
  values, and the empty unsupported analogue diagnostic are recorded in
  `docs/phase5_assumptions.md`.
- The required same-component/same-vehicle-type short-series analogue remains
  **blocked**: the catalogue yields 0 eligible donors among 160 parts. The
  required CSV exists but has no forecast rows; no fabricated forecast was
  emitted. The v4.2 gate therefore remains partial.
- `py -3.10 -m pytest -q tests/test_phase5_scheduling.py` — **49 passed**;
  `py -3.13 -m pytest -q tests/test_phase5_scheduling.py` — **49 passed**.
  `python -m py_compile src/scheduling/phase5.py src/scheduling/engine.py
  src/scheduling/checker.py tests/test_phase5_scheduling.py`,
  Pylance diagnostics for `phase5.py`, and `git diff --check` passed.
- The full repository test suite and dataset validator were not run in this
  session. The prior validator invocation read `demand_qty` values from sealed
  final-test rows for integrity validation, not prediction scoring. No claim
  is made that final-test labels were never accessed. This session’s scheduling
  run used only saved held-validation forecast paths and origin-visible
  warm-up/train-validation history; frozen forecast artifacts were not changed.
- **Next command:** `python -m src.scheduling.phase5 --demo`. It replays the
  validation-only simulated schedule and validates existing output bytes
  without overwriting different content. Do not run the validator or proceed
  to Phase 6 until the fallback and remaining v4.2 gate evidence are resolved.

## Phase 6 comparison update (2026-11-15)

- **Phase 5 remains PARTIAL. Phase 6 is COMPLETE against its v4.2 gate under
  the user's explicit sequencing deviation. Phase 7 is NOT STARTED.** The
  Phase 5 analogue fallback is still blocked; this update does not close it or
  authorize Phase 7.
- Compared the frozen top two point candidates
  (`lightgbm_tweedie::random_1`, `lightgbm_quantile::random_1`) with the
  historical-average + D2 reorder-point baseline, MRP and risk-adjusted MRP.
  The common held-validation cohort produced 6,756 scheduler/candidate/
  initialization rows. No final-test labels, model training, tuning,
  champion reselection, Phase 5 rerun or dataset validator was used.
- Primary-initialization winner: `lightgbm_quantile::random_1` +
  `risk_adjusted_mrp`, weighted rank score **0.5125**. This does not change
  the frozen Track A point champion. The winner changed in 3/12 one-factor
  +/-10-point weight perturbations; paired bootstrap and 48 cost-sensitivity
  outputs were generated.
- The required forecast-error versus service/cost comparison is recorded in
  the report: Tweedie + MRP had the lowest weekly WAPE (45.02%), best fill
  rate (87.48%) and lowest emergency-cost proxy (18,100,933); Quantile + MRP
  had the lowest carrying-cost proxy (23,296,400). The lowest-error candidate
  therefore led on service and emergency cost, but not every cost component;
  the weighted winner remains Quantile + risk-adjusted MRP.
- The provisional 14-week schedule has 2,781 rows (1,961 firm, 820
  unresolved). Its 806 firm orders pass the aggregate independent checker;
  no firm row has failed checks. Validation comparison checker failures are
  unresolved-only (`release_before_week_1` and/or `unfilled_requirement`);
  ledger failures are zero. The D2 stress scenario has zero firm paths and
  remains explicitly reported as stress evidence, not hidden.
- KPI outputs include formula, unit, basis, denominator, traditional-baseline
  comparison and worse-than-baseline status. Phase-out inventory is reported
  separately. Obsolete parts P0046/P0087 lack D2 snapshots, so their stock
  values and weeks of cover are `not_measurable`.
- Schedule stability is unavailable from the saved one-run histories; the
  scoring component uses simulated schedule adherence only. Supplier-risk
  flag has no approved threshold and remains null. Synthetic results are
  method validation, not factory savings.
- Outputs: `results/phase6_scheduler_comparison_v1.csv`,
  `results/phase6_kpi_comparison_v1.csv`,
  `results/phase6_weight_sensitivity_v1.csv`,
  `results/phase6_paired_bootstrap_v1.csv`,
  `results/phase6_cost_sensitivity_v1.csv`,
  `results/phase6_initialization_comparison_v1.csv`,
  `results/phase6_exclusions_v1.csv`,
  `results/phase6_checker_summary_v1.csv`,
  `results/phase6_provisional_firm_schedule_v1.csv`, and
  `reports/phase6_comparison_report.md`.
- `py -3.10 -m pytest -q tests/test_phase5_scheduling.py tests/test_phase6_scheduling.py`
  and the corresponding `py -3.13` command: **59 passed each**.
  `py -3.10 -m py_compile src/scheduling/phase6.py
  tests/test_phase6_scheduling.py` and `git diff --check` passed. The full
  repository test suite was not run.
- **Reproduction command:** `py -3.10 -m src.scheduling.phase6`. It refuses
  to overwrite any existing Phase 6 output. The prior validator execution
  historically read sealed final-test labels for integrity checks; no claim
  is made that those labels were never accessed. Phase 6 used saved held
  validation forecasts and origin-visible warm-up/train-validation history.
- **Next:** do not begin Phase 7 in this checkpoint. Resolve the Phase 5
  fallback/gate and obtain explicit authorization before opening the final
  test or implementing Phase 7.

## Phase 7 readiness audit (2026-11-15)

- Phase 7 remains **NOT STARTED**. The readiness review was read-only: no Phase
  5/6/7 runner, dataset validator, tests, training, or final-test target values
  were accessed during that audit.
- Phase 6 handoff verified: current `configs/project.yaml` SHA-256 matches
  `46a6183450abad85c84ea3df41e412709e189b9314d42c8d1fb0c34a533314cc`;
  the saved primary winner remains
  `lightgbm_quantile::random_1` + `risk_adjusted_mrp` (score 0.5125), and the
  nine Phase 6 v1 comparison/KPI/sensitivity/bootstrap/cost/initialization/
  exclusion/checker/provisional-schedule outputs exist.
- The final-test inputs exist in the source dataset, but no Phase 7 runner or
  final-test report, final schedule, or final KPI output was found. The
  existing `results/forecast_future_14w_v41.csv` is the provisional bridge,
  not the final future forecast.
- Phase 5's analogue fallback remains a blocker for full short-series
  coverage: the diagnostic found zero eligible donors and emitted no forecast
  rows. The final forecast identity also needs confirmation: the frozen Track A
  point champion is Tweedie, while the Phase 6 winning pair is Quantile +
  risk-adjusted MRP.
- The v4.2 Phase 7 clause does not define an executable command, final-test
  horizon/origin procedure, quantitative performance threshold, or retry
  behavior. The prior validator's structural read of sealed `demand_qty`
  values is documented above; it was not prediction evaluation, and no claim
  is made that the labels were never accessed.
- **Readiness decision: NOT READY to consume the final test.** Resolve the
  Phase 5 sequencing/fallback issue and Phase 7 protocol/model decisions, then
  obtain explicit authorization for the single-use evaluation. Do not reuse
  the same final-test window if it has been consumed and a change is required;
  establish a new evaluation window instead.

## Approved Phase 5 no-donor exception (2026-11-16)

- The user approved preserving the exact `component` + `vehicle_type` donor
  rule and recording series without a donor as
  `UNFORECASTED_NO_DEFENSIBLE_ANALOGUE`. No broadened metadata match, invented
  quantity, or zero-demand substitution is allowed.
- The exception covers 47 unclassified part-region series across P0007, P0009,
  P0024, P0025, P0042, P0062, P0068, P0090, P0091, P0092, P0106, and P0122.
  At each held origin (2025-01-13, 2025-04-21, 2025-07-28, 2025-11-03), the
  exact-rule eligible donor count is zero. P0091 remains unforecasted because
  it has no development observations and no validated launch/exposure scaling.
- `docs/SPEC.md` sections 3 and 12(c) now state the approved exception.
  Versioned detail is recorded in
  `results/phase5_analogue_exceptions_v1.csv`,
  `results/phase5_analogue_exceptions_v1.json`, and
  `results/phase5_analogue_exception_summary_v1.md`. Existing v5 Phase 5/6
  outputs and the legacy empty `analogue_fallback_forecasts.csv` are preserved.
- The Phase 5 status remains PARTIAL until the updated implementation and
  focused tests are reviewed and the remaining gate evidence is accepted. The
  Phase 5 gate requires the hand cases and existing suite to pass on Python
  3.10 and 3.13, validator 32/32, zero hard violations on firm schedules,
  inventory reconciliation, identical repeat-run evidence, and frozen
  assumptions. No tests or validators were run for this update. Do not run
  the final-test validator; its prior documented invocation read sealed-test
  targets for integrity checks.
- Phase 6 remains COMPLETE under its previously approved sequencing deviation;
  its configuration hash and artifacts were not changed. Phase 7 remains NOT
  STARTED.

## Phase 7 runner implementation and safe preflight (2026-11-16)

- Implemented the frozen Phase 7 protocol in `configs/phase7.yaml` and the
  fail-closed runner in `src/evaluation/phase7.py`. Operations are separated
  into protocol validation, target-free preflight, explicitly authorized
  single-use final-test evaluation, and separately authorized post-evaluation
  finalization. The evaluation attempt marker is created exclusively before
  target-bearing reads; a failed consumed attempt is not retryable.
- Added fitting-role support to `src/forecasting/global_models.py` while
  preserving the existing `train_val` default, and an optional horizon/run-id
  to the Phase 6 scheduling helper while preserving its existing defaults.
  Added synthetic safeguards in `tests/test_phase7_runner.py`.
- Safe preflight result: **BLOCKED** because Phase 5 remains PARTIAL. The
  metadata-only split check confirmed train-validation ends 2026-03-23,
  final-test spans 2026-03-30–2026-09-21 (26 weeks), and future spans
  2026-09-28–2026-12-28 (14 weeks). Output destinations were available.
  Preflight reported `target_values_read: false`,
  `target_columns_requested: []`, `outcome_columns_requested: []`, and
  `evaluation_consumption_state: NOT_CONSUMED`. The D2 snapshot is
  2026-09-26, after the test window; scheduling results must be described as
  simulated policy comparisons, not historical operating-state evidence.
- Verification actually run: `py -3.10 -B -m pytest -q -p no:cacheprovider
  tests/test_phase7_runner.py` — **6 passed**; no final-test data was used by
  these synthetic-fixture tests. In-memory Python syntax compilation passed
  for the changed Python sources. `git diff --check` passed with a pre-existing
  LF/CRLF working-copy warning for `docs/Spec.md`.
- No final-test evaluation, finalization, training, Phase 5/6 execution,
  validator, or full repository test suite was run. No Phase 7 evaluation or
  finalization output was created. Phase 7 final-test evaluation remains
  **NOT STARTED** and is blocked pending Phase 5 closure and separate user
  authorization.

## Phase 7 targeted safety corrections (2026-11-16)

- Finalization now requires a completed marker and saved evaluation metadata
  with explicit boolean target-consumption fields, matching run/protocol
  identities, and matching frozen configuration and split hashes. Frozen
  configuration and split metadata are revalidated before training data loads.
- Authorized target reads now validate split metadata first, inspect only
  future-role targets for the required blank-target invariant, and load
  target-bearing rows only for warmup/train-validation through the origin and
  the approved final-test window. `out_*` columns remain outside the read
  allowlist. Populated targets on flagged missing weeks fail closed; training
  also explicitly masks those rows while retaining observed zeros.
- The Phase 5 dependency gate now requires the current top-level Phase 5 status
  and a structured `results/phase5_exit_gate_evidence_v1.json` record covering
  the required exit criteria. That evidence record is not present and Phase 5
  remains **PARTIAL**; no acceptance decision or Phase 5 status was changed.
- The report explicitly explains that lead-time metrics are a single
  origin-anchored 26-week aggregate; weeks 15–26 remain a weekly extended
  diagnostic. The D2 post-window limitation remains stated.
- Focused synthetic-only verification: **19 Phase 7 tests passed** on Python
  3.10; in-memory syntax compilation and `git diff --check` passed. No final-test
  targets, Phase 5/6/7 runs, validators, finalization, or full test suite were
  accessed or executed.   Phase 7 remains unevaluated and unfinalized.

  ## Phase 8A — part-category classification (2026-11-16)

  - **Status: COMPLETE for the specified method-validation gate, with automatic
    filling disabled by the confidence gate.** This is validation on the
    synthetic catalogue only, not evidence of production accuracy. At the Phase
    8A checkpoint, Phase 8B and Phase 9 had not started.
  - Read `docs/SPEC.md` v4.2 §10, this state file, the updated problem statement,
    the D3 schema, loader, generator, and classifier code before implementation.
    The original problem-statement PDF is absent; the updated PDF is present.
  - Replaced the unsuitable row-stratified evaluation with the SPEC baseline:
    character 2–5-gram TF-IDF and balanced Logistic Regression, seed 42. The
    only model inputs are the documented allowlist `part_name`, `component`,
    and `material`. `part_category`, IDs, `part_description` (which embeds
    category text in generated data), outcome/target fields, and all other
    attributes are excluded. D3 contains 160 labeled rows, 42 components, and
    8 categories; no labels were missing or unknown.
  - Evaluation: deterministic 31-development-component / 11-held-out-component
    split (41 holdout parts, zero component overlap), then 5-fold
    `StratifiedGroupKFold` on development components only. Every development row
    received one OOF prediction. Grouped OOF: accuracy **27.73%**, macro-F1
    **0.2106**, weighted-F1 **0.2648** (119 rows). Held-out components:
    accuracy **24.39%**, macro-F1 **0.0694**, weighted-F1 **0.2168** (41 rows;
    5 of 8 classes present). Per-fold, per-class support, confusion matrices,
    and class/component imbalance are recorded in the versioned report/CSVs.
  - The development-only OOF threshold search found no candidate meeting the
    configured 80% selective-accuracy floor. Therefore threshold is 1.0 and
    `auto_accept_enabled` is false: missing/unknown categories are predicted
    for review but remain unresolved and require clarification; existing
    catalogue categories remain authoritative and are never overwritten.
    Conflicts are returned explicitly and require clarification. The supplied
    synthetic catalogue contains no missing labels, so fill performance cannot
    be measured.
  - The saved serving model was loaded and exercised. The report, model,
    preprocessing artifact, class mapping, metadata/config/data hashes, grouped
    fold results, threshold table, held-out predictions, per-class results, and
    confusion matrix are saved under versioned `phase8a_v1` paths:
    `configs/part_category.yaml`,
    `artifacts/models/part_category_phase8a_v1.joblib`,
    `artifacts/models/part_category_preprocessor_phase8a_v1.joblib`,
    `artifacts/models/part_category_class_mapping_phase8a_v1.json`,
    `results/part_category_phase8a_*_v1.*`, and
    `reports/part_category_phase8a_v1.md`. Existing legacy classification
    results and source data were preserved. `PartCategoryPredictor` and its
    result types are available through a lazy package-level export.
  - Reproducibility check: reran the pipeline against copied config and D3 data
    in an isolated temporary workspace; grouped-CV and held-out accuracy,
    macro-F1, weighted-F1, and sample counts matched the saved outputs exactly.
  - Verification: focused tests **11 passed on Python 3.10** and **11 passed on
    Python 3.13**; in-memory syntax compilation passed; `git diff --check`
    passed with LF/CRLF working-copy warnings for `docs/Spec.md` and
    `src/classification/__init__.py`.
    Python 3.13 emitted eight SciPy L-BFGS deprecation warnings. The grouped
    splitter warned that the rarest class has only four rows, fewer than the
    five folds; this limitation is documented. No full repository suite,
    validator, Phase 8B, or Phase 9 was run.

## Phase 8B — Django REST API and controlled Ask AI (2026-11-16)

- **Status: COMPLETE for the Phase 8B implementation gate. Phase 9 has not
  started.** The application remains a simulated planning system; it is not
  connected to an ERP or purchase-order service.
- Implemented one Django/DRF backend with SimpleJWT, SQLite development default,
  environment-selected MySQL/PostgreSQL, and direct in-process imports of the
  existing scheduling and simulation packages. Role enforcement is server-side
  for viewer, planner, approver, and admin actions. Refresh rotation, blacklist
  revocation, password validation, paginated result APIs, and environment-only
  `DJANGO_SECRET_KEY` are configured.
- Added relational models and migrations for users/roles, catalogue parts,
  planning runs, versioned forecast/schedule references, warnings, KPIs,
  conversations/messages, proposals/immutable versions, and append-only
  hash-chained audit records. Results are path-constrained and hash-checked
  before service reads.
- The controlled Ask AI fallback supports English, Hindi, Hinglish, and
  Marathi. It parses to intents and calls allowlisted services only; it has no
  direct database/model write access and cannot auto-apply or calculate order
  quantities. Proposal quantities use deterministic supplier MOQ/order
  multiples and are rechecked with the existing hard-constraint checker at
  confirmation. Confirmation/revert create new versioned schedule files and
  recalculate affected-part inventory/KPI impacts using
  `src.simulation.weekly.simulate_weekly`; these are explicitly simulated
  comparisons over saved forecast paths.
- Added 192 generated labeled paraphrase requests (128 development, 64 held
  out). Held-out intent accuracy **1.000**, applicable entity accuracy **1.000**
  (16 cases), blocked-intent recognition **1.000** (16 phrase fixtures),
  cross-language consistency **1.000** (16 per language), incorrect auto-apply
  **0.000**, invented-quantity **0.000**, median/p95 local parser latency
  **0.021/0.032 ms**, external model calls/cost **0/0**. These are generated
  paraphrase-fixture results, not human-validated multilingual performance.
  No fluent-human review was performed, so claims are limited to paraphrase
  robustness. The offline blocked-intent metric measures intent labels only;
  authorization and hard-constraint enforcement are covered by API tests.
  Audit completeness is not an offline metric; API tests verify audit chaining
  and lifecycle records.
- Focused synthetic/API verification: `py -3.10 -B backend/manage.py test
  backend.tests.test_phase8b_api -v 1` — **15 tests passed**. Coverage includes
  roles, authentication/revocation, proposal confirmation and duplicate/stale
  rejection, a two-worker version compare-and-swap, hard-constraint rejection,
  prompt injection, cancellation, revert/idempotency, audit integrity, result
  pagination, assistant fixtures, and SQLite backup/restore. Django system
  check passed; `makemigrations --check --dry-run` reported no changes;
  Pylance syntax checks passed for changed Python modules; `git diff --check`
  passed (only pre-existing LF/CRLF working-copy warnings were reported).
  The full repository suite and dataset validators were not run.
- Deliverables include `backend/` (Django project/apps, services, APIs,
  migrations, and tests), updates to `requirements.txt`, development and
  evaluation instructions in `README.md`, and versioned
  `results/assistant_requests_phase8b_v1.csv`,
  `results/assistant_evaluation_phase8b_v1.json`, and
  `reports/assistant_evaluation_phase8b_v1.md`. The original problem-statement
  PDF was not present; the updated PDF is available. Existing unrelated dirty
  work and artifacts were preserved.
- Regression command: `py -3.10 -B backend/manage.py test
  backend.tests.test_phase8b_api`. At this checkpoint, Phase 9 had not yet been
  authorized; see the Phase 9 entry below for later work.

## Phase 9 — planning website, monitoring, and robustness (2026-11-16)

- **Status: PARTIAL.** The React/TypeScript website is implemented and verified
  as the primary demonstration deliverable. Phase 9 is not complete: offline
  retraining with gated promotion, saved-champion robustness evidence, and the
  full repository test gate remain open. Phase 5 remains PARTIAL; Phase 7
  remains unevaluated and unfinalized.
- Established a shared responsive design system and application shell, then
  implemented API-backed Overview, Forecasts, Firm Schedule, Inventory & Supply,
  KPI Analytics, Ask AI, Audit, Run History, and Settings pages. Pages display
  saved results only; the assistant's read-only browser query returned saved
  schedule evidence without creating a proposal. Proposal application remains
  an explicit role-checked action.
- Added forecast bias and tracking-signal alert presentation driven only by
  saved KPI values. The bias alert uses the configured ±10% threshold;
  tracking-signal ±4 is explicitly a provisional display assumption, not a
  frozen acceptance rule. The registered run has no tracking-signal record and
  therefore shows it as not measurable.
- Added `tests/test_phase9_robustness.py`. Its three synthetic-only tests passed:
  doubled point/P99 demand increased order quantity with a feasible checker
  result; supplier OTD/delay inputs changed risk-adjusted lead time from 7 to 21
  days with a feasible checker result; infeasible capacity remained unresolved
  and was rejected by the checker. These test engine behavior, not champion-path
  robustness or business performance.
- Frontend verification: 6 Vitest tests passed; TypeScript check and Vite build
  passed in 0.99 seconds (main JavaScript 299.08 kB, 95.05 kB gzip; chart chunk
  365.32 kB, 105.39 kB gzip). Browser smoke review covered all nine routes.
  At a 390 px viewport, the body had no horizontal overflow; the schedule table
  remained scrollable within its own container. Exact evidence and saved Phase 6
  KPI table are in     `reports/02_phase9_closeout_v1.md`; machine-readable measurements are in
  `results/03_phase9_ui_validation_v1.json`, and the short summary is in
  `reports/01_phase9_executive_summary_v1.md`.
- The selected Phase 6 `origin_visible` KPI rows in
  `results/phase6_kpi_comparison_v1.csv` show WAPE 46.35% versus 57.46% baseline,
  forecast bias -34.60% versus -1.09%, fill rate 83.89% versus 84.77%, stock-out
  rate 10.48% versus 9.59%, critical availability 94.56% versus 91.76%, and
  carrying-cost proxy 25,601,949.91 versus 9,297,403.88. These are saved
  train/validation and simulated comparisons, not final-test results or proven
  savings.
- The full repository `pytest` suite was not run: existing tests use
  `load_all`/`load_demand` on D1, which would load sealed target-bearing rows.
  Only focused frontend tests/build and synthetic Phase 9 scheduling tests were
  executed. No Phase 5/6 pipeline, validator, or final-test evaluation was run;
  no final-test target values were accessed in this Phase 9 work.
- **Next executable command:** from the repository root,
  `Push-Location .\frontend; npm run dev` starts the local UI demonstration.
  Before closing Phase 9, implement and validate offline retraining/promotion
  safeguards, produce champion-path robustness evidence without opening the
  final-test window, and obtain an approved safe full-suite test strategy.
