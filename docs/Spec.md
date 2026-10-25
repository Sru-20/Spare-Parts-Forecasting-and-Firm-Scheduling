# Spare Parts Demand Forecasting & Firm Scheduling: SPEC v4.1 (frozen after this amendment)

**Stack:** React + TypeScript, Django + Django REST Framework, one Python package for forecasting and scheduling, one relational database (SQLite in dev, MySQL or Postgres in production). No second backend, no queue, no extra services.
**Sources of truth:** both problem-statement PDFs, the three CSVs, `generate_datasets.py`, `validate_datasets.py`, the 23 Oct progress report. If a file disagrees with this spec, report it and follow the file.
**Goal:** Right Part + Right Quantity + Right Time + Right Inventory, shown by measured KPIs on synthetic data. Say "method validated on simulated data", never "proven savings".

## 0. What changed from v3 (v4 is v3 made simpler and safer)
| Area | v4 change |
|---|---|
| Forecast scope | smaller required set; ensemble and router only if validation shows a repeatable gain; Phase 3 results are kept, nothing is thrown away |
| Statistics | 4 validation origins is little evidence: rank and show uncertainty; gates are configurable rules, not automatic rejection; no "significant" claims without a bootstrap interval |
| Scheduling | inventory equation, timing convention and `firm`/`unresolved` defined **before** any scheduler; MRP first; CP-SAT only if a measured gap remains |
| Security | replaced with one short, testable checklist, release-blocking |
| Operations | every run is versioned (`run_id`); dashboard never mixes runs |
| Scope | three stages: planning core, web app, controlled AI |
| Learning loop | offline retrain and gated promotion, never live model changes |
| Naming | outputs are called a **simulated** firm schedule |
**v4.1 amendments (final freeze):** metric units fixed; weekly timing convention tested; inventory order of operations; factor traceability appendix; proposal lifecycle rules; KPI basis labels in API/UI; bias-overlap rationale. Any later change needs a logged reason in `STATE.md`.
Unchanged from v3: Django backend, manufacturing priorities, Track A/B/C weights, selection order (gates, weights, downstream tie-break), cost proxies, final test in its own phase.

## 1. Goal and priorities
Build an app that forecasts part/region demand for 14 weeks, turns it into a feasible schedule, reports the 10 PDF KPIs and early warnings, and offers a multilingual Ask AI that proposes but never applies a change without explicit confirmation. Compare against a traditional baseline (historical average + current policy/MRP).

Manufacturing priorities, in order: (1) critical-part availability, (2) emergency procurement, (3) excess and obsolete stock, (4) schedule stability, (5) real supply limits (MOQ, multiples, lead time, capacity, SOB), (6) accuracy at lead-time level.
Non-goals: ERP/supplier integration, real purchase orders, microservices, deep-learning model zoo, claims of real savings.

## 2. Data rules (all unchanged and still binding)
- Files: D1 `01_demand_forecasting.csv` (120,861 x 62), D2 `02_scheduling_inputs.csv` (158 x 61, one snapshot), D3 `03_spare_parts_catalogue.csv` (160 x 29). Validator 32/32; tests pass on Python 3.10 and 3.13.
- Splits come from `split_role` (warm-up, train/validation to 2026-03-23, sealed final test, 14 future weeks).
- Never modify originals; derived data goes to `artifacts/`.
- `is_missing_week = 1` is excluded from fitting and scoring, never zero-filled. Genuine zeros stay. Future rows are prediction-only.
- **Forbidden features:** `out_*` columns, `intended_sbc_class`, `demand_driver_tags`, `lifecycle_pattern` (unless proven known at forecast time). Every lag, rolling value, encoder and scaler uses only data available at the origin.
- Join by documented keys, test row counts; never duplicate weekly rows.
- 47 short series: label `Unknown/ShortHistory`, flag low confidence, forecast with an analogue (same `component` and `vehicle_type` seasonal profile). Never infer a class from test or future data.
- Obsolete parts (P0046, P0087): no new orders. Target: `demand_qty`, scored on raw demand; one validation-only check on `demand_qty_capped`.
- **Known-in-advance rule (record in Phase 4):** festival calendar, plant calendar/shutdowns and promotions/campaigns published at the origin are usable; unplanned market events are not. Use direct per-horizon forecasting, or recursive using only forecast or known-in-advance inputs; state which.

## 3. Architecture, efficiency and security

### Architecture
- `backend/` Django + DRF + SimpleJWT: apps `accounts`, `planning`, `assistant`. Django owns auth, transactions, proposals, audit.
- `ml/` plain Python (data, features, forecasting, scheduling, simulation, evaluation). Django imports it directly.
- `frontend/` React + TS with hooks; no Redux, no SSR.
- `configs/` holds every business parameter (weights, gates, costs, service targets, horizon, languages, latency budget). Data is read through one schema-validated loader.

### Efficiency and scale (keep it simple)
- Forecasts, schedules, KPIs and warnings are **batch jobs** (Django management commands) that save versioned results. Dashboards read saved results only; they never run inference.
- Only `propose_change` computes live (budget in config, default 5 s).
- Paginate every list; index part, region, week and run_id; `select_related`/`bulk_create`; Parquet for big intermediate tables; lazy-load pages and virtualize long tables.
- Stateless API, so more workers can be added behind one reverse proxy. Add caching only for an endpoint measured to be slow, with a clear invalidation rule. No Redis, Celery, Kubernetes or extra services unless measured runtimes require them.
- Record batch runtime and endpoint latency in `results/performance_log.csv`.

### Run management (reproducibility)
Every forecast and schedule run writes one `runs` record: `run_id`, timestamp, data version (file hash), config version (hash), model version, horizon, scheduling policy, checker version, status, duration, error, output location. The API serves a forecast and schedule only together under one `run_id`. Retraining is offline; a new model is promoted only after beating the current champion on the fixed validation benchmark. Nothing changes live.

### Security checklist (each item has a test; the release-blocking ones are in section 11)
- **Auth:** server-side permission on every endpoint and object. Roles: viewer (read), planner (propose), approver (confirm), admin. Planners cannot confirm; users confirm only proposals they are allowed to approve. Short token expiry, refresh and logout revoke tokens, password rules via Django validators.
- **Django hardening:** `DEBUG=False`, explicit `ALLOWED_HOSTS`, narrow CORS and CSRF origins, HTTPS and secure cookies in production, security headers, secrets only in environment variables, locked `requirements.txt`, `pip-audit` before release.
- **Input:** DRF serializers validate type, range, ID and size on every body; ORM only, no string-built SQL.
- **Rate limits:** login, Ask AI, `propose_change`.
- **AI isolation:** prompts, imported data and model replies are untrusted. The model has no database access and cannot bypass the scheduling engine. It may only produce typed intents; the server re-validates part IDs, quantities and permissions deterministically. No credentials or unnecessary business data are sent to an external provider. On parse failure, timeout or ambiguity: ask for clarification or use the rule-based fallback.
- **Proposal integrity and lifecycle:** a proposal records the schedule version it was calculated on and gets its own version. Confirmation is valid only if (a) the schedule version and proposal version still match, (b) the proposal has not expired (default 60 minutes, configurable), (c) the confirming user holds the approver role **now**, and (d) the approver is **not the proposer** (`allow_self_approval=false` by default; if enabled, it is flagged in the audit entry). If the schedule changed after the proposal was created, the proposal becomes `stale`, cannot be confirmed, and must be recalculated. Apply in one transaction; reject duplicate confirmations; idempotency key on retries; concurrent edits tested. A revert is a new, separately authorized change that links to the original audit entry; original entries are never edited or deleted. A revert is allowed only if the schedule has not changed since the change it undoes; otherwise a new proposal is required.
- **Audit:** append-only (no update or delete through the API), records who proposed, approved and applied, before/after values, validation results, timestamps; secrets redacted.
- **Backup:** documented backup and a restore test with integrity checks.
This is a strong student-project baseline, not a substitute for a professional security review.

## 4. Scope in three stages
1. **Planning core:** data checks and leakage tests, baseline and selected ML forecasts, 14-week output, deterministic MRP, independent constraint checker, one simulator, KPIs, reproducible result files.
2. **Web app:** Django APIs, roles, React dashboards over stored results, warnings, audit history, security tests.
3. **Controlled AI and extras:** read-only Ask AI first, then typed proposals, validation, confirmation, cancel/revert, audit. Only then: multilingual evaluation depth, CP-SAT, extra models, Tamil.
Each stage must work before the next starts. Optional items are added only to fix a measured weakness.

## 5. Forecasting

### Required set
Naive; seasonal naive; one intermittent baseline (SBA or TSB, whichever validates better); global LightGBM **point** (Tweedie or Poisson loss, chosen on validation; Tweedie suits zero-heavy counts); LightGBM **quantile** (P50, P90, P95, P99). The equal-weight ensemble already exists in your results and stays as a candidate. The SBC router and anything else (moving average, CatBoost, ETS) are added only if validation shows a meaningful, repeatable gain; otherwise they stay in the backlog. Keep every run, including failures, in `results/track_a_candidates.csv`.
Tuning: same fixed budget per model, tuning origins only, seed 42. Router rules, if used, are learned on training/tuning data only and validated as one whole policy.

### Roles
- Point forecast drives netting, MRP and KPIs.
- Quantile forecast drives safety stock and early warnings. It is judged on coverage and pinball loss, and on P50 for point metrics, never on signed bias.
- Check quantiles do not cross (P50 <= P90 <= P95 <= P99). If they do, sort them or apply a documented correction and evaluate it on validation.

### Features (16 PDF factors only)
Lags 1/4/13/52 (test lag 52 for leakage); rolling mean/median/std from past values; week, month, fiscal calendar; festival flag and weeks-to-festival; holiday, shutdown, working days; published promotions/campaigns; region, category, criticality, ABC, fulfilment mode, part type; vehicle type/model seasonality; zero-run length and time since last demand. No vehicle parc/age unless known at origin; no substitution groups; only an obsolete flag from lifecycle.

### Exact metric definitions (write these into `configs/` and tests)
- **WAPE** = sum|A - F| / sum A; if sum A = 0, report "undefined" and exclude from averages, never 0.
- **Signed bias %** = (sum F - sum A) / sum A. Positive means over-forecast. Same zero rule.
- **Asymmetric bias** = |bias| if bias >= 0, else `under_weight` x |bias|; `under_weight` is a configurable scenario (default 2), not a verified business cost.
- **MASE** scales by the in-sample one-step naive error of the training period; undefined for an all-zero training series (excluded and counted).
- **Lead-time demand** for a part = sum of weekly forecast and actual over ceil(lead time in days / 7) weeks from the origin; lead-time WAPE compares those two sums. Parts with lead time above 14 weeks use the 14-week horizon plus a labelled extension rule (last-period average).
- **Units:** store and compute all error metrics as **ratios** (WAPE 0.6926 means 69.26%); convert to percent only for display. **Forecast Accuracy %** = 100 x (1 - WAPE), the requested project KPI, always shown beside WAPE (0.6926 gives 30.74%); it can go negative when WAPE exceeds 1 and is not a probability of being right. API fields are named with their unit (`wape_ratio`, `accuracy_pct`) and tests check the conversion.
- **Forecast vs actual variance** = signed and absolute difference, per part and period.
- **Coverage** = share of actuals at or below the quantile, reported by criticality and horizon **with sample counts**; groups with few observations are labelled "insufficient data".
- Every metric is reported weekly and lead-time-aggregated, overall and by SBC class, ABC, criticality, region, category and horizon step. When they disagree, lead-time results and intermittent-series service outcomes take precedence.

## 6. Selection: simple and honest

### Order of decisions
1. **Rules (configurable, shown as pass/fail with uncertainty).** Default rules: beats seasonal naive and the historical-average baseline on lead-time WAPE; overall |bias| <= 10%; critical under-forecast bias no worse than -10%; for quantile models, P95 coverage within +/-5 points on critical parts. With four validation origins these are screening rules: report each result with a bootstrap interval (resample series and origins together). Set `gates.enforce` in config; default is to enforce only the baseline-beating rule and to flag, not reject, the others when the interval includes the threshold. If evidence is inconclusive, keep the simpler model. Do not say "significant" unless the interval supports it. Diebold-Mariano is optional and reported with this limit stated.
2. **Weighted score** among the models that pass (rank normalization within the eligible set; ties on simplicity, explainability, runtime).
3. **Sensitivity and tie-break.** Perturb each weight by +/-10 points (renormalized). If the top model wins in at least 75% of perturbations it is the champion; otherwise report the contenders and break the tie by **downstream simulation** (service level and cost on the validation period). The lowest forecast error is not automatically the best planning model.
Weighted scores are selection scores, not forecast percentages or savings.

### Track A weights (editable assumptions)
| Component | Weight |
|---|---|
| Overall error (70% lead-time WAPE, 30% weekly WAPE) | 20 |
| Critical/A-class error (lead-time WAPE + asymmetric bias) | 20 |
| Intermittent/lumpy (lead-time WAPE and MASE, 50/50) | 15 |
| Asymmetric absolute bias | 15 |
| Calibration (coverage error + pinball; no quantiles = minimum score) | 15 |
| Stability across validation origins | 10 |
| Runtime and explainability | 5 |
| **Total** | **100** |

Note: asymmetric bias appears in both the critical-part component and its own 15-point component, so under-forecasts on critical parts are deliberately penalised twice, because they are the costliest failure. Run the weight sensitivity before finalizing and report the winner with that overlap removed as an extra scenario.

### Track B weights (editable assumptions)
Critical-part availability 25; overall fill rate / stock-out % 15; emergency procurement 15; excess inventory and carrying cost 20; schedule adherence and stability 15; turnover 10 (total 100). A hard-constraint violation disqualifies a scheduler.

### Track C weights
Intent accuracy 25; entity accuracy 25; rule-validation correctness 20; multilingual consistency 15; audit completeness 10; latency/cost 5. Disqualifiers: change applied without confirmation, invented quantity.

### Business defaults (used until the business team replies; edit in `configs/business.yaml`)
Priority: critical stock-outs, emergency procurement, excess, overall error. Service targets come from D2 `service_level_target` (never invented). Cost proxies, labelled: emergency cost = unit cost x 1.5 (test 1.25 to 2.0); stock-out penalty = unit cost x 10 / 5 / 2 for high / medium / low criticality (vary +/-50%); carrying cost from the supplied rate; downtime cost is "not measurable". Report whether the winner changes across these ranges.

## 7. Scheduling: define the rules before writing the engine

### Inventory contract (one convention for scheduler and simulator)
- Time unit: week. Week 1 is the first future week.
- **Order of operations each week (same in scheduler and simulator):** (1) receive scheduled deliveries; (2) add receipts to opening stock; (3) allocate available stock first to existing backlog (critical parts) then to current demand, in that order; (4) record fulfilled quantity and the remainder as backlog (critical parts, carried forward) or lost sales (others); (5) compute closing stock and reconcile: `Closing = Opening + Receipts - Fulfilled`, `Fulfilled = min(Opening + Receipts, Demand + Backlog)` for critical parts and `min(Opening + Receipts, Demand)` for others. Receipts due in week t arrive at the start of week t, before demand.
- Unfulfilled demand is tracked separately: **backlog** (carried forward) for critical parts, **lost sales** for others, unless the data says otherwise. Closing stock is never negative; negative inventory is never treated as real stock.
- **Timing:** an order released at the start of week r arrives at week r + ceil(lead time in days / 7); this is deliberately conservative (even a 1-day lead time arrives the next week) and is the single convention for scheduler and simulator. Release is no earlier than week 1. Expected arrival weeks are tested: lead time 1 day gives r+1, 7 days r+1, 8 days r+2, 14 days r+2. An order whose arrival falls after week 14 is kept, marked `arrives_beyond_horizon`, and not counted as supply inside the horizon. An order that cannot arrive in time to cover a shortage is reported as late, not hidden.
- Order quantity = max(net requirement, MOQ) rounded up to the order multiple. Pack quantity noted.
- Usable stock = on-hand - reserved - blocked, checked against `available_qty`. Open POs (due this month plus due later) and in-transit are counted once. D2 has no pending-requirement column: use the supplied backorder field as a labelled assumption.
- Safety stock: quantile-based over the lead-time window at the part's service level, with the long-lead-time rule above. Capacity and committed capacity are in the same unit as order quantity; allocation across up to three suppliers follows SOB as a **target** (deviations reported) unless the data proves it is a binding contract.
- `firm` = every hard constraint checked and passed by the independent checker (MOQ, multiple, capacity, lead time, non-negative stock, no new orders for obsolete parts, no import supplier for production parts). Anything else is `unresolved` with the failed check and reason. Never replace an infeasible result with arbitrary quantities.
- Call the output a **simulated firm schedule** in the report and UI. It is decision support on synthetic data and a single inventory snapshot, not an operational commitment or a replay of real inventory.

### Hand-calculated tests (write before comparing schedulers)
(1) enough stock, no order; (2) shortage covered by a correctly timed receipt; (3) order that misses its date because of lead time; (4) MOQ and multiple rounding; (5) capacity exhausted; (6) obsolete part gets no order; (7) backorder or open PO not counted twice; (8) lead times of 1, 7, 8 and 14 days give the expected arrival weeks and a beyond-horizon case; (9) critical part with existing backlog and insufficient stock for both backlog and demand (backlog served first, remainder carried forward); (10) ordinary part in the same situation (remainder lost, no carry-forward). Plus: inventory conservation, determinism for seed 42, and the checker rejecting deliberately invalid schedules.

### Schedulers
Build **deterministic MRP first**, with the baseline policy (historical-average forecast + simple reorder). Add (s,S)/quantile safety stock as a variant if cheap. Add one CP-SAT model only after MRP works and a measured limit (for example multi-supplier capacity conflicts) justifies it; MRP must work without a solver. Run all schedulers on identical demand paths and frozen assumptions. **Forecast-by-scheduler check:** top 2 forecast candidates x eligible schedulers on the validation period; choose the winning pair before the final test and report whether the lowest-error forecast also gives the best service and cost.

### Schedule columns
`run_id, part_id, part_name, region/plant, week, forecast_qty, opening_usable_stock, expected_receipts, projected_inventory, backlog_qty, net_requirement, safety_stock, service_level_target, supply_mode, supplier_id, order_qty, release_week/date, due_date, MOQ/multiple adjustment, capacity_allocation, SOB_target, criticality, priority, reason, warnings, status (firm | unresolved), failed_checks`.

## 8. The 10 KPIs
Each shows formula, unit, baseline, denominator and status (`measured`, `proxy`, `not measurable`); same demand path, initialization and supply assumptions for baseline and proposed. Never turn a zero baseline into a percentage: show the absolute change or "not defined". List every case where the system is worse than baseline.
- Every KPI value in the API and UI carries a `basis` field: `measured` (from data), `simulated` (simulator outcome), `proxy` (uses an assumed cost) or `not_measurable`. Simulated and proxy values are never shown as observed real-world improvements.
- Obsolete and excess stock (PDF outcome "reduce excess and obsolete inventory"): report phase-out/obsolete parts' stock value and weeks of cover separately from ordinary excess, before and after.
- Measured on data: Forecast Accuracy % (beside WAPE), Forecast vs Actual Variance, Forecast Bias.
- Simulator outcomes (proxy): Stock-Out %, Service Level/Fill Rate, Schedule Adherence %, Emergency Procurement Reduction, Excess Inventory Reduction, Carrying Cost Reduction.
- Inventory Turnover: only if demand/COGS and average inventory value are both available; otherwise not measurable. Downtime cost: not measurable.

## 9. Ask AI (controlled, multilingual)
Build in two steps: **(a) read-only** questions (schedule, forecast, inventory, warnings, KPI explanations, find parts by category), then **(b) typed proposals**.
Workflow: request -> intent and entities -> fetch schedule -> deterministic proposal -> validate (MOQ, SOB, lead time, capacity, inventory, criticality, service level) -> show before/after, impact, warnings -> explicit confirmation -> apply in a transaction -> recalculate KPIs -> audit entry. Example: "increase electrical-part order quantities by 10% for the next schedule". Ambiguous or violating requests get an explanation and a clarifying question. Replies are in the user's language where feasible; if not, reply in English and say so. Users need no predefined commands. Cancel and revert supported; changes affect only the simulated schedule. "Looks good" in chat is not a confirmation; only the explicit UI action is.
Languages: English, Hindi, Hinglish, Marathi (Tamil optional later). Evaluation: at least 150 labelled requests (valid, ambiguous, unknown part, bad percentage, constraint-violating, mixed-language, typo, cancel/revert, bypass attempt, prompt-injection attempt, the PDF example); development and test cases kept separate; fluent human spot-check of at least 20% per language, otherwise state that the set measures paraphrase robustness only. Report accuracy per language, entity accuracy, validation correctness, audit completeness, latency. **Incorrect auto-apply rate and invented-quantity rate must be 0.** Tiers: rules baseline, one multilingual classifier, one LLM with structured output (if no API key, use and document the fallback).

## 10. API, pages and early warnings
Endpoints (versioned `/api/v1/`): parts, forecasts, schedules, kpis, warnings, assistant/messages, proposals/{id}/confirm|cancel|revert, audit, runs. Views validate and delegate to services.
Pages: overview and risks; forecasts (actual vs forecast, error); simulated firm schedule (firm and unresolved visibly different); inventory and supply; KPI comparison with measured/proxy/not-measurable labels; Ask AI; audit; run history.
Warnings: likely stock-out within lead time, demand surge/decline, supplier capacity shortfall, delayed orders, critical-part exposure, excess or obsolete stock risk. Each shows part, period, risk, evidence and recommended action from validated rules.

## 11. Phases, gates and release tests
Phases 0 to 3 are complete (data, EDA, features, candidate evaluation). Do not redo them.

- **Phase 4. Champion.** Fill gaps in section 5 only if needed. Apply rules, weights, sensitivity, bootstrap intervals; record the known-in-advance decision; check quantile crossing; freeze champion(s), backtest forecasts and the 14-week forecast with quantiles. Gate: champion or contender set chosen with uncertainty shown; test untouched.
- **Phase 5. Scheduling core.** Contract and hand tests from section 7, netting engine, independent checker, simulator, frozen assumptions and cost proxies. Gate: all toy tests pass.
- **Phase 6. Schedulers and firm schedule.** MRP and baseline (plus optional variants), Track B weights with sensitivity, forecast-by-scheduler check, 14-week simulated firm schedule. Gate: zero hard-constraint violations; checker passes the winner.
- **Phase 7. Final test (once).** Retrain the champion, evaluate on the sealed test, run the winning scheduler on the same demand, compare with the traditional baseline, report all 10 KPIs and where the system is worse. Gate: test consumed once and logged in `STATE.md`.
- **Phase 8. Django API and Ask AI.** Models, auth, roles, endpoints, runs table, proposal versioning, confirm/cancel/revert, audit, assistant evaluation.
- **Phase 9. UI, monitoring and report.** React pages; bias and tracking-signal alerts; offline retrain with gated promotion; three robustness tests on the champions (demand shock, supplier delay, infeasible capacity); final report with methodology, selection matrices, sensitivity, KPI table, limits and a real-data adoption checklist (data mapping, cost inputs, supplier-reliability calibration, planner review, shadow-mode pilot); one-page summary with the synthetic-data caveat first.

**Release-blocking tests (the app is not releasable until all pass):**
1. A viewer cannot read or change protected data by calling the API directly.
2. A user cannot confirm a proposal they are not authorized to approve.
3. A stale or duplicate confirmation cannot apply a change twice; concurrent edits are handled.
4. An infeasible schedule cannot be marked `firm` or applied.
5. Prompt injection cannot bypass authorization, invent a quantity or trigger a write without confirmation.
6. No secrets in source control; production settings expose no debug information.
7. A backup restores and passes integrity checks.
Also: data and leakage tests (missing weeks not zero, no future rows in fitting, no forbidden columns in features, features at an origin unchanged when later rows are appended), selection tests (weights sum to 100, asymmetric-bias function, reproducible sensitivity), UI tests (no hard-coded metrics, proxy labels shown), and `pytest` green.

## 12. Deliverables
Working app (React, Django, database); `ml/` package; data dictionary, quality report, leakage audit; `results/` files (`track_a_candidates.csv`, `track_a_selection_matrix.csv`, `forecast_future_14w.csv`, `track_b_candidates.csv`, `track_b_selection_matrix.csv`, `firm_schedule_14w.csv`, `kpi_before_after.csv`, `final_test_report.md`, `performance_log.csv`); assistant test set and safety results; README (setup, migrations, import, run, test, demo, backup/restore); `SPEC.md`, `STATE.md`, final report, executive summary.

## 13. Rules for the coding agent
1. Read `SPEC.md` and `STATE.md` at session start; continue from the first unfinished phase; one phase at a time; run `pytest` at every gate.
2. At each checkpoint report work done, files changed, commands run, real test results, assumptions, open issues, next command; update `STATE.md`.
3. Never invent numbers or completed features. Every metric comes from executed code or is labelled an assumption, proxy or limitation.
4. Same splits, metric code, tuning budget, seed 42, simulator and demand path for every candidate. Tune and select on train/validation only; open the final test once.
5. Forecasting predicts; the scheduling engine computes quantities; the assistant only interprets language and calls approved functions.
6. Keep it simple and testable: no unrequested features, extra services or speculative refactors. If data cannot support a requirement, use a labelled proxy or mark it not measurable.

## Appendix A. Factor traceability (fill in during Phase 4, verify against the real columns)
For each PDF factor record: source column(s) in D1/D2/D3, known at forecast time (yes/no), missing-data handling, and status: **direct** (measured), **approximated** (proxy) or **unavailable**. Expected status below is a starting point to verify, not a fact.
| # | PDF factor | Expected source | Known at forecast time | Missing handling | Expected status |
|---|---|---|---|---|---|
| 1 | Historical consumption and trends | D1 demand and date | past only | missing weeks excluded | direct |
| 2 | Part-wise patterns | D1 part id, D3 category | yes | n/a | direct |
| 3 | Seasonal variations | D1 calendar fields | yes | derive from date | direct |
| 4 | Festival periods, holidays | D1 calendar/festival fields | yes (published calendar) | zero flag if blank | direct, generator-injected |
| 5 | Vehicle/service seasonality | D1 or D3 vehicle type/model | yes | unknown category | approximated |
| 6 | Monthly and weekly trends | derived from D1 | past only | n/a | direct |
| 7 | Market and regional patterns | D1 region, market fields | region yes; market only if published | blank = no event | approximated |
| 8 | Previous-year period | D1 lag 52 | past only | seasonal profile fallback | direct |
| 9 | Lead time | D2 | yes | part excluded and flagged | direct |
| 10 | Existing inventory | D2 snapshot | yes (single snapshot) | flag | direct, snapshot only |
| 11 | Open orders, pending requirements | D2 open PO fields; backorder as assumption | yes | zero if blank | approximated |
| 12 | Supplier/production capacity | D2 supplier slots | yes | slot unavailable | direct |
| 13 | Criticality, service level | D2/D3 | yes | default by criticality, flagged | direct |
| 14 | Spikes, unusual consumption | D1 capped vs raw demand; outcome flags for evaluation only | past only | n/a | approximated |
| 15 | Promotions, campaigns, market events | D1 promotion/campaign fields | only if published at origin | blank = none | approximated |
| 16 | Working days, shutdowns, plant holidays | D1 working days, shutdown flag | yes (plant calendar) | derive from calendar | direct |

## Appendix B. Limitations
Data is synthetic; D2 is a single snapshot, not a ledger; promotions, events and calendar effects may be generator artifacts; weights and cost proxies are assumptions; four validation origins give limited statistical power; real supplier reliability and emergency costs need calibration before any real use.
