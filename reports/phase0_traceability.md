# Phase 0 — Requirements Traceability and Execution Plan

Source of truth: [`docs/Spec.pdf`](../docs/Spec.pdf) (final execution guide); updated requirements: [`docs/Spare_Parts_Forecasting_Problem_Statement_Updated.pdf`](../docs/Spare_Parts_Forecasting_Problem_Statement_Updated.pdf). The original problem-statement PDF referenced in the README is not present. The updated document retains the original factors and KPIs.

“Directly supported” means an observed field or explicit requirement exists, not that a model or operational benefit has been validated. “Derived feature” means computation from observed fields is required. “Simulation needed” means an outcome depends on the Phase 5 simulator. No real-plant savings are measurable from this synthetic data.

The traceability status vocabulary is **Directly supported**, **Derived feature**, **Simulation needed**, and **Not measurable**. Status describes the evidence/method needed; explanatory notes explicitly flag causal/real-plant claims that remain not measurable.

## Sixteen influencing factors

| Requirement | PDF section | Track | Dataset column/module | Method | Verification | Status |
|---|---|---|---|---|---|---|
| Historical consumption and demand trends | Updated §Key Factors; Spec §1.9 | A | D1 `demand_qty` | Time-aware lags and rolling summaries; raw demand is label | Phase 2 leakage tests; Phase 3 rolling-origin evaluation | Directly supported |
| Part-wise demand patterns | Updated §Key Factors | A | D1 `part_id`, `region`, `demand_qty` | Model each part-region series and segment from training history | Phase 1 series counts; Phase 2 segment report | Directly supported |
| Seasonal variations | Updated §Key Factors | A | D1 `month`, `week_of_year`, calendar fields | Derive seasonal features from dates and training observations | Phase 2 origin-truncation test | Derived feature |
| Festival periods and holidays | Updated §Key Factors | A | D1 `festival_*`, `public_holiday_flag`, festival proximity | Use known calendar events; assess by event window | Phase 2 event audit and backtest slices | Directly supported |
| Vehicle/service seasonality | Updated §Key Factors | A/B | D1 `vehicle_system`, `vehicle_type`, `vehicle_parc`, `avg_vehicle_age_years`; D1/D2 `criticality`, `service_level_target` | Use available fleet and service-priority proxies; do not claim actual service-job demand | Phase 2 feature audit; Phase 5 service simulation | Derived feature; causal effect not measurable |
| Monthly and weekly demand trends | Updated §Key Factors | A | D1 `week_start_date`, `month`, `week_of_year`, `demand_qty` | Derive weekly/monthly history summaries at each origin | Phase 2 leakage test; Phase 3 time backtest | Derived feature |
| Market and regional demand patterns | Updated §Key Factors | A | D1 `region`, `market_condition_index`, `market_event_flag` | Include only values known at forecast origin; unknown future event flags stay unavailable | Phase 1 availability audit; Phase 3 regional evaluation | Directly supported |
| Previous-year corresponding-period demand | Updated §Key Factors | A | D1 `demand_lag52` and dates | Recompute same-period lag from origin-visible history; do not trust precomputed values until audited | Phase 2 truncation test | Derived feature |
| Procurement/production lead time | Updated §Key Factors; Spec §1.11 | B | D2 `sup*_lead_time_days` | Supplier-specific receipt dates and lead-time feasibility | Phase 5 constraint checker; Phase 6 schedule validation | Directly supported |
| Existing inventory and stock levels | Updated §Key Factors; Spec §1.11 | B | D2 on-hand, reserved, blocked, available, safety stock, reorder point | Net usable stock once; reconcile quantities before simulation | Phase 1 schema tests; Phase 5 conservation tests | Directly supported |
| Open orders and pending requirements | Updated §Key Factors; Spec §1.11 | B | D2 in-transit, open PO split, ETA, `backorder_qty` | Avoid duplicate receipts; derive pending requirements explicitly from available backlog/demand; there is no pending-requirement column | Phase 1 field audit; Phase 5 known-answer tests | Derived feature |
| Supplier/production capacity | Updated §Key Factors; Spec §1.11 | B | D2 `sup*_monthly_capacity`, `sup*_capacity_committed_pct` | Allocate only feasible residual capacity; document supplier aggregation | Phase 5 independent constraint checker | Directly supported |
| Part criticality and service level | Updated §Key Factors; Spec §1.11 | A/B | D1/D2/D3 `criticality`, `service_level_target`, `planning_priority` | Score service against per-part target and criticality segment | Phase 1 consistency validator; Phase 6 KPI table | Directly supported |
| Demand spikes and unusual consumption | Updated §Key Factors; Spec §§1.4, 1.11 | A | D1 `demand_qty`; `out_spike_flag`, `out_unusual_demand_flag` only as labels | Detect/evaluate using origin history; barred `out_*` flags may not be features | Phase 1 leakage tests; Phase 3 spike slices | Directly supported |
| Promotions, campaigns and market events | Updated §Key Factors; Spec §1.11 | A | D1 `promotion_*`, `campaign_*`, `market_event_flag` | Include only known-in-advance events; unknown future market events remain missing | Phase 1 availability audit; Phase 3 event ablation | Directly supported |
| Working days, shutdowns and plant holidays | Updated §Key Factors; Spec §1.11 | A/B | D1 `plant_working_days`, `plant_shutdown_flag`, shutdown proximity; D2 supplier schedule data | Forecast with known calendar; decide plant-calendar handling in frozen simulator | Phase 2 feature tests; Phase 5 simulator tests | Directly supported |

## Ten original KPIs

| KPI | PDF section | Track | Dataset column/module | Method | Verification | Status |
|---|---|---|---|---|---|---|
| Forecast Accuracy % | Updated §Key Performance Indicators; Spec §1.6 | A | D1 `demand_qty`; `src/evaluation/` | `100 − WAPE`, report both weekly and lead-time levels and limitations | Phase 3 metric unit tests; Phase 7 one-time test | Derived feature |
| Forecast vs. Actual Variance | Updated §Key Performance Indicators; Spec §1.6 | A | D1 actuals and origin forecasts | Population variance of paired forecast errors; report means | Phase 3 metric unit tests | Derived feature |
| Schedule Adherence % | Updated §Key Performance Indicators; Spec §1.6 | B | `src/simulation/`, due/release/receipt schedule | On-time completed quantity / scheduled quantity | Phase 5 simulator tests; Phase 6 paired runs | Simulation needed |
| Stock-Out % | Updated §Key Performance Indicators; Spec §1.6 | B | D1 demand; `src/simulation/` | Unmet units / demanded units under frozen inventory policy | Phase 5 toy tests; Phase 6 paired demand paths | Simulation needed |
| Inventory Turnover | Updated §Key Performance Indicators; Spec §1.6 | B | D2 `unit_cost`, simulated inventory and served demand | Annualized cost of demand served / average inventory value | Phase 5 formula tests; Phase 6 KPI output | Simulation needed |
| Excess Inventory Reduction | Updated §Key Performance Indicators; Spec §1.6 | B | Simulated stock and configured target | Report excess quantity/value and paired change vs. current-policy baseline | Phase 5 proxy/target definition; Phase 6 baseline comparison | Simulation needed |
| Emergency Procurement Reduction | Updated §Key Performance Indicators; Spec §1.6 | B | `src/simulation/`; no actual emergency cost column | Simulate emergency orders; provisional cost = emergency qty × D2 unit cost × 2.0; compare paired count/cost | Phase 5 freeze and proxy sensitivity; Phase 6 candidate matrix | Simulation needed; real savings not measurable |
| Service Level / Fill Rate | Updated §Key Performance Indicators; Spec §1.6 | B | D1 demand; D2 criticality/target; simulator | Filled / demanded units by part and criticality vs. target | Phase 5 conservation tests; Phase 6 segment report | Simulation needed |
| Forecast Bias | Updated §Key Performance Indicators; Spec §1.6 | A | D1 actuals and forecasts | Signed forecast error / total actual; report sign convention and zero-total case | Phase 3 metric unit tests | Derived feature |
| Inventory Carrying Cost Reduction | Updated §Key Performance Indicators; Spec §§1.6–1.7 | B | D2 `carrying_cost_rate`, `unit_cost`; simulator | Inventory value × annual rate × simulated days / 365; if absent, provisional 25% annual proxy; compare to baseline | Phase 5 proxy and calendar freeze; Phase 6 paired run | Simulation needed; real savings not measurable |

Proxy-cost comparisons are comparative synthetic simulations, not observed financial savings. D2 does not provide stock-out or emergency-procurement cost; configure those assumptions only when Phase 5 freezes the simulator.

The Phase 0 provisional proxy is a stock-out penalty per unmet unit of `unit_cost × {High: 2.0, Medium: 1.0, Low: 0.5}` and emergency-procurement total cost of `quantity × unit_cost × 2.0`. These are explicit design assumptions, not estimates from the synthetic data. Prefer D2's part-specific carrying rate; use 25% annual only as a labelled fallback. Phase 5 must review and freeze or replace these proxies before any simulator comparison.

## Updated assistant, controls and audit requirements

| Requirement | PDF section | Track | Dataset column/module | Method | Verification | Status |
|---|---|---|---|---|---|---|
| English/Hindi natural-language interaction; clarify ambiguity | Updated §1 Multilingual Support; Spec §1.8 | C | `src/assistant/`; configurable language list | Intent/entities in supported languages; ask clarification when confidence/required fields are insufficient | Phase 8 labelled bilingual and ambiguity cases | Directly supported |
| Intent, entity and period/action/quantity identification | Updated §2 workflow; Spec §1.6 | C | `src/assistant/` typed request schema | Structured interpretation; no free-form quantities from the LLM | Phase 8 intent/entity exact-match tests | Derived feature |
| Validate role, permissions and business rules | Updated §§2, 4; Spec §1.10 | C | `backend/` Django/DRF permissions; imported `src/scheduling/`; D2 supplier/inventory constraints | Django authorization plus deterministic scheduling validation in-process | Phase 8 unauthorized/invalid request tests | Directly supported |
| Retrieve schedules and prepare a before/after proposal with impact/warnings | Updated §2 workflow | C | Persisted schedules; approved tools | Read stored schedule, invoke deterministic proposal/validation functions | Phase 8 tool-contract tests | Directly supported |
| Explicit confirmation/cancellation; no auto-apply | Updated §§2, 4; Spec §1.10 | C | Proposal version, role, apply/revert API | Approver-bound versioned confirmation; cancellation is non-mutating | Phase 8 duplicate/stale/bypass HTTP tests | Directly supported |
| Apply approved change and recalculate schedule/KPIs | Updated §2 workflow | B/C | Scheduling engine and persisted result | Only validated approved proposal reaches apply; quantities are computed by scheduler | Phase 8 integration tests | Simulation needed |
| Audit request, interpretation, proposal, approval, final change and timestamp | Updated §4 Controls and Auditability | C | Backend append-only audit log | Record complete state transitions and actor/time | Phase 8 audit completeness tests | Directly supported |
| LLM must not invent quantities; MOQ/SOB/lead-time/capacity/inventory/service constraints | Updated §4; Spec §§1.4, 1.6 | B/C | Django API; imported `src/scheduling/` independent checker | LLM calls typed Django application functions; scheduler is sole quantity authority | Phase 5 checker and Phase 8 adversarial tests; zero violations | Directly supported |
| Explain violations and clarify ambiguous requests | Updated §§2, 4 | C | Assistant response and validation errors | Return explicit reason and clarification prompt; no silent fallback | Phase 8 invalid/ambiguous suite | Derived feature |
| Traceability and multilingual request handling | Updated §5 Additional Success Criteria | C | Assistant evaluation set and audit store | Exact intent/entities, cross-language consistency and required audit fields | Phase 8 scorecard | Derived feature |

## Phase 0 discrepancies and confirmed facts

- The README references `docs/Spare_Parts_Forecasting_Problem_Statement.pdf`, but the original PDF is absent. `docs/Spec.pdf` is the final execution guide, not the original problem statement. The updated problem statement is present and includes the original factors/KPIs plus additional assistant requirements. No source PDF was overwritten or copied.
- The actual CSVs match their README row/column counts. The actual 616 series have 119 Smooth, 97 Erratic, 133 Intermittent, 220 Lumpy, and 47 short/unclassified, matching the guide.
- D1 has 48.55% genuine observed zero-demand history among non-missing historical rows (54,511 of 112,277). Blank missing/future labels are not zeros.
- D2 has 158 rows for 158 non-obsolete parts. Obsolete P0046 and P0087 have history but no D2/future rows, as designed.
- The validator's current run is 32/32 PASS, not merely the guide's claimed first inspection.
- The Track B listed default weights total 95%; retain the exact listed components, score against that fixed total, and do not silently invent/redistribute a missing 5%.
- D2 has no pending-requirement column and no stock-out/emergency-procurement cost fields. Pending demand and proxy costs remain explicit Phase 5 design decisions.

## Generator findings and synthetic-data artifacts

`data/generate_datasets.py` uses the Python standard library and `random.Random(42)`. Its deterministic synthetic construction includes 160 catalogue parts, four region scales, weekly calendar/festival drivers, component seasonality, lifecycle curves, demand-class-specific occurrence and size distributions, and up to ten attempts retuning occurrence/size variability to match the intended class on train/validation.

- The generator injects 22 sampled historical promotion weeks plus planned Diwali promotion weeks in the future, and explicitly defines six campaign windows including the future 2026-10-05–2026-10-26 campaign. Promotion-sensitive series receive a post-promotion 0.72 multiplier.
- It injects substitution groups and a 2025-04-07 demand shift (cannibalized ×0.55; substitute ×1.65), as well as lifecycle, project orders, batching, panic/bullwhip, dead-stock, shutdown, and step-change effects. These are known construction mechanisms, not independently discovered causal relationships.
- Demand occurrence, size, market index, outcomes, missing-row locations, supplier attributes, and D2 inventory are generated with the fixed random seed. The historical `out_*` labels come from a synthetic stock process initialized from average demand; they do not establish observed plant outcomes.
- `demand_qty_capped` and `out_spike_flag` are built from a cap selected using nonfuture warm-up/train-validation data, not a cap dynamically fitted per rolling origin. Re-estimate any cap within training folds before using it as a learned transformation.
- `demand_class_trainval`, `adi_trainval`, `cv2_trainval`, `abc_class`, and `part_type` are generated/derived with train-validation information. Their availability must be evaluated per forecast origin; they are not safe default features for earlier origins.

These generator-specific artifacts motivate the Phase 1 feature-availability audit and Phase 2 truncation-based leakage tests.

## Execution plan

| Phase | Scope | Exit evidence / gate |
|---|---|---|
| 0 | Setup, source reconciliation, environment, traceability, specification/config freeze | Validator and pytest results, reports, split confirmation, user-visible overview |
| 1 | Typed data loading, validation, joins, feature availability/leakage audit | Data tests, leakage audit, target decision |
| 2 | EDA, segmentation and origin-safe feature engineering | Feature/segment reports and leakage tests |
| 3 | Track A Tier 1 baselines and forecasting candidates | Rolling-origin candidate results; no final-test use |
| 4 | Track A ensembles/router/calibration/champion | Sensitivity/stats and frozen champion; test sealed |
| 5 | Formal schedule constraints and frozen simulator | Deterministic simulator tests and independent feasibility checker |
| 6 | Track B scheduler and forecast-scheduler matrix | Zero-violation schedule, KPI/sensitivity/bootstrap evidence |
| 7 | End-to-end comparison and one-time final test | Final-test report and all ten original KPIs or explicit not-measurable flags |
| 8 | Track C controlled multilingual Ask AI | Confirmation/audit/security tests and separate candidate results |
| 9 | Tier 1 robustness and final report | Pytest green, reproducible file-backed results, final state |

Every phase is gated, and later phases remain blocked until their predecessors pass. Tier 2 requires a measured Tier 1 gap; Tier 3 is deferred unless requested.
