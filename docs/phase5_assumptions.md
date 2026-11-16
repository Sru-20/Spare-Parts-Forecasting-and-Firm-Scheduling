# Phase 5 simulator assumptions and evidence

**Status:** PARTIAL — the approved no-eligible-donor exception is documented and implemented; post-change Phase 5 gate verification remains pending.
**Version:** `phase5-assumptions-v3`
**Claim scope:** method validated on simulated data; no factory savings claim.
**Configuration source:** `configs/project.yaml` and `configs/business.yaml`.
**Frozen config SHA-256:**
- `configs/project.yaml`: `46A6183450ABAD85C84EA3DF41E412709E189B9314D42C8D1FB0C34A533314CC`
- `configs/business.yaml`: `ED0E3DF890504ED02F2AEF27EE3BA6CA4A431EF6E40D07951DC4D86B231E6540`

Version v3 replaces the historical D2-based primary start with the v4.2
origin-visible trailing-history rule. The D2 snapshot is retained only as the
separately labelled `d2_snapshot_stress` case.

## Current v4.2 closure evidence (2026-11-15; gate verification pending)

- Implemented the primary opening stock as
  `ceil(origin-visible trailing 13-week mean) * (ceil(SOB-weighted lead days / 7) + 1)`.
  The primary path requires at least eight complete observed weeks; primary
  open POs and backlog are zero. History is split-filtered and strictly prior
  to each of the four saved held-validation origins. D2 inventory, receipts,
  and backlog are used only in the separate `d2_snapshot_stress` scenario.
- `python -m src.scheduling.phase5 --demo` completed successfully. The
  authoritative outputs use the `phase5_primary_v5_*` and
  `phase5_d2_snapshot_stress_v5_*` prefixes. Repeated primary outputs were
  byte-identical; see `../results/phase5_repeat_run_evidence_v5.json`.
- Population: 640 possible forecast part-origins; 570 complete paths, 42
  without forecast rows, 22 with missing observed regional labels, and 6
  with an incomplete forecast horizon. Seven otherwise complete paths were
  excluded from scheduling because the parts lacked a D2 supplier snapshot
  (3 High, 4 Medium; all seven were origin-visible Lumpy). No path was
  excluded for insufficient opening-history length.
- Median initial inventory cover (weeks), calculated against origin-visible
  mean demand:

  | Criticality | Origin-visible primary | D2 stress |
  |---|---:|---:|
  | High | 4.071 | 0.481 |
  | Medium | 4.078 | 0.505 |
  | Low | 4.045 | 0.514 |

- Primary results: **467 firm**, **93 unresolved**, **3 not ordered** across
  563 schedule checks. There were **0 hard-constraint failures on firm
  schedules** and **0 MRP or simulator inventory-ledger failures**. Eight
  unresolved schedules had a first-week lead-time infeasibility; 85 had
  supplier-capacity/MOQ-rounding and unfilled-requirement failures. Their
  remaining unfilled requirements totalled 1,636,196 simulated units. The D2
  stress run produced 256 firm, 305 unresolved, and 2 not ordered.
- Primary simulated KPIs: fill rate **87.48%**, stockout part-weeks **6.79%**,
  109 emergency orders, emergency-cost proxy **18,100,932.63**, and carrying-
  cost proxy **25,342,132.10**. These are replay metrics, not savings claims.
- The six-row sensitivity report includes baseline, opening stock x0.5/x1.5,
  due-later receipts at weeks 3/6, and per-unit-binomial rejection; configured
  defaults were not changed. Unresolved schedules were 124 at x0.5, 69 at
  x1.5, and 93 for each of the other variants.
- The user approved a narrow no-donor data-limitation exception on 2026-11-16.
  The eligibility rule remains exact `component` + exact `vehicle_type`;
  inferred vehicle-family/category matches are not accepted as compatible
  donors. All 160 catalogue pairs are unique, so all 47 unclassified
  part-region series have zero eligible donors at each of the four held
  validation origins: 2025-01-13, 2025-04-21, 2025-07-28, and 2025-11-03.
  The affected part IDs are P0007, P0009, P0024, P0025, P0042, P0062, P0068,
  P0090, P0091, P0092, P0106, and P0122. Each series-origin is recorded as
  `UNFORECASTED_NO_DEFENSIBLE_ANALOGUE`; no forecast row or quantity is emitted
  and unavailable forecasts are never replaced with zero demand. The 43
  short-history series have no eligible exact-rule donor. P0091's four
  regional series have no development observations and no validated
  launch/exposure scaling, so they also remain unforecasted even if a metadata
  peer were proposed.
- The new versioned evidence is
  `../results/phase5_analogue_exceptions_v1.csv` (188 series-origin records),
  `../results/phase5_analogue_exceptions_v1.json`, and
  `../results/phase5_analogue_exception_summary_v1.md`. The existing empty
  `../results/analogue_fallback_forecasts.csv`, diagnostic, and v5 planner
  report are preserved as historical evidence, not rewritten as forecasts.
  The frozen High-criticality bias note remains **-9.13% on the
  point estimate only**; its earlier bootstrap interval crossed -10%, so the
  result is inconclusive, not a statistically established pass.
- `py -3.10 -m pytest -q tests/test_phase5_scheduling.py` and the matching
  Python 3.13 command each passed **49 tests**. Python compilation,
  `git diff --check`, and Pylance diagnostics for `phase5.py` passed. The full
  repository suite and dataset validator were not run in this session. The
  validator was deliberately not rerun because a previous execution read
  sealed-test target values for integrity checks (not prediction evaluation).
- **Gate verdict: PARTIAL pending verification.** The approved data-limitation
  exception replaces the requirement to invent or broaden fallback forecasts;
  it does not waive the remaining Phase 5 gate. Run the required tests on
  Python 3.10 and 3.13 after review. The full-suite result is unavailable.
  Do not rerun the dataset validator because the established validator reads
  sealed-test target values; use only previously recorded 32/32 evidence or
  obtain approval for a target-isolated alternative. Existing hard-constraint,
  inventory-reconciliation, repeat-run, and sensitivity evidence remains
  recorded above.

## Historical D2-snapshot results (assumptions-v2; superseded)

The following inventory initialization and result tables document the earlier
v2 run only. They are retained for provenance and are not evidence that the v4.2
primary-initialization gate passes.

## Inputs, scope, and initialization

- D2 (`data/02_scheduling_inputs.csv`) is a single inventory/supplier snapshot
  dated 2026-09-26, not a transaction ledger. For each held-validation origin,
  the same `available_qty` and `backorder_qty` are reused as a synthetic
  initialization, identically across policies. There is no historical
  warm-up. Inventory starts from D2's reconciled `available_qty`; the initial
  stock multiplier is 1.0 by default.
- Require exact `available_qty = on_hand_qty - reserved_qty - blocked_qty`.
  Missing or inconsistent required values are errors, not silently imputed.
- Demand paths use the four saved validation origins and complete observed
  actuals aggregated over regions. The planning point forecast is the saved
  `lightgbm_tweedie::random_1`; safety-stock quantiles come from the saved
  `lightgbm_quantile::random_1`. No forecast model was retrained, and no
  forecast result was changed.
- D2 is one row per part. The schedule's region/plant label is `ALL`;
  regional allocation is not modelled.
- D2 `backorder_qty` is the labelled pending-requirement assumption: it is
  initialized as backlog for High-criticality parts only. Medium/Low initial
  backorders are written off; subsequent Medium/Low unmet demand is lost sales.
- D2 has no individual PO identifiers or receipt dates. `in_transit_qty` and
  `open_po_due_this_month_qty` map to week 1; `open_po_due_later_qty` maps to
  week 4 by default. Due buckets must reconcile to `open_po_qty` and are
  counted once. Sensitivity weeks 3 and 6 are also run. The later D2 snapshot's
  `earliest_open_po_eta` is not projected into validation dates.
- The full factor traceability table and evidence are in
  [Phase 5 Appendix A](../reports/phase5_appendix_a.md). These D2/D3 fields
  postdate the historical validation origins, so their use is explicitly a
  synthetic replay, not historical knowledge at those origins.

## Dataset coverage and starting inventory

There are 640 possible part-origin paths (160 catalogue parts × four origins).
The saved validation forecasts/actuals yielded 570 complete paths:

| Path status/reason | Paths |
|---|---:|
| Complete 14-week forecast and observed actual path | 570 |
| No forecast rows for part-origin | 42 |
| At least one missing observed regional label | 22 |
| Incomplete 14-week forecast horizon | 6 |

All 48 `Unclassified` paths are excluded: their reason is no forecast rows
(42) or missing observed regional labels (6). By criticality, exclusions are
44/236 High (18.64%), 20/264 Medium (7.58%), and 6/140 Low (4.29%). By SBC
segment: Erratic 4/96, Intermittent 4/132, Lumpy 7/244, Smooth 7/120, and
Unclassified 48/48. Exclusions are therefore not representative, especially
for Unclassified parts and High-criticality paths; the metrics below describe
only retained complete paths.

Seven of the 570 complete paths are not schedulable because obsolete parts
P0046 (four origins) and P0087 (three origins) are absent from D2. The core
therefore schedules 563 paths.

Initial cover is `D2 available_qty / (14-week observed validation demand / 14)`.
Zero-demand paths have infinite cover and are counted separately; quantiles
below use only finite cover values.

| Criticality | Paths | Zero-demand | Median weeks | P10 weeks | P90 weeks | <1 week | >=14 weeks |
|---|---:|---:|---:|---:|---:|---:|---:|
| High | 189 | 4 | 0.519 | 0.027 | 2.732 | 69.31% | 4.76% |
| Medium | 240 | 1 | 0.614 | 0.035 | 3.505 | 63.33% | 1.67% |
| Low | 134 | 1 | 0.519 | 0.074 | 3.227 | 61.19% | 2.99% |

This is a synthetic initialization. It is not evidence that actual historical
inventory was at these levels.

## SOB, lead-time, capacity, and supplier uncertainty

- In D2, supplier SOB shares sum to exactly 100% for all 158/158 rows.
  `generate_datasets.py` assigns synthetic share patterns (including 100%,
  split shares, and 50/30/20) and uses them as supplier-load allocation
  weights. The source does not establish a binding contract limit. SOB is
  therefore an allocation target; realized deviations are reported but are
  not hard-constraint failures.
- Arrival timing is
  `release_week + max(1, ceil(lead_time_days / 7))`: 1, 7, 8, and 14 days map
  to weeks `r+1`, `r+1`, `r+2`, and `r+2`. Release weeks cannot precede week 1.
- For lead times beyond 14 weeks, planning demand extends the saved forecast
  using the average of its final four weeks for each extra week. The true
  arrival week is retained and flagged beyond horizon; no order is received
  inside the 14-week simulation when its arrival is later. The actual demand
  path is not extended for the scheduler simulation. Three D2 parts have
  SOB-weighted lead times over 14 weeks: P0027, P0096, and P0097.
- Capacity is `supN_monthly_capacity × (1 - supN_capacity_committed_pct/100)`
  in order-quantity units per calendar month. No verified plant working-day or
  shutdown calendar exists, so capacity is not adjusted by weekdays. A
  capacity shortfall remains unresolved; no replacement quantity is invented.
- The generator samples `supN_reject_pct` as a supplier percentage (domestic
  0.2–6.5%, import 0.3–3.5%) and derives reliability from OTD, rejection, and
  an additional adjustment. It contains no per-unit rejected-quantity or
  transaction evidence. The frozen default is therefore a seeded whole-lot
  rejection probability; a per-unit binomial variant is reported as a
  sensitivity, not asserted as the real supplier process. `supN_reliability_pct`
  is not applied as a second failure probability because that would overlap
  OTD and rejection.
- Supplier late/delay behavior uses `supN_otd_pct` and
  `supN_avg_delay_days`, with seed 42.

## Safety stock and measured forecast coverage

For each part, select the smallest available saved forecast quantile at or
above its D2 service target, capped at P99. No D2 target exceeds 0.99. Sum
regional point and quantile forecasts to part-level weekly values, then sum
those weekly values over the lead-time window. For demand beyond week 14, use
the final-four-week mean extension described above. Safety stock is
`max(0, selected lead-time quantile demand - lead-time point demand)`.
Summing weekly marginal quantiles is a conservative approximation; it is not
a calibrated joint lead-time distribution.

The lead-time coverage check is whether actual lead-time demand is no greater
than point lead-time demand plus the computed safety stock. It uses the same
lead-time extension when needed.

| Criticality | Part-origins | Coverage | Service target |
|---|---:|---:|---:|
| High | 189 | 99.47% | 99% |
| Medium | 240 | 100.00% | 95% |
| Low | 134 | 94.03% | 90% |

Coverage by criticality and each forecast horizon follows. Each `n` is the
number of part-origin observations at that horizon. A group with fewer than
30 observations would be marked **insufficient data**; no group meets that
condition (minimum `n` is 134).

| Horizon | n High | High P90 | High P95 | High P99 | n Medium | Medium P90 | Medium P95 | Medium P99 | n Low | Low P90 | Low P95 | Low P99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H01 | 189 | 96.30% | 97.88% | 99.47% | 240 | 95.83% | 99.17% | 100.00% | 134 | 91.79% | 97.01% | 99.25% |
| H02 | 189 | 93.65% | 98.41% | 100.00% | 240 | 94.58% | 98.33% | 100.00% | 134 | 91.04% | 99.25% | 100.00% |
| H03 | 189 | 90.48% | 96.30% | 99.47% | 240 | 94.17% | 98.75% | 100.00% | 134 | 93.28% | 97.01% | 100.00% |
| H04 | 189 | 92.06% | 95.77% | 98.94% | 240 | 96.67% | 97.50% | 99.58% | 134 | 93.28% | 97.76% | 100.00% |
| H05 | 189 | 91.01% | 95.77% | 99.47% | 240 | 87.50% | 94.58% | 99.17% | 134 | 84.33% | 89.55% | 98.51% |
| H06 | 189 | 85.19% | 91.53% | 98.41% | 240 | 89.58% | 95.00% | 97.50% | 134 | 90.30% | 92.54% | 97.76% |
| H07 | 189 | 93.12% | 97.35% | 99.47% | 240 | 91.67% | 95.83% | 99.17% | 134 | 94.78% | 98.51% | 100.00% |
| H08 | 189 | 93.12% | 95.77% | 99.47% | 240 | 93.33% | 97.92% | 99.17% | 134 | 91.04% | 95.52% | 98.51% |
| H09 | 189 | 95.24% | 97.35% | 98.94% | 240 | 91.67% | 99.17% | 100.00% | 134 | 92.54% | 94.03% | 100.00% |
| H10 | 189 | 90.48% | 93.12% | 98.41% | 240 | 93.33% | 96.25% | 98.75% | 134 | 92.54% | 97.01% | 100.00% |
| H11 | 189 | 90.48% | 94.71% | 98.41% | 240 | 92.50% | 95.42% | 100.00% | 134 | 91.04% | 94.78% | 99.25% |
| H12 | 189 | 91.01% | 95.24% | 99.47% | 240 | 93.33% | 96.67% | 99.58% | 134 | 92.54% | 94.78% | 99.25% |
| H13 | 189 | 90.48% | 94.71% | 99.47% | 240 | 88.33% | 93.75% | 98.75% | 134 | 87.31% | 94.78% | 98.51% |
| H14 | 189 | 89.95% | 95.24% | 99.47% | 240 | 90.42% | 95.42% | 99.58% | 134 | 84.33% | 91.79% | 99.25% |

No tuning-origin quantile forecasts are saved. The optional calibration
multiplier could not be fitted without those forecasts and remains **off**;
the frozen core above uses the uncalibrated default.

## Critical and A-class lead-time results

These metrics use the 563 D2-backed part-origin paths and the frozen champion
point forecast. ABC class is read from the spare-parts catalogue. Lead-time
WAPE is total absolute lead-time error divided by total actual lead-time
demand. Signed bias is `sum(forecast - actual) / sum(actual)`; negative means
under-forecast. Asymmetric absolute bias doubles the absolute negative bias,
as configured. The critical under-forecast rule passes when signed bias is no
worse than -10%.

| Group | Part-origins | Lead-time WAPE | Signed bias | Asymmetric absolute bias | -10% under-forecast rule |
|---|---:|---:|---:|---:|---|
| High criticality | 189 | 25.00% | -9.13% | 18.27% | Pass |
| ABC A | 118 | 20.99% | -3.93% | 7.86% | Pass |
| High criticality and ABC A | 52 | 26.01% | -9.44% | 18.89% | Pass |

## Unresolved plans and pull-ahead result

The one-pass rule attempts earlier releases using earlier calendar-month
capacity while preserving the normal lead-time calculation and never
releasing before week 1. It was run on the same validation plans before
marking them unresolved:

| Measure | Before | After |
|---|---:|---:|
| Unresolved part-origin schedules | 121 | 121 |
| Pull-ahead orders / quantity | — | 0 / 0 |

All 121 unresolved plans have `supplier_capacity` plus
`unfilled_requirement`; there are 57 High, 46 Medium, and 18 Low plans. The
weekly report contains 921 shortfall rows totalling 2,447,375 units. By
criticality, High has 382 rows / 1,753,356 units, Medium 346 / 616,564, and
Low 193 / 77,455.

The remaining shortfalls were grouped by requirement month, actual supplier
slots allocated in that week, and shortfall size:

| Month | Rows | Shortfall units |
|---|---:|---:|
| 2025-01 | 78 | 144,128 |
| 2025-02 | 85 | 221,625 |
| 2025-03 | 59 | 205,456 |
| 2025-04 | 45 | 137,595 |
| 2025-05 | 64 | 126,864 |
| 2025-06 | 46 | 93,409 |
| 2025-07 | 21 | 42,243 |
| 2025-08 | 113 | 287,847 |
| 2025-09 | 116 | 323,056 |
| 2025-10 | 60 | 242,885 |
| 2025-11 | 94 | 219,470 |
| 2025-12 | 82 | 226,356 |
| 2026-01 | 42 | 126,992 |
| 2026-02 | 16 | 49,449 |

| Shortfall band (units) | Rows | Shortfall units |
|---|---:|---:|
| <100 | 104 | 5,551 |
| 100–999 | 371 | 143,801 |
| 1,000–9,999 | 387 | 1,411,191 |
| >=10,000 | 59 | 886,832 |

Allocated supplier slots in unresolved shortfall rows: none 788 rows /
2,151,191 units; slot 1 75 / 165,709; slot 2 29 / 60,559; slot 3 9 / 27,591;
slots 1+2 8 / 6,652; slots 2+3 11 / 34,633; slots 1+2+3 1 / 1,040.
“None” means that week had no allocated order against the remaining need; it
does not mean the part has no supplier in D2.

The pull-ahead diagnosis found 119 week-1 shortfalls, which cannot be pulled
earlier, and no valid earlier-month order for the other needs. Eight
requirements had raw unused capacity at least equal to MOQ, but each was for
P0133/S006 with only 1.39 units left against a 5-unit order multiple; the
allocator correctly rounded usable capacity to zero. Thus no feasible
earlier-month order was available, and the remaining plans stay unresolved.
No quantity was fabricated.

## Weekly simulator and cost definitions

- Receipts arrive at the start of each week. High backlog is served before
  current demand; remaining High backlog carries forward. Medium/Low shortages
  are lost sales. Closing stock is never negative, and inventory is reconciled
  each week.
- Supplier rejection is applied to planned supplier orders. Emergency orders
  are outside the schedule, bypass MOQ/multiple/capacity and supplier
  uncertainty, and are excluded from the independent schedule checker. They
  are logged separately by count, quantity, and cost. An emergency order is
  triggered by remaining High backlog at week close and is received at the
  start of the next week. Stock-out is measured at week close **before** that
  emergency receipt.
- Emergency procurement cost is a provisional proxy: unit cost × 1.5
  (sensitivity range 1.25–2.0). Stock-out penalty is unit cost × 10/5/2 for
  High/Medium/Low (sensitivity ±50%). Carrying cost uses D2
  `carrying_cost_rate`; the 25% annual fallback is used only if the D2 value is
  absent. Downtime cost is not measurable.
- Supplier monthly capacity is preserved without weekday/calendar
  adjustment; demand is the saved validation weekly actual path. No
  plant-specific calendar was verified.

Executed default-core metrics (all simulated/proxy, not factory outcomes):

| Measure | Result |
|---|---:|
| Supplier order lines | 6,718 |
| MRP / simulation weekly rows | 7,882 / 7,882 |
| Feasible / unresolved schedules | 442 / 121 |
| Independent hard-constraint violations | 0 |
| MRP / simulator ledger failures | 0 / 0 |
| Service fill rate | 71.55% |
| Stockout part-week rate | 20.38% |
| Emergency events / quantity | 534 / 45,371 |
| Emergency cost proxy | 104,895,524.69 |
| Excess inventory proxy | 347,584.05 units |
| Carrying-cost proxy | 22,253,438.61 |

## Frozen-default sensitivity results

One factor changes per scenario; all use seed 42. Values are from executed
core runs and remain simulated/proxy measures.

| Scenario | Fill rate | Emergency count | Excess units | Carrying-cost proxy | Unresolved schedules |
|---|---:|---:|---:|---:|---:|
| Baseline (stock 1.0, due week 4, whole-lot) | 71.55% | 534 | 347,584.05 | 22,253,438.61 | 121 |
| Initial stock multiplier 0.5 | 69.02% | 575 | 366,533.78 | 22,123,149.96 | 127 |
| Due-later receipt week 3 | 71.63% | 532 | 342,530.57 | 22,152,397.39 | 120 |
| Due-later receipt week 6 | 71.38% | 540 | 348,981.06 | 22,240,451.50 | 123 |
| Per-unit-binomial rejection | 69.94% | 542 | 294,443.21 | 21,110,730.23 | 121 |

Sensitivity values are not used to replace or alter the frozen defaults.

## Hand tests and gate evidence

All ten SPEC section 7 hand cases passed:

1. **PASS** — Enough usable stock emits no replenishment order.
2. **PASS** — A shortage is covered by a receipt timed to the start of the required week.
3. **PASS** — An order whose lead time misses the required date remains marked late.
4. **PASS** — MOQ and order-multiple rounding is applied to actual supplier orders.
5. **PASS** — Exhausted capacity leaves an unresolved shortage without invented quantity.
6. **PASS** — Obsolete parts receive no new order.
7. **PASS** — Backorder and open-PO quantities are each counted once.
8. **PASS** — Lead times of 1/7/8/14 days map to `r+1/r+1/r+2/r+2`, plus an
   explicitly flagged beyond-horizon order.
9. **PASS** — A critical part carries unmet backlog and schedules emergency supply
   separately.
10. **PASS** — An ordinary part loses its unmet demand rather than carrying backlog.

Additional checks: **PASS** — inventory conservation and nonnegative closing
stock; **PASS** — repeat default-core run returned an identical summary;
**PASS** — seeded supplier realization is deterministic; **PASS** — the
independent checker rejects deliberately invalid schedules. The independent
checker found zero hard-constraint violations in the executed default run.

Tuning-origin quantile calibration remains skipped because those forecasts are
not saved. The final test remains sealed; only held-validation data was used.
The Phase 5 results are synthetic validation evidence and are not proof of
factory savings.
