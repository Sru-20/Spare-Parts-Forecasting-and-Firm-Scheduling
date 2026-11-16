# 02. Phase 9 UI close-out and evidence

**Status: PARTIAL.** The website and its safe demonstration path are implemented and verified, but the Phase 9 specification gate is not closed. This is method validation on synthetic data, not evidence of factory savings or operational performance.

## Delivered

- React/TypeScript application shell and shared design system with responsive navigation, run context, search, theme switching, notices, and reusable page components.
- API-backed overview, forecasts, firm schedule, inventory and supply, KPI analytics, Ask AI, audit, run history, and settings pages. Pages display saved results only and preserve firm versus unresolved status.
- KPI monitoring alerts sourced only from saved `forecast_bias` and `tracking_signal` records. The ±10% bias alert limit follows `configs/project.yaml`; the ±4 tracking-signal display limit is a provisional convention, not a frozen acceptance criterion. Missing tracking-signal data is shown as not measurable, never as healthy.
- Focused synthetic robustness tests for demand shock, supplier delay, and infeasible capacity. They exercise the existing deterministic scheduling engine; they do **not** claim champion-level scenario results.

## Verification

| Check | Result |
|---|---|
| `Push-Location .\frontend; npm run test` | 6 tests passed |
| `Push-Location .\frontend; npm run build` | TypeScript check and Vite production build passed; 0.99 seconds, 299.08 kB main JS (95.05 kB gzip), 365.32 kB chart chunk (105.39 kB gzip) |
| `py -3.10 -m pytest -q tests/test_phase9_robustness.py` | 3 synthetic tests passed |
| Authenticated browser route smoke check | Overview, forecasts, schedule, inventory, KPIs, Ask AI, audit, runs, and settings loaded without visible error alerts |
| Responsive browser check | At 390 px viewport, body width and scroll width were both 375 px; schedule overflow remained within its table scroller |
| Ask AI read-only demo | Returned saved schedule-row evidence and “No automatic write”; one conversation/message was stored in the isolated local demo database, and no proposal was created |
| Full repository `pytest` | Not run. Existing tests call `load_all`/`load_demand` on the original demand CSV (for example `tests/test_data_phase1.py`, `tests/test_backtest_phase3.py`, and `tests/test_features_phase2.py`), which would read sealed target-bearing rows. |

The root directory has no `npm run build` script; frontend commands must be run from `frontend/`.
The machine-readable measurements and browser smoke results are recorded in [03_phase9_ui_validation_v1.json](../results/03_phase9_ui_validation_v1.json).

## Saved Phase 6 comparison displayed by the UI

These are saved train/validation comparison values for `lightgbm_quantile::random_1` + `risk_adjusted_mrp`, `origin_visible`, from `results/phase6_kpi_comparison_v1.csv`. The comparator is the traditional historical-average forecast plus D2 reorder-point rule. These are not final-test results.

| KPI | Candidate | Traditional baseline | Basis |
|---|---:|---:|---|
| Forecast WAPE | 46.35% | 57.46% | Measured |
| Forecast accuracy | 53.65% | 42.54% | Measured; WAPE-derived |
| Forecast bias | -34.60% | -1.09% | Measured; material under-forecast alert |
| Fill rate | 83.89% | 84.77% | Simulated |
| Stock-out rate | 10.48% | 9.59% | Simulated |
| Critical-part availability | 94.56% | 91.76% | Simulated |
| Emergency order events | 144 | 218 | Simulated |
| Carrying-cost proxy | 25,601,949.91 | 9,297,403.88 | Simulated proxy; candidate is worse |

The underlying Phase 6 report contains the selection score, sensitivity results, checker evidence, and additional limitations: [phase6_comparison_report.md](./phase6_comparison_report.md). Forecast selection evidence remains in [model_comparison_report.md](../results/model_comparison_report.md) and its referenced result files. D2 inventory inputs postdate validation; cost and inventory comparisons are scenario estimates, not historical operational evidence.

## Robustness test scope

All three tests use synthetic 14-week inputs in `tests/test_phase9_robustness.py` and do not load project datasets:

- Demand shock: doubles point and P99 fixture demand; order quantity increases and the hard-constraint checker passes.
- Supplier delay: fixture OTD changes from 100% to 0% and average delay from 0 to 14 days; risk-adjusted lead time changes from 7 to 21 days and the checker passes.
- Infeasible capacity: fixture capacity is 10 units/month against 20 units/week; the plan remains unresolved and the checker rejects it.

These checks validate engine behavior under controlled fixtures only. A robustness evaluation using the selected champion's saved forecast paths and measured scenario KPIs remains open.

## Remaining Phase 9 gate items

1. Offline retraining with gated promotion is not implemented; Settings explicitly exposes no disconnected retraining or promotion controls.
2. The tracking-signal KPI is absent from the registered Phase 6 run, so its alert is correctly not measurable.
3. Champion-path demand-shock, supplier-disruption, and infeasible-capacity evaluations with saved-result evidence remain open; current tests are synthetic mechanism checks.
4. The full repository test gate is unrun to preserve the sealed final-test target boundary.
5. Phase 5 remains PARTIAL and Phase 7 remains unevaluated and unfinalized. No final-test target values were accessed during this Phase 9 work.

## Real-data adoption checklist

- [ ] Map real part, region/plant, unit-of-measure, lifecycle, and supersession identifiers; reconcile catalogue ownership.
- [ ] Validate dated inventory, reservations, blocked stock, in-transit quantities, open-PO buckets, and missing-week semantics.
- [ ] Calibrate supplier lead time, OTD, delay, rejection, capacity, MOQ/order multiples, and share-of-business from auditable records.
- [ ] Replace proxy costs with finance-approved stock-out, emergency, carrying, and excess/obsolete inventory costs.
- [ ] Obtain planner review of forecasts, unresolved orders, exceptions, critical-part thresholds, and alert cutoffs.
- [ ] Run authorized historical backtests and robustness scenarios, then operate in shadow mode with no ERP writes.
- [ ] Require independent model/schedule review, backup/restore verification, security approval, and explicit promotion approval before operational use.

## Next executable command

For a local demonstration, from the repository root run `Push-Location .\frontend; npm run dev`. The next development work remains within Phase 9; do not open the final-test window without the separate authorization required by the evaluation protocol.
