# Phase 6 scheduler comparison

**Basis:** method validated on simulated data. The 14-week future schedule is provisional; no factory savings or supplier commitment is claimed.
**Phase 5 remains PARTIAL.** This Phase 6 work proceeds under the user-approved sequencing deviation; it does not satisfy the v4.2 sequential Phase 5 gate.

- Config SHA-256: `46a6183450abad85c84ea3df41e412709e189b9314d42c8d1fb0c34a533314cc`
- Winning validation pair (primary initialization): `lightgbm_quantile::random_1` + `risk_adjusted_mrp`; score 0.5125.
- Sensitivity winner changed in 3/12 one-factor ±10-point perturbations.
- Primary firm/unresolved paths: 229/3149.
- D2 stress firm/unresolved paths: 0/3378.
- Firm-order checker failures: 0; unresolved checker rows: 6573, reasons: `{"release_before_week_1": 5645, "release_before_week_1|unfilled_requirement": 871, "unfilled_requirement": 57}`; ledger failures: 0.
- Excluded candidate paths: 70; reasons: `{"incomplete_14_week_path|missing_historical_average_prediction": 12, "missing_D2_scheduling_snapshot": 14, "missing_actual": 44}`.
- Provisional future rows: 2781 (1961 firm, 820 unresolved); all emitted rows carry `PROVISIONAL` status.
- Aggregate independent checker for provisional firm orders: feasible=True; checked orders=806.
- Forecast/service/cost check (primary): lowest weekly WAPE was `lightgbm_tweedie::random_1` + `mrp` (45.02%); best fill rate was the same pair (87.48%), as was the lowest emergency-cost proxy (18,100,933). The lowest carrying-cost proxy was `lightgbm_quantile::random_1` + `mrp` (23,296,400). Thus, the lowest-error forecast also gave the best service and emergency-cost proxy under MRP, but not the lowest carrying cost; the weighted winner remains Quantile + risk-adjusted MRP.

## Limitations

- Baseline is a separate historical-average forecast plus D2 reorder-point rule. Its trigger uses projected lead-time inventory position; order quantity replenishes to the D2 reorder point, subject to supplier constraints.
- D2 is one snapshot postdating validation origins. Its use is a stress scenario except in the required historical baseline policy; primary MRP initialization is origin-visible with zero opening POs/backlog.
- Track B schedule stability between replans is not measurable from the saved one-run histories. The 15% adherence/stability scoring component therefore uses simulated on-time received quantity only; neither value is an operational measurement.
- Inventory turnover is a 14-week simulated COGS / average inventory value ratio, not annualized.
- Supplier risk flag is null because the repository does not define an approved threshold for that field. Capacity allocation is represented by allocated order quantity against the reported supplier capacity.
- The unsupported short-series analogue fallback remains deferred; paths without complete saved forecasts are excluded with reasons.
- Phase-out stock value and weeks of cover are reported before and after the validation simulation. Obsolete parts P0046/P0087 have no D2 scheduling snapshot; their values and weeks of cover are explicitly `not_measurable`, not inferred.

## Outputs

- `results/phase6_scheduler_comparison_v1.csv`
- `results/phase6_kpi_comparison_v1.csv`
- `results/phase6_weight_sensitivity_v1.csv`
- `results/phase6_paired_bootstrap_v1.csv`
- `results/phase6_cost_sensitivity_v1.csv`
- `results/phase6_initialization_comparison_v1.csv`
- `results/phase6_exclusions_v1.csv`
- `results/phase6_checker_summary_v1.csv`
- `results/phase6_provisional_firm_schedule_v1.csv`
