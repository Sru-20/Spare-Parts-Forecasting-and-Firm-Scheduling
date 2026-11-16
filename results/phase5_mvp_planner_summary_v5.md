# Phase 5 simulated planning demonstration

Method validated on simulated data; this is not evidence of factory savings or a supplier commitment.

## Initialization and schedule evidence

- Primary initialization: origin_visible; 467 firm, 93 unresolved, 3 not ordered; 470 checker-clean including not-ordered runs.
- D2 stress initialization: d2_snapshot_stress; 256 firm, 305 unresolved, 2 not ordered; 258 checker-clean including not-ordered runs.
- Primary firm-order hard-constraint violations: 0; unresolved runs with checker hard failures: 8; MRP ledger failures: 0; simulator ledger failures: 0.
- Initial cover by criticality: `{"High": {"at_least_14_weeks_pct": 10.582010582010582, "less_than_one_week_pct": 0.0, "median_weeks": 4.070671378091873, "p10_weeks": 3.0236663903352303, "p90_weeks": 13.053398209714276, "part_origins": 189, "zero_demand_part_origins": 6}, "Low": {"at_least_14_weeks_pct": 16.417910447761194, "less_than_one_week_pct": 0.0, "median_weeks": 4.0451923886503325, "p10_weeks": 3.0083887667079763, "p90_weeks": 14.653008407777195, "part_origins": 134, "zero_demand_part_origins": 0}, "Medium": {"at_least_14_weeks_pct": 8.333333333333332, "less_than_one_week_pct": 0.0, "median_weeks": 4.078431372549019, "p10_weeks": 3.0218361831505636, "p90_weeks": 11.70000000000002, "part_origins": 240, "zero_demand_part_origins": 2}}`.
- D2 stress initial cover by criticality: `{"High": {"at_least_14_weeks_pct": 3.7037037037037033, "less_than_one_week_pct": 68.78306878306879, "median_weeks": 0.48148148148148145, "p10_weeks": 0.021079789189851695, "p90_weeks": 2.7121673382241998, "part_origins": 189, "zero_demand_part_origins": 6}, "Low": {"at_least_14_weeks_pct": 1.4925373134328357, "less_than_one_week_pct": 62.68656716417911, "median_weeks": 0.5141129396259118, "p10_weeks": 0.07551641596973906, "p90_weeks": 3.103133656509696, "part_origins": 134, "zero_demand_part_origins": 0}, "Medium": {"at_least_14_weeks_pct": 2.9166666666666665, "less_than_one_week_pct": 65.0, "median_weeks": 0.5053571428571428, "p10_weeks": 0.029024841599568613, "p90_weeks": 3.5738181818181833, "part_origins": 240, "zero_demand_part_origins": 2}}`.
- Sensitivity scenarios recorded: baseline, initial_stock_0_5, initial_stock_1_5, due_later_week_3, due_later_week_6, rejection_per_unit_binomial.

## Order examples

- Firm order: part_id=P0001, supplier_id=S005, supplier_slot=1, supplier_type=Domestic, supply_mode=Procurement, lifecycle_status=Active, release_week=1, arrival_week=5, lead_time_days=25.0, effective_lead_time_days=25.0, supplier_risk_score=0.307, supplier_reject_pct=6.01, risk_adjusted=False, order_qty=50, moq=50, order_multiple=1, monthly_capacity_available=1391.1999999999996, sob_target_pct=50.0, arrives_beyond_horizon=False, capacity_period=2025-02, sob_deviation_pp=50.0, run_id=phase5:2025-01-13:P0001, region=ALL, requested_week=1, net_requirement=28, safety_stock=656.2901600653195, service_level_target=0.95, criticality=Medium, planning_priority=Medium, reason=lead_time_demand_plus_quantile_safety_stock, late=False, warnings=nan, status=firm, failed_checks=nan, origin_date=2025-01-13, forecast_model=lightgbm_tweedie_point+lightgbm_quantile_ss, plan_status=firm, independent_checker_status=firm, independent_checker_failed_checks=nan, initialization_scenario=origin_visible
- Unresolved schedule: run_id=phase5:2025-01-13:P0006, part_id=P0006, origin_date=2025-01-13, week_1_start_date=2025-01-20, initialization_scenario=origin_visible, initial_available_qty=132, trailing_mean_weekly_demand=11.923076923076923, complete_observed_history_weeks=13, plan_status=unresolved, status=unresolved, checker_status=unresolved, plan_failed_checks=moq_multiple_rounding|supplier_capacity, independent_checker_failed_checks=unfilled_requirement, feasible=False, failed_checks=moq_multiple_rounding|supplier_capacity|unfilled_requirement, criticality=High

## Limitation

Analogue fallback is blocked: Every catalogue component/vehicle_type pair is unique; no donor exists under the governing Phase 5 rule. No fallback values were fabricated. The empty `analogue_fallback_forecasts.csv` is diagnostic only; it is not a forecast deliverable.
The dataset validator was intentionally not rerun. A previous validator execution read sealed final-test target values for integrity checks, not prediction evaluation; this session used held-validation forecast artifacts only.

Reproduce with `python -m src.scheduling.phase5 --demo`.
