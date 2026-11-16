from __future__ import annotations

import pandas as pd

from src.scheduling.checker import check_schedule
from src.scheduling.engine import build_mrp_schedule, supplier_options


def _part(**overrides):
    part = {
        "part_id": "P-ROBUST",
        "lifecycle_status": "Active",
        "supply_mode": "Procurement",
        "criticality": "High",
        "service_level_target": 0.99,
        "on_hand_qty": 100,
        "reserved_qty": 0,
        "blocked_qty": 0,
        "available_qty": 100,
        "backorder_qty": 0,
        "in_transit_qty": 0,
        "open_po_qty": 0,
        "open_po_due_this_month_qty": 0,
        "open_po_due_later_qty": 0,
        "sup1_id": "S-ROBUST",
        "sup1_type": "Domestic",
        "sup1_sob_pct": 100.0,
        "sup1_lead_time_days": 7.0,
        "sup1_monthly_capacity": 10000.0,
        "sup1_capacity_committed_pct": 0.0,
        "sup1_moq": 10.0,
        "sup1_order_multiple": 5.0,
        "sup1_otd_pct": 100.0,
        "sup1_avg_delay_days": 0.0,
        "sup1_reject_pct": 0.0,
        "sup1_reliability_pct": 100.0,
    }
    part.update(overrides)
    return part


def _forecasts(point: float, p99: float) -> pd.DataFrame:
    return pd.DataFrame({
        "week": range(1, 15),
        "week_start_date": pd.date_range("2025-01-06", periods=14, freq="W-MON"),
        "forecast_qty": [point] * 14,
        "forecast_p99": [p99] * 14,
    })


def _plan(part: dict, forecasts: pd.DataFrame) -> dict:
    return build_mrp_schedule(
        part,
        forecasts,
        run_id="phase9-synthetic-robustness",
        risk_adjusted=True,
    )


def _check(plan: dict, part: dict) -> dict:
    checked_part = dict(part)
    for supplier in supplier_options(part, risk_adjusted=True):
        checked_part[f"sup{supplier['slot']}_lead_time_days"] = (
            supplier["effective_lead_time_days"]
        )
    requirements = plan["weekly"][
        ["part_id", "week", "net_requirement"]
    ].rename(columns={"week": "release_week"})
    return check_schedule(
        plan["orders"],
        {part["part_id"]: checked_part},
        requirements=requirements,
    )


def test_demand_shock_increases_replenishment_without_constraint_violations():
    part = _part()
    normal = _plan(part, _forecasts(10.0, 12.0))
    shocked = _plan(part, _forecasts(20.0, 24.0))

    assert normal["status"] == shocked["status"] == "firm"
    assert shocked["orders"]["order_qty"].sum() > normal["orders"]["order_qty"].sum()
    assert _check(normal, part)["feasible"]
    assert _check(shocked, part)["feasible"]


def test_supplier_delay_is_reflected_in_risk_adjusted_lead_time():
    forecasts = _forecasts(10.0, 12.0)
    normal = _plan(_part(), forecasts)
    delayed_part = _part(sup1_otd_pct=0.0, sup1_avg_delay_days=14.0)
    delayed = _plan(delayed_part, forecasts)

    assert normal["lead_time_days"] == 7.0
    assert delayed["lead_time_days"] == 21.0
    assert delayed["status"] == "firm"
    assert _check(delayed, delayed_part)["feasible"]


def test_infeasible_capacity_remains_unresolved_not_executable():
    part = _part(sup1_monthly_capacity=10.0)
    plan = _plan(part, _forecasts(20.0, 24.0))

    assert plan["status"] == "unresolved"
    assert "supplier_capacity" in plan["failed_checks"]
    assert plan["orders"].empty or plan["orders"]["status"].eq("unresolved").all()
    assert not _check(plan, part)["feasible"]
