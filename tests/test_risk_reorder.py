from __future__ import annotations

import pandas as pd
import pytest

from src.scheduling.engine import (
    build_mrp_schedule,
    gross_up_for_rejection,
    risk_adjusted_lead_days,
)
from src.scheduling.risk_reorder import _risk_levels


def _part(otd=100.0, delay=0.0, reject=0.0, reliability=100.0, **overrides):
    part = {
        "part_id": "P1",
        "lifecycle_status": "Active",
        "supply_mode": "Procurement",
        "criticality": "High",
        "service_level_target": 0.99,
        "on_hand_qty": 2,
        "reserved_qty": 0,
        "blocked_qty": 0,
        "available_qty": 2,
        "backorder_qty": 0,
        "in_transit_qty": 0,
        "open_po_qty": 0,
        "open_po_due_this_month_qty": 0,
        "open_po_due_later_qty": 0,
        "unit_cost": 10.0,
        "current_reorder_point_qty": 10,
    }
    for slot in range(1, 4):
        present = slot == 1
        part.update(
            {
                f"sup{slot}_id": f"S{slot}" if present else None,
                f"sup{slot}_type": "Domestic" if present else None,
                f"sup{slot}_sob_pct": 100.0 if present else None,
                f"sup{slot}_lead_time_days": 7.0 if present else None,
                f"sup{slot}_monthly_capacity": 100000.0 if present else None,
                f"sup{slot}_capacity_committed_pct": 0.0 if present else None,
                f"sup{slot}_moq": 1.0 if present else None,
                f"sup{slot}_order_multiple": 1.0 if present else None,
                f"sup{slot}_otd_pct": otd if present else None,
                f"sup{slot}_avg_delay_days": delay if present else None,
                f"sup{slot}_reject_pct": reject if present else None,
                f"sup{slot}_reliability_pct": reliability if present else None,
            }
        )
    part.update(overrides)
    return part


def _forecasts():
    return pd.DataFrame(
        {
            "week": range(1, 15),
            "week_start_date": pd.date_range("2025-01-06", periods=14, freq="W-MON"),
            "forecast_qty": [10.0] * 14,
            "forecast_p90": [12.0] * 14,
            "forecast_p95": [14.0] * 14,
            "forecast_p99": [16.0] * 14,
        }
    )


def test_risk_adjusted_lead_time_formula():
    assert risk_adjusted_lead_days(30, 80, 5) == 31
    assert risk_adjusted_lead_days(30, 100, 5) == 30


def test_rejection_gross_up():
    assert gross_up_for_rejection(100, 20) == 125
    assert gross_up_for_rejection(50, 0) == 50


def test_risk_adjusted_plan_orders_more_and_later():
    part = _part(otd=80.0, delay=7.0, reject=20.0, reliability=70.0)
    baseline = build_mrp_schedule(part, _forecasts(), risk_adjusted=False)
    adjusted = build_mrp_schedule(part, _forecasts(), risk_adjusted=True)
    assert baseline["orders"]["order_qty"].sum() > 0
    assert adjusted["orders"]["order_qty"].sum() > baseline["orders"]["order_qty"].sum()
    assert adjusted["orders"]["arrival_week"].min() > baseline["orders"]["arrival_week"].min()
    assert adjusted["risk_adjusted"] is True
    assert adjusted["supplier_risk_score"] == pytest.approx(0.3)


def test_plan_exposes_reorder_columns():
    plan = build_mrp_schedule(_part(), _forecasts())
    weekly = plan["weekly"]
    for column in (
        "reorder_point",
        "reorder_trigger_week",
        "below_reorder_point_flag",
        "supplier_risk_score",
    ):
        assert column in weekly
    assert plan["reorder_point"] >= 0
    assert weekly["below_reorder_point_flag"].dtype == bool


def test_risk_levels_split_into_population_tertiles():
    parts = {
        f"P{index}": _part(reliability=100.0 - index * 2) for index in range(1, 7)
    }
    levels = _risk_levels(parts)
    assert sorted(levels.values()).count("low") == 2
    assert sorted(levels.values()).count("medium") == 2
    assert sorted(levels.values()).count("high") == 2
    assert levels["P6"] == "high"
    assert levels["P1"] == "low"
