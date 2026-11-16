from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.scheduling.checker import check_inventory_ledger, check_schedule
from src.scheduling.engine import (
    allocate_supplier_orders,
    calculate_net_requirement,
    calculate_usable_stock,
    order_arrival_week,
    rounded_order_qty,
    select_service_quantile,
    snapshot_receipts,
    build_mrp_schedule,
)
from src.scheduling.phase5 import (
    D2_STRESS_INITIALIZATION,
    PRIMARY_INITIALIZATION,
    _generate_origin_plans,
    _load_yaml_unique,
    _propose_pull_ahead_orders,
    _write_unresolved_reports,
    load_origin_visible_history,
    origin_visible_opening_stock,
    write_analogue_fallback_diagnostic,
)
from src.simulation.weekly import realize_supplier_orders, simulate_weekly

ROOT = Path(__file__).resolve().parents[1]


def _part(**overrides):
    part = {
        "part_id": "P1",
        "lifecycle_status": "Active",
        "supply_mode": "Procurement",
        "criticality": "High",
        "service_level_target": 0.99,
        "on_hand_qty": 10,
        "reserved_qty": 2,
        "blocked_qty": 1,
        "available_qty": 7,
        "backorder_qty": 0,
        "in_transit_qty": 0,
        "open_po_qty": 0,
        "open_po_due_this_month_qty": 0,
        "open_po_due_later_qty": 0,
        "earliest_open_po_eta": None,
        "unit_cost": 10.0,
    }
    for slot, share, lead, cap, commitment, moq, multiple, supplier_type in [
        (1, 100.0, 7.0, 100.0, 0.0, 1.0, 1.0, "Domestic"),
        (2, None, None, None, None, None, None, None),
        (3, None, None, None, None, None, None, None),
    ]:
        part.update(
            {
                f"sup{slot}_id": f"S{slot}" if share is not None else None,
                f"sup{slot}_type": supplier_type,
                f"sup{slot}_sob_pct": share,
                f"sup{slot}_lead_time_days": lead,
                f"sup{slot}_monthly_capacity": cap,
                f"sup{slot}_capacity_committed_pct": commitment,
                f"sup{slot}_moq": moq,
                f"sup{slot}_order_multiple": multiple,
                f"sup{slot}_otd_pct": 100.0 if share is not None else None,
                f"sup{slot}_avg_delay_days": 0.0 if share is not None else None,
                f"sup{slot}_reject_pct": 0.0 if share is not None else None,
                f"sup{slot}_reliability_pct": 100.0 if share is not None else None,
            }
        )
    part.update(overrides)
    return part


def _forecasts(first_week=1.0, **overrides):
    values = {
        "week": range(1, 15),
        "week_start_date": pd.date_range("2025-01-06", periods=14, freq="W-MON"),
        "forecast_qty": [first_week] + [0.0] * 13,
        "forecast_p90": [first_week] + [0.0] * 13,
        "forecast_p95": [first_week] + [0.0] * 13,
        "forecast_p99": [first_week] + [0.0] * 13,
    }
    values.update(overrides)
    return pd.DataFrame(values)


def test_enough_usable_stock_needs_no_replenishment():
    part = _part()
    usable = calculate_usable_stock(part)
    requirement = calculate_net_requirement(
        forecast_qty=5, backorder_qty=0, usable_stock=usable, receipts=0
    )
    assert usable == 7
    assert requirement == 0


def test_mrp_enough_stock_emits_no_order():
    result = build_mrp_schedule(_part(), _forecasts(first_week=5))
    assert result["orders"].empty
    assert result["status"] == "firm"
    assert (result["weekly"]["region"] == "ALL").all()


def test_required_snapshot_values_are_not_silently_defaulted_to_zero():
    with pytest.raises(ValueError, match="available_qty is missing"):
        calculate_usable_stock(_part(available_qty=None))
    with pytest.raises(ValueError, match="open_po_due_later_qty is missing"):
        snapshot_receipts(_part(open_po_due_later_qty=None))


def test_shortage_is_fulfilled_by_receipt_timed_at_week_start():
    result = simulate_weekly(
        initial_stock=0,
        demand=[5],
        receipts=[{"week": 1, "quantity": 5}],
        criticality="High",
    )
    assert result.loc[0, "fulfilled_qty"] == 5
    assert result.loc[0, "closing_stock"] == 0
    assert result.loc[0, "stockout_qty"] == 0


def test_order_that_arrives_after_need_is_marked_late_not_hidden():
    due_week = order_arrival_week(release_week=1, lead_time_days=14)
    assert due_week == 3
    assert due_week > 1
    order = {
        "release_week": 1,
        "arrival_week": due_week,
        "requested_cover_week": 1,
        "late": due_week > 1,
    }
    assert order["late"] is True


def test_shortage_is_covered_by_existing_receipt_at_start_of_week():
    part = _part(
        available_qty=0,
        on_hand_qty=3,
        reserved_qty=2,
        blocked_qty=1,
        open_po_qty=5,
        open_po_due_this_month_qty=5,
    )
    result = build_mrp_schedule(part, _forecasts(first_week=5))
    assert result["weekly"].iloc[0]["expected_receipts"] == 5
    assert result["weekly"].iloc[0]["fulfilled_qty"] == 5


def test_mrp_marks_first_week_need_as_late_when_supplier_lead_exceeds_arrival():
    part = _part()
    forecasts = pd.DataFrame(
        {
            "week": range(1, 15),
            "week_start_date": pd.date_range("2025-01-20", periods=14, freq="W-MON"),
            "forecast_qty": [20.0] + [1.0] * 13,
            "forecast_p90": [21.0] + [1.1] * 13,
            "forecast_p95": [22.0] + [1.2] * 13,
            "forecast_p99": [23.0] + [1.3] * 13,
        }
    )
    result = build_mrp_schedule(part, forecasts)
    assert result["orders"].iloc[0]["arrival_week"] == 2
    assert bool(result["orders"].iloc[0]["late"])
    assert "late_for_current_week_need" in result["orders"].iloc[0]["warnings"]


def test_order_quantity_applies_moq_then_order_multiple():
    assert rounded_order_qty(11, moq=8, order_multiple=5) == 15
    assert rounded_order_qty(0, moq=8, order_multiple=5) == 0


def test_supplier_allocation_applies_moq_and_multiple_to_real_order():
    part = _part(sup1_moq=8, sup1_order_multiple=5)
    result = allocate_supplier_orders(part, 11, release_week=1)
    assert result["orders"][0]["order_qty"] == 15


def test_exhausted_capacity_reports_unresolved_without_fabricating_quantity():
    part = _part(sup1_monthly_capacity=20, sup1_capacity_committed_pct=50)
    allocation = allocate_supplier_orders(
        part, quantity=11, release_week=1, capacity_used={"S1": 10}
    )
    assert allocation["orders"] == []
    assert "capacity" in "|".join(allocation["failed_checks"]).lower()
    assert allocation["status"] == "unresolved"


def test_supplier_share_is_reported_as_a_target_not_a_hard_constraint():
    part = _part(
        sup2_id="S2",
        sup2_type="Domestic",
        sup2_sob_pct=40.0,
        sup2_lead_time_days=7.0,
        sup2_monthly_capacity=100.0,
        sup2_capacity_committed_pct=0.0,
        sup2_moq=1.0,
        sup2_order_multiple=1.0,
        sup2_otd_pct=100.0,
        sup2_avg_delay_days=0.0,
        sup2_reject_pct=0.0,
        sup2_reliability_pct=100.0,
    )
    allocation = allocate_supplier_orders(part, 10, release_week=1)
    assert allocation["status"] == "firm"
    assert allocation["sob_target_deviation_reported"] is True
    assert sum(row["order_qty"] for row in allocation["orders"]) >= 10
    assert all("sob_deviation_pp" in row for row in allocation["orders"])


def test_obsolete_part_never_receives_a_new_order():
    part = _part(lifecycle_status="Obsolete")
    allocation = allocate_supplier_orders(part, quantity=12, release_week=1)
    assert allocation["orders"] == []
    assert "obsolete" in "|".join(allocation["failed_checks"]).lower()


def test_backorder_and_open_po_are_each_counted_once_in_netting():
    usable = calculate_usable_stock(_part())
    requirement = calculate_net_requirement(
        forecast_qty=20,
        backorder_qty=3,
        usable_stock=usable,
        receipts=5 + 4,
    )
    assert requirement == 7


def test_mrp_nets_backorder_and_open_po_without_duplicate_bucket_counting():
    part = _part(
        available_qty=0,
        on_hand_qty=3,
        reserved_qty=2,
        blocked_qty=1,
        backorder_qty=3,
        in_transit_qty=0,
        open_po_qty=9,
        open_po_due_this_month_qty=5,
        open_po_due_later_qty=4,
    )
    result = build_mrp_schedule(part, _forecasts(first_week=10))
    assert result["weekly"].iloc[0]["expected_receipts"] == 5
    assert result["weekly"].iloc[0]["backlog_qty"] == 8


def test_open_po_due_buckets_and_in_transit_are_counted_once():
    part = _part(
        in_transit_qty=4,
        open_po_qty=9,
        open_po_due_this_month_qty=5,
        open_po_due_later_qty=4,
        earliest_open_po_eta=None,
    )
    receipts = snapshot_receipts(part)
    assert receipts == [
        {"week": 1, "quantity": 4},
        {"week": 1, "quantity": 5},
        {"week": 4, "quantity": 4},
    ]
    assert sum(row["quantity"] for row in receipts) == 13
    with pytest.raises(ValueError, match="open_po_qty"):
        snapshot_receipts({**part, "open_po_qty": 10})


@pytest.mark.parametrize(
    ("lead_days", "expected"),
    [(1, 2), (7, 2), (8, 3), (14, 3)],
)
def test_conservative_lead_time_week_mapping(lead_days, expected):
    assert order_arrival_week(release_week=1, lead_time_days=lead_days) == expected


def test_beyond_horizon_arrival_is_retained_and_flagged():
    arrival = order_arrival_week(release_week=1, lead_time_days=102)
    part = _part(
        available_qty=0,
        on_hand_qty=3,
        sup1_lead_time_days=102,
        sup1_monthly_capacity=10000,
    )
    result = build_mrp_schedule(part, _forecasts(first_week=20))
    assert arrival == 16
    assert (result["orders"]["arrival_week"] == 16).any()
    assert result["orders"]["arrives_beyond_horizon"].any()


def test_long_lead_quantile_uses_labelled_last_four_week_extension():
    from src.scheduling.engine import lead_time_demand

    forecasts = pd.DataFrame(
        {"forecast_qty": [1.0] * 14, "forecast_p99": [2.0] * 14}
    )
    assert lead_time_demand(forecasts, 140, "forecast_qty") == pytest.approx(20)
    assert lead_time_demand(forecasts, 140, "forecast_p99") == pytest.approx(40)


def test_critical_backlog_is_fulfilled_before_current_demand():
    result = simulate_weekly(
        initial_stock=4,
        demand=[3],
        receipts=[],
        initial_backlog=3,
        criticality="High",
    )
    row = result.iloc[0]
    assert row["backlog_fulfilled_qty"] == 3
    assert row["current_demand_fulfilled_qty"] == 1
    assert row["backlog_qty"] == 2
    assert row["closing_stock"] == 0


def test_noncritical_remainder_is_lost_not_carried_forward():
    result = simulate_weekly(
        initial_stock=4,
        demand=[7, 0],
        receipts=[],
        initial_backlog=3,
        criticality="Medium",
    )
    assert result.loc[0, "fulfilled_qty"] == 4
    assert result.loc[0, "lost_sales_qty"] == 3
    assert result["backlog_qty"].tolist() == [0, 0]


def test_inventory_conservation_and_closing_stock_never_negative():
    result = simulate_weekly(
        initial_stock=3,
        demand=[5, 2],
        receipts=[{"week": 2, "quantity": 4}],
        criticality="High",
    )
    assert (
        result["opening_stock"] + result["receipts_qty"] - result["fulfilled_qty"]
    ).tolist() == result["closing_stock"].tolist()
    assert result["closing_stock"].ge(0).all()


def test_independent_inventory_checker_rejects_bad_reconciliation():
    ledger = pd.DataFrame(
        [
            {
                "opening": 2,
                "receipts": 3,
                "fulfilled": 4,
                "closing": 2,
            }
        ]
    )
    result = check_inventory_ledger(
        ledger, "opening", "receipts", "fulfilled", "closing"
    )
    assert not result["feasible"]
    assert "inventory_conservation" in result["failed_checks"]


def test_supplier_delay_and_rejection_are_seed_deterministic():
    orders = pd.DataFrame(
        [
            {
                "supplier_id": "S1",
                "release_week": 1,
                "arrival_week": 2,
                "order_qty": 10,
            }
            for _ in range(20)
        ]
    )
    suppliers = {
        "S1": {
            "sup1_otd_pct": 75,
            "sup1_avg_delay_days": 8,
            "sup1_reject_pct": 10,
        }
    }
    first = realize_supplier_orders(orders, suppliers, seed=42)
    second = realize_supplier_orders(orders, suppliers, seed=42)
    pd.testing.assert_frame_equal(first, second)


def test_whole_lot_and_per_unit_rejection_variants_are_valid_and_deterministic():
    orders = pd.DataFrame(
        [
            {
                "supplier_id": "S1",
                "supplier_slot": 1,
                "release_week": 1,
                "arrival_week": 2,
                "order_qty": 100,
            }
        ]
    )
    suppliers = {
        "S1": {
            "sup1_otd_pct": 100,
            "sup1_avg_delay_days": 0,
            "sup1_reject_pct": 10,
        }
    }
    first = realize_supplier_orders(
        orders, suppliers, seed=42, rejection_variant="per_unit_binomial"
    )
    second = realize_supplier_orders(
        orders, suppliers, seed=42, rejection_variant="per_unit_binomial"
    )
    pd.testing.assert_frame_equal(first, second)
    assert 0 <= first.iloc[0]["received_qty"] <= 100
    assert first.iloc[0]["received_qty"] + first.iloc[0]["rejected_qty"] == 100


def test_critical_stockout_is_counted_before_emergency_receipt():
    result = simulate_weekly(
        initial_stock=0,
        demand=[5, 0],
        receipts=[],
        criticality="High",
        emergency_rule=True,
    )
    assert result.loc[0, "stockout_at_close_qty"] == 5
    assert bool(result.loc[0, "stockout_at_close_flag"])
    assert result.loc[0, "emergency_order_qty"] == 5
    assert result.loc[1, "receipts_qty"] == 5


def test_supplier_realization_rejects_missing_supplier_inputs():
    orders = pd.DataFrame(
        [{"supplier_id": "S1", "release_week": 1, "arrival_week": 2, "order_qty": 1}]
    )
    with pytest.raises(ValueError, match="Missing supplier snapshot"):
        realize_supplier_orders(orders, {}, seed=42)


def test_independent_checker_rejects_deliberately_invalid_schedule():
    part = _part(sup1_moq=10, sup1_order_multiple=5)
    schedule = pd.DataFrame(
        [
            {
                "part_id": "P1",
                "supplier_id": "S1",
                "supply_mode": "Procurement",
                "lifecycle_status": "Active",
                "release_week": 1,
                "arrival_week": 1,
                "lead_time_days": 7,
                "order_qty": 7,
                "moq": 10,
                "order_multiple": 5,
                "monthly_capacity_available": 100,
                "status": "firm",
            }
        ]
    )
    result = check_schedule(schedule, {"P1": part}, horizon_weeks=14)
    assert not result["feasible"]
    assert {
        "moq",
        "moq_multiple_rounding",
        "order_multiple",
        "lead_time",
    } <= set(result["failed_checks"])


def test_week_one_unmet_demand_is_flagged_as_release_before_week_one():
    part = _part(
        available_qty=0, on_hand_qty=3, reserved_qty=2, blocked_qty=1
    )
    result = build_mrp_schedule(
        part, _forecasts(first_week=5)
    )

    assert result["status"] == "unresolved"
    assert "release_before_week_1" in result["failed_checks"]
    checked = check_schedule(result["orders"], {"P1": part})
    assert "release_before_week_1" in checked["failed_checks"]


def test_moq_multiple_rounding_reason_is_reported_when_capacity_is_below_moq():
    allocation = allocate_supplier_orders(
        _part(
            sup1_moq=10,
            sup1_order_multiple=5,
            sup1_monthly_capacity=5,
        ),
        quantity=10,
        release_week=1,
        capacity_period="2025-01",
    )

    assert allocation["orders"] == []
    assert "supplier_capacity" in allocation["failed_checks"]
    assert "moq_multiple_rounding" in allocation["failed_checks"]


def test_independent_checker_enforces_supplier_month_capacity_across_orders():
    part = _part(sup1_monthly_capacity=20, sup1_capacity_committed_pct=50)
    schedule = pd.DataFrame(
        [
            {
                "run_id": "r1",
                "origin_date": "2025-01-13",
                "part_id": "P1",
                "supplier_id": "S1",
                "release_week": 1,
                "arrival_week": 2,
                "lead_time_days": 7,
                "order_qty": qty,
                "capacity_period": "2025-01",
                "net_requirement": 12,
            }
            for qty in (6, 6)
        ]
    )
    result = check_schedule(schedule, {"P1": part})
    assert not result["feasible"]
    assert "supplier_capacity" in result["failed_checks"]


def test_checker_checks_requirement_per_release_not_across_the_whole_plan():
    part = _part()
    schedule = pd.DataFrame(
        [
            {
                "run_id": "r1",
                "part_id": "P1",
                "supplier_id": "S1",
                "release_week": week,
                "arrival_week": week + 1,
                "lead_time_days": 7,
                "order_qty": 5,
                "net_requirement": 5,
            }
            for week in (1, 2)
        ]
    )
    result = check_schedule(schedule, {"P1": part})
    assert result["feasible"]


def test_independent_checker_catches_a_missing_order_for_a_week():
    part = _part()
    requirements = pd.DataFrame(
        [
            {
                "run_id": "r1",
                "part_id": "P1",
                "release_week": 2,
                "net_requirement": 5,
            }
        ]
    )
    result = check_schedule(
        pd.DataFrame(),
        {"P1": part},
        requirements=requirements,
    )
    assert not result["feasible"]
    assert "unfilled_requirement" in result["failed_checks"]


def test_independent_checker_rejects_malformed_requirements():
    result = check_schedule(
        pd.DataFrame(),
        {"P1": _part()},
        requirements=pd.DataFrame([{"part_id": "P1", "release_week": 1}]),
    )
    assert not result["feasible"]
    assert any(
        failure.startswith("missing_requirement_fields")
        for failure in result["failed_checks"]
    )


@pytest.mark.parametrize(
    ("service_level", "expected"),
    [(0.90, ("forecast_p90", 0.90)), (0.91, ("forecast_p95", 0.95)),
     (0.96, ("forecast_p99", 0.99)), (0.995, ("forecast_p99", 0.99))],
)
def test_service_quantile_rounds_up_and_caps_at_p99(service_level, expected):
    assert select_service_quantile(service_level) == expected


def test_pull_ahead_uses_earlier_month_capacity_and_keeps_lead_time():
    part = _part(sup1_monthly_capacity=100, sup1_capacity_committed_pct=0)
    forecasts = _forecasts()
    forecasts["week_start_date"] = pd.date_range(
        "2025-01-06", periods=14, freq="W-MON"
    )
    item = {
        "run_id": "phase5:2025-01-06:P1",
        "part_id": "P1",
        "snapshot": part,
        "forecast_group": forecasts,
        "orders": pd.DataFrame(),
        "plan": {
            "failed_checks": ["supplier_capacity"],
            "weekly": pd.DataFrame(
                [
                    {
                        "week": 5,
                        "week_start_date": pd.Timestamp("2025-02-03"),
                        "net_requirement": 8,
                        "criticality": "High",
                    }
                ]
            ),
        },
    }
    pulled = _propose_pull_ahead_orders(
        pd.Timestamp("2025-01-06"),
        [item],
        {("S1", "2025-02"): 100},
    )["P1"]
    assert len(pulled) == 1
    assert pulled[0]["capacity_period"] == "2025-01"
    assert pulled[0]["release_week"] >= 1
    assert pulled[0]["arrival_week"] == pulled[0]["release_week"] + 1
    assert pulled[0]["arrival_week"] < 5 + 1


def test_phase5_assumption_hashes_match_frozen_document_and_config_version():
    assumptions = (ROOT / "docs" / "phase5_assumptions.md").read_text(
        encoding="utf-8"
    )
    config = yaml.safe_load(
        (ROOT / "configs" / "project.yaml").read_text(encoding="utf-8")
    )
    version = re.search(r"\*\*Version:\*\*\s+`([^`]+)`", assumptions)
    assert version is not None
    assert (
        config["scheduling"]["simulator_assumptions"]["version"]
        == version.group(1)
    )
    for relative_path in ("configs/project.yaml", "configs/business.yaml"):
        recorded = re.search(
            rf"`{re.escape(relative_path)}`:\s+`([0-9A-F]{{64}})`",
            assumptions,
        )
        assert recorded is not None
        current = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        assert current.upper() == recorded.group(1)


def test_emergency_assumption_config_keeps_rule_and_cost_together():
    config = yaml.safe_load(
        (ROOT / "configs" / "project.yaml").read_text(encoding="utf-8")
    )
    emergency = config["scheduling"]["simulator_assumptions"][
        "emergency_procurement"
    ]
    assert emergency["trigger"] == "critical_backlog_remaining_at_week_close"
    assert emergency["quantity"] == "exact_remaining_critical_backlog"
    assert emergency["arrival"] == "start_of_next_week"
    assert emergency["unit_cost_multiplier"] == 1.5


def test_origin_visible_history_skips_sealed_and_future_target_values(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    path = data / "01_demand_forecasting.csv"
    columns = [
        "week_start_date",
        "part_id",
        "region",
        "split_role",
        "demand_qty",
        "is_missing_week",
    ]
    rows = [
        [
            week.strftime("%Y-%m-%d"),
            "P1",
            "R1",
            "train_val",
            "2",
            "0",
        ]
        for week in pd.date_range("2025-01-06", periods=8, freq="W-MON")
    ]
    rows.extend(
        [
            ["2025-02-24", "P2", "R1", "sealed_final_test", "not-a-number", "0"],
            ["2025-03-10", "P2", "R1", "future", "not-a-number", "0"],
        ]
    )
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)

    history = load_origin_visible_history(
        tmp_path, [pd.Timestamp("2025-03-03")]
    )

    assert history[("2025-03-03", "P1")]["complete_observed_weeks"] == 8
    assert history[("2025-03-03", "P1")]["trailing_mean_weekly_demand"] == 2
    assert (
        history[("2025-03-03", "P1")]["origin_visible_sbc_class"]
        == "Unknown/ShortHistory"
    )
    assert ("2025-03-03", "P2") not in history


def test_origin_visible_opening_stock_uses_exact_eight_week_formula():
    result = origin_visible_opening_stock(
        _part(sup1_lead_time_days=8),
        {
            "complete_observed_weeks": 8,
            "trailing_mean_weekly_demand": 3.2,
        },
    )

    assert result["rounded_weekly_mean_demand"] == 4
    assert result["lead_time_weeks"] == 2
    assert result["opening_available_qty"] == 12


def test_origin_visible_opening_stock_rejects_fewer_than_eight_weeks():
    with pytest.raises(ValueError, match="insufficient_origin_visible_history"):
        origin_visible_opening_stock(
            _part(),
            {
                "complete_observed_weeks": 7,
                "trailing_mean_weekly_demand": 3.2,
            },
        )


def test_primary_and_d2_initialization_are_separate_scenarios():
    origin = pd.Timestamp("2025-03-03")
    part = _part(
        backorder_qty=3,
        open_po_qty=5,
        open_po_due_this_month_qty=5,
    )
    paths = _forecasts(first_week=0)
    paths["part_id"] = "P1"
    history = {
        (origin.date().isoformat(), "P1"): {
            "complete_observed_weeks": 8,
            "trailing_mean_weekly_demand": 2.0,
        }
    }

    primary, _, _ = _generate_origin_plans(
        origin, paths, {"P1": part}, {}, {}, 1.0, 4,
        PRIMARY_INITIALIZATION, history,
    )
    stress, _, _ = _generate_origin_plans(
        origin, paths, {"P1": part}, {}, {}, 1.0, 4,
        D2_STRESS_INITIALIZATION, history,
    )

    assert primary[0]["snapshot"]["available_qty"] == 4
    assert primary[0]["snapshot"]["open_po_qty"] == 0
    assert primary[0]["snapshot"]["backorder_qty"] == 0
    assert stress[0]["snapshot"]["available_qty"] == part["available_qty"]
    assert stress[0]["snapshot"]["open_po_qty"] == part["open_po_qty"]
    assert stress[0]["snapshot"]["backorder_qty"] == part["backorder_qty"]


def test_phase5_yaml_loader_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text("assumptions:\n  seed: 42\n  seed: 7\n", encoding="utf-8")

    with pytest.raises(yaml.constructor.ConstructorError, match="duplicate key"):
        _load_yaml_unique(path)


def test_unresolved_month_report_includes_first_week_lead_time_failure(tmp_path):
    results = tmp_path / "results"
    checks = pd.DataFrame(
        [
            {
                "run_id": "r1",
                "status": "unresolved",
                "failed_checks": "release_before_week_1",
                "criticality": "High",
                "week_1_start_date": pd.Timestamp("2025-01-06"),
            }
        ]
    )

    _write_unresolved_reports(
        checks, pd.DataFrame(), results, "test"
    )

    monthly = pd.read_csv(results / "test_unresolved_by_month.csv")
    assert monthly.loc[0, "requirement_month"] == "2025-01"
    assert monthly.loc[0, "failure_reason"] == "release_before_week_1"
    assert monthly.loc[0, "unresolved_schedules"] == 1


def test_analogue_exception_records_unforecasted_series_without_zero_fill(tmp_path):
    for folder in ("data", "artifacts", "configs", "results"):
        (tmp_path / folder).mkdir()
    pd.DataFrame(
        [
            {"part_id": "P1", "component": "A", "vehicle_type": "4W Sedan"},
            {"part_id": "P0091", "component": "B", "vehicle_type": "4W Hatch"},
            {"part_id": "P2", "component": "C", "vehicle_type": "2W"},
        ]
    ).to_csv(tmp_path / "data" / "03_spare_parts_catalogue.csv", index=False)
    pd.DataFrame(
        [
            {
                "part_id": "P1",
                "region": "East",
                "classification_status": "unclassified_short",
                "observed_weeks": 5,
            },
            {
                "part_id": "P0091",
                "region": "North",
                "classification_status": "unclassified_no_development_observations",
                "observed_weeks": 0,
            },
            {
                "part_id": "P2",
                "region": "East",
                "classification_status": "classified",
                "observed_weeks": 40,
            },
        ]
    ).to_parquet(tmp_path / "artifacts" / "series_segments.parquet")
    (tmp_path / "configs" / "project.yaml").write_text(
        "phase4:\n"
        "  tier2_random_forest:\n"
        '    held_validation_origins: ["2025-01-13", "2025-04-21"]\n',
        encoding="utf-8",
    )
    legacy_forecasts = tmp_path / "results" / "analogue_fallback_forecasts.csv"
    legacy_diagnostic = tmp_path / "results" / "analogue_fallback_diagnostic.json"
    legacy_forecasts.write_text("legacy forecast artifact\n", encoding="utf-8")
    legacy_diagnostic.write_text("legacy diagnostic artifact\n", encoding="utf-8")

    summary = write_analogue_fallback_diagnostic(tmp_path)
    records = pd.read_csv(
        tmp_path / "results" / "phase5_analogue_exceptions_v1.csv"
    )

    assert len(records) == 4
    assert records["status"].eq("UNFORECASTED_NO_DEFENSIBLE_ANALOGUE").all()
    assert not records["forecast_emitted"].any()
    assert records["eligible_donor_count"].eq(0).all()
    assert "forecast_qty" not in records.columns
    assert summary["series_origin_records"] == 4
    assert summary["forecast_qty_substituted_with_zero"] is False
    p0091 = records.loc[records["part_id"].eq("P0091")]
    assert p0091["reason"].str.contains(
        "no_development_observations;no_validated_launch_exposure_scaling"
    ).all()
    assert legacy_forecasts.read_text(encoding="utf-8") == "legacy forecast artifact\n"
    assert legacy_diagnostic.read_text(encoding="utf-8") == "legacy diagnostic artifact\n"


def test_analogue_exception_refuses_to_hide_an_eligible_exact_donor(tmp_path):
    for folder in ("data", "artifacts", "configs", "results"):
        (tmp_path / folder).mkdir()
    pd.DataFrame(
        [
            {"part_id": "P1", "component": "A", "vehicle_type": "4W Sedan"},
            {"part_id": "P2", "component": "A", "vehicle_type": "4W Sedan"},
        ]
    ).to_csv(tmp_path / "data" / "03_spare_parts_catalogue.csv", index=False)
    pd.DataFrame(
        [
            {
                "part_id": "P1",
                "region": "East",
                "classification_status": "unclassified_short",
                "observed_weeks": 5,
            }
        ]
    ).to_parquet(tmp_path / "artifacts" / "series_segments.parquet")
    (tmp_path / "configs" / "project.yaml").write_text(
        "phase4:\n"
        "  tier2_random_forest:\n"
        '    held_validation_origins: ["2025-01-13"]\n',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Exact component/vehicle-type donors exist"):
        write_analogue_fallback_diagnostic(tmp_path)
