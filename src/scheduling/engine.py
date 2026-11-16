"""Deterministic MRP rules for weekly spare-parts replenishment."""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta
from math import ceil
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SUPPLIER_FIELDS = (
    "id",
    "type",
    "sob_pct",
    "lead_time_days",
    "monthly_capacity",
    "capacity_committed_pct",
    "moq",
    "order_multiple",
    "otd_pct",
    "avg_delay_days",
    "reject_pct",
    "reliability_pct",
)


def _number(value: Any, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return default
    return float(value)


def _required_number(source: Mapping[str, Any], field: str) -> float:
    value = source.get(field)
    if value is None or pd.isna(value):
        raise ValueError(f"Required scheduling input {field} is missing")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"Required scheduling input {field} is not finite")
    return number


def gross_up_for_rejection(quantity: float, reject_pct: float) -> float:
    """Order more so the expected accepted quantity covers the requirement."""
    if not np.isfinite(reject_pct) or not 0 <= reject_pct < 100:
        raise ValueError("Reject rate must be finite and below 100 percent")
    return float(quantity) / (1.0 - float(reject_pct) / 100.0)


def risk_adjusted_lead_days(
    lead_time_days: float, otd_pct: float, avg_delay_days: float
) -> float:
    """Expected lead time including the late-delivery share."""
    if not 0 <= otd_pct <= 100 or avg_delay_days < 0:
        raise ValueError("OTD and average delay values are invalid")
    return float(lead_time_days) + (1.0 - float(otd_pct) / 100.0) * float(
        avg_delay_days
    )


def calculate_usable_stock(snapshot: Mapping[str, Any]) -> int:
    on_hand = _required_number(snapshot, "on_hand_qty")
    reserved = _required_number(snapshot, "reserved_qty")
    blocked = _required_number(snapshot, "blocked_qty")
    reported = _required_number(snapshot, "available_qty")
    quantities = (on_hand, reserved, blocked, reported)
    if any(value < 0 or not value.is_integer() for value in quantities):
        raise ValueError("D2 stock quantities must be non-negative whole units")
    usable = on_hand - reserved - blocked
    if usable < 0 or not np.isclose(usable, reported, atol=0, rtol=0):
        raise ValueError(
            "D2 usable-stock reconciliation failed: on_hand_qty - "
            "reserved_qty - blocked_qty must equal available_qty"
        )
    return int(usable)


def calculate_net_requirement(
    forecast_qty: float,
    backorder_qty: float,
    usable_stock: float,
    receipts: float,
    safety_stock: float = 0,
) -> int:
    need = (
        _number(forecast_qty)
        + _number(backorder_qty)
        + _number(safety_stock)
        - _number(usable_stock)
        - _number(receipts)
    )
    return max(0, int(ceil(need - 1e-12)))


def order_arrival_week(release_week: int, lead_time_days: float) -> int:
    if release_week < 1:
        raise ValueError("Orders cannot be released before week 1")
    if not np.isfinite(lead_time_days) or lead_time_days <= 0:
        raise ValueError("Supplier lead time must be a positive finite number")
    return int(release_week + max(1, ceil(float(lead_time_days) / 7.0)))


def rounded_order_qty(
    net_requirement: float, moq: float, order_multiple: float
) -> int:
    need = max(0, int(ceil(_number(net_requirement) - 1e-12)))
    if need == 0:
        return 0
    minimum = max(1, int(ceil(_number(moq, 1))))
    multiple = max(1, int(ceil(_number(order_multiple, 1))))
    return int(ceil(max(need, minimum) / multiple) * multiple)


def select_service_quantile(service_level: float) -> tuple[str, float]:
    choices = {"forecast_p90": 0.90, "forecast_p95": 0.95, "forecast_p99": 0.99}
    if not 0 < service_level < 1:
        raise ValueError("Service level must be strictly between zero and one")
    return next(
        (
            (column, level)
            for column, level in choices.items()
            if level >= service_level
        ),
        ("forecast_p99", 0.99),
    )


def lead_time_demand(
    forecasts: pd.DataFrame, lead_time_days: float, value_column: str
) -> float:
    if value_column not in forecasts:
        raise ValueError(f"Missing forecast column: {value_column}")
    lead_weeks = max(1, ceil(float(lead_time_days) / 7.0))
    values = pd.to_numeric(forecasts[value_column], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    if not np.isfinite(values).all():
        raise ValueError(f"{value_column} contains missing or non-finite values")
    if lead_weeks <= len(values):
        return float(values[:lead_weeks].sum())
    if len(values) < 4:
        raise ValueError("At least four weeks are required for long-lead extension")
    extension = float(values[-4:].mean()) * (lead_weeks - len(values))
    return float(values.sum() + extension)


def calculate_safety_stock(
    forecasts: pd.DataFrame, lead_time_days: float, service_level: float
) -> dict[str, float | str]:
    column, quantile = select_service_quantile(service_level)
    point_demand = lead_time_demand(forecasts, lead_time_days, "forecast_qty")
    quantile_demand = lead_time_demand(forecasts, lead_time_days, column)
    return {
        "quantile_column": column,
        "quantile": quantile,
        "lead_time_point_demand": point_demand,
        "lead_time_quantile_demand": quantile_demand,
        "safety_stock": max(0.0, quantile_demand - point_demand),
    }


def supplier_options(
    part: Mapping[str, Any], risk_adjusted: bool = False
) -> list[dict[str, Any]]:
    options = []
    for slot in range(1, 4):
        values = {
            field: part.get(f"sup{slot}_{field}") for field in SUPPLIER_FIELDS
        }
        if not values["id"] or pd.isna(values["id"]):
            continue
        required_fields = (
            "type",
            "sob_pct",
            "lead_time_days",
            "monthly_capacity",
            "capacity_committed_pct",
            "moq",
            "order_multiple",
            "otd_pct",
            "avg_delay_days",
            "reject_pct",
        )
        for field in required_fields:
            value = values[field]
            if value is None or pd.isna(value):
                raise ValueError(
                    f"Required supplier input sup{slot}_{field} is missing"
                )
            if field != "type" and not np.isfinite(float(value)):
                raise ValueError(
                    f"Required supplier input sup{slot}_{field} is not finite"
                )
        values["slot"] = slot
        values["sob_pct"] = float(values["sob_pct"])
        values["lead_time_days"] = float(values["lead_time_days"])
        values["monthly_capacity"] = float(values["monthly_capacity"])
        values["capacity_committed_pct"] = float(
            values["capacity_committed_pct"]
        )
        values["otd_pct"] = float(values["otd_pct"])
        values["avg_delay_days"] = float(values["avg_delay_days"])
        values["reject_pct"] = float(values["reject_pct"])
        if not 0 <= values["sob_pct"] <= 100:
            raise ValueError(f"sup{slot}_sob_pct must be between 0 and 100")
        if values["lead_time_days"] <= 0:
            raise ValueError(f"sup{slot}_lead_time_days must be positive")
        if values["monthly_capacity"] < 0 or not 0 <= values[
            "capacity_committed_pct"
        ] <= 100:
            raise ValueError(f"sup{slot} capacity values are invalid")
        if values["moq"] <= 0 or values["order_multiple"] <= 0:
            raise ValueError(f"sup{slot} MOQ and order multiple must be positive")
        if not 0 <= values["otd_pct"] <= 100 or not 0 <= values[
            "reject_pct"
        ] <= 100 or values["avg_delay_days"] < 0:
            raise ValueError(f"sup{slot} delivery and rejection values are invalid")
        values["moq"] = max(1, int(ceil(_number(values["moq"], 1))))
        values["order_multiple"] = max(
            1, int(ceil(_number(values["order_multiple"], 1)))
        )
        reliability = values.get("reliability_pct")
        if reliability is None or pd.isna(reliability):
            raise ValueError(
                f"Required supplier input sup{slot}_reliability_pct is missing"
            )
        reliability = float(reliability)
        if not 0 <= reliability <= 100:
            raise ValueError(f"sup{slot}_reliability_pct must be between 0 and 100")
        values["risk_score"] = 1.0 - reliability / 100.0
        values["effective_lead_time_days"] = (
            risk_adjusted_lead_days(
                values["lead_time_days"],
                values["otd_pct"],
                values["avg_delay_days"],
            )
            if risk_adjusted
            else values["lead_time_days"]
        )
        options.append(values)
    return options


def sob_weighted_lead_time(
    part: Mapping[str, Any], risk_adjusted: bool = False
) -> float:
    options = supplier_options(part, risk_adjusted)
    if not options or sum(row["sob_pct"] for row in options) <= 0:
        raise ValueError("No usable supplier lead-time/SOB mapping")
    total = sum(row["sob_pct"] for row in options)
    return float(
        sum(
            row["sob_pct"] * row["effective_lead_time_days"]
            for row in options
        )
        / total
    )


def part_supplier_risk_score(part: Mapping[str, Any]) -> float:
    options = supplier_options(part)
    total = sum(row["sob_pct"] for row in options)
    if not options or total <= 0:
        return float("nan")
    return float(
        sum(row["sob_pct"] * row["risk_score"] for row in options) / total
    )


def snapshot_receipts(
    part: Mapping[str, Any], due_later_week: int = 4
) -> list[dict[str, int]]:
    if due_later_week < 1:
        raise ValueError("Due-later receipt week must be at least 1")
    open_po = _required_number(part, "open_po_qty")
    due_this = _required_number(part, "open_po_due_this_month_qty")
    due_later = _required_number(part, "open_po_due_later_qty")
    in_transit = _required_number(part, "in_transit_qty")
    if any(
        value < 0 or not value.is_integer()
        for value in (open_po, due_this, due_later, in_transit)
    ):
        raise ValueError("D2 inbound quantities must be non-negative whole units")
    if not np.isclose(open_po, due_this + due_later, atol=0, rtol=0):
        raise ValueError(
            "D2 open_po_qty must equal due-this-month plus due-later buckets"
        )
    return [
        {"week": 1, "quantity": int(in_transit)},
        {"week": 1, "quantity": int(due_this)},
        {"week": due_later_week, "quantity": int(due_later)},
    ]


def allocate_supplier_orders(
    part: Mapping[str, Any],
    quantity: int,
    release_week: int,
    capacity_used: Mapping[str, float] | None = None,
    horizon_weeks: int = 14,
    capacity_period: Any = None,
    risk_adjusted: bool = False,
) -> dict[str, Any]:
    requested = max(0, int(quantity))
    if requested == 0:
        return {"orders": [], "failed_checks": [], "status": "firm"}
    if str(part.get("lifecycle_status", "")).casefold() == "obsolete":
        return {
            "orders": [],
            "failed_checks": ["obsolete_part_order"],
            "status": "unresolved",
        }
    options = supplier_options(part, risk_adjusted)
    if str(part.get("supply_mode", "")).casefold() == "production":
        options = [
            row for row in options if str(row["type"]).casefold() != "import"
        ]
    failed: list[str] = []
    if not options:
        return {
            "orders": [],
            "failed_checks": [*failed, "no_eligible_supplier"],
            "status": "unresolved",
        }
    capacities = dict(capacity_used or {})

    def period_for(row: Mapping[str, Any]) -> Any:
        if capacity_period is None:
            return (release_week - 1) // 4
        if isinstance(capacity_period, (date, datetime, pd.Timestamp)):
            arrival = order_arrival_week(
                release_week, row["effective_lead_time_days"]
            )
            return (
                pd.Timestamp(capacity_period)
                + pd.Timedelta(days=7 * (arrival - 1))
            ).strftime("%Y-%m")
        return capacity_period

    options.sort(key=lambda row: (-row["sob_pct"], row["slot"]))
    target_total = sum(row["sob_pct"] for row in options)
    remaining = requested
    allocated: dict[str, int] = defaultdict(int)
    multiple_blocked_quantity = False
    for row in options:
        if remaining <= 0:
            break
        period = period_for(row)
        share = row["sob_pct"] / target_total if target_total > 0 else 0
        coverage = max(1, int(ceil(requested * share - 1e-12)))
        gross = (
            gross_up_for_rejection(coverage, row["reject_pct"])
            if risk_adjusted
            else float(coverage)
        )
        proposed = rounded_order_qty(gross, row["moq"], row["order_multiple"])
        capacity = row["monthly_capacity"] * (
            1.0 - row["capacity_committed_pct"] / 100.0
        )
        capacity_key = (str(row["id"]), period)
        used = _number(capacities.get(capacity_key, capacities.get(str(row["id"]))))
        available = max(0.0, capacity - used)
        max_multiple = int(available // row["order_multiple"]) * row[
            "order_multiple"
        ]
        if max_multiple < row["moq"]:
            multiple_blocked_quantity = multiple_blocked_quantity or available > 0
            continue
        amount = min(proposed, max_multiple)
        if amount < row["moq"]:
            multiple_blocked_quantity = multiple_blocked_quantity or available > 0
            continue
        if amount < proposed and available > 0:
            multiple_blocked_quantity = True
        allocated[str(row["id"])] += amount
        capacities[capacity_key] = used + amount
        covered = (
            min(float(coverage), amount * (1.0 - row["reject_pct"] / 100.0))
            if risk_adjusted
            else float(amount)
        )
        remaining -= int(ceil(covered))

    if remaining > 0:
        for row in options:
            if remaining <= 0:
                break
            supplier_id = str(row["id"])
            period = period_for(row)
            capacity_key = (supplier_id, period)
            capacity = row["monthly_capacity"] * (
                1.0 - row["capacity_committed_pct"] / 100.0
            )
            available = max(
                0.0,
                capacity
                - _number(capacities.get(capacity_key, capacities.get(supplier_id))),
            )
            gross_remaining = (
                gross_up_for_rejection(remaining, row["reject_pct"])
                if risk_adjusted
                else float(remaining)
            )
            rounded_remaining = rounded_order_qty(
                gross_remaining, row["moq"], row["order_multiple"]
            )
            available_multiple = (
                int(available // row["order_multiple"]) * row["order_multiple"]
            )
            amount = min(rounded_remaining, available_multiple)
            if amount < rounded_remaining and available > 0:
                multiple_blocked_quantity = True
            if amount < row["moq"]:
                multiple_blocked_quantity = multiple_blocked_quantity or available > 0
                continue
            allocated[supplier_id] += amount
            capacities[capacity_key] = (
                _number(capacities.get(capacity_key, capacities.get(supplier_id)))
                + amount
            )
            covered = (
                min(
                    float(remaining),
                    amount * (1.0 - row["reject_pct"] / 100.0),
                )
                if risk_adjusted
                else float(amount)
            )
            remaining -= int(ceil(covered))

    orders = []
    allocated_total = sum(allocated.values())
    for row in options:
        amount = allocated.get(str(row["id"]), 0)
        if not amount:
            continue
        arrival = order_arrival_week(
            release_week, row["effective_lead_time_days"]
        )
        period = period_for(row)
        orders.append(
            {
                "part_id": part.get("part_id"),
                "supplier_id": str(row["id"]),
                "supplier_slot": row["slot"],
                "supplier_type": row["type"],
                "supply_mode": part.get("supply_mode"),
                "lifecycle_status": part.get("lifecycle_status"),
                "release_week": int(release_week),
                "arrival_week": arrival,
                "lead_time_days": row["lead_time_days"],
                "effective_lead_time_days": row["effective_lead_time_days"],
                "supplier_risk_score": row["risk_score"],
                "supplier_reject_pct": row["reject_pct"],
                "risk_adjusted": bool(risk_adjusted),
                "order_qty": int(amount),
                "moq": row["moq"],
                "order_multiple": row["order_multiple"],
                "monthly_capacity_available": row["monthly_capacity"]
                * (1.0 - row["capacity_committed_pct"] / 100.0),
                "sob_target_pct": row["sob_pct"],
                "arrives_beyond_horizon": arrival > horizon_weeks,
                "capacity_period": period,
            }
        )
    if remaining > 0:
        failed.append("supplier_capacity")
        if multiple_blocked_quantity:
            failed.append("moq_multiple_rounding")
    if allocated_total < requested and "supplier_capacity" not in failed:
        failed.append("supplier_capacity")
    allocated_shares = {
        row["supplier_id"]: row["order_qty"] * 100.0 / allocated_total
        for row in orders
    } if allocated_total else {}
    for row in orders:
        row["sob_deviation_pp"] = allocated_shares[row["supplier_id"]] - row[
            "sob_target_pct"
        ]
    return {
        "orders": orders,
        "failed_checks": sorted(set(failed)),
        "status": "firm" if not failed else "unresolved",
        "requested_qty": requested,
        "allocated_qty": allocated_total,
        "capacity_used": capacities,
        "sob_target_deviation_reported": bool(orders),
    }


def build_mrp_schedule(
    part: Mapping[str, Any],
    weekly_forecasts: pd.DataFrame,
    run_id: str = "phase5-validation",
    horizon_weeks: int = 14,
    capacity_used: Mapping[str, float] | None = None,
    due_later_week: int = 4,
    initial_stock_multiplier: float = 1.0,
    preplanned_orders: Sequence[Mapping[str, Any]] = (),
    risk_adjusted: bool = False,
) -> dict[str, Any]:
    """Build a deterministic 14-week MRP plan from origin-safe forecasts."""
    if len(weekly_forecasts) != horizon_weeks:
        raise ValueError(f"Expected exactly {horizon_weeks} forecast weeks")
    forecasts = weekly_forecasts.sort_values("week").reset_index(drop=True).copy()
    if forecasts["week"].astype(int).tolist() != list(range(1, horizon_weeks + 1)):
        raise ValueError("Forecast weeks must be consecutive and start at week 1")
    forecasts["week_start_date"] = pd.to_datetime(
        forecasts["week_start_date"], errors="coerce"
    ) if "week_start_date" in forecasts else pd.NaT
    if str(part.get("lifecycle_status", "")).casefold() == "obsolete":
        return {
            "run_id": run_id,
            "part_id": part.get("part_id"),
            "orders": pd.DataFrame(),
            "weekly": pd.DataFrame(),
            "safety_stock": np.nan,
            "status": "firm",
            "failed_checks": [],
            "capacity_used": dict(capacity_used or {}),
        }
    try:
        if not np.isfinite(initial_stock_multiplier) or initial_stock_multiplier < 0:
            raise ValueError("Initial stock multiplier must be finite and non-negative")
        available = int(
            np.floor(calculate_usable_stock(part) * initial_stock_multiplier)
        )
        lead_days = sob_weighted_lead_time(part, risk_adjusted)
        supplier_risk_score = part_supplier_risk_score(part)
        existing_receipts = snapshot_receipts(part, due_later_week)
        protection = calculate_safety_stock(
            forecasts, lead_days, _number(part.get("service_level_target"))
        )
        reorder_point = (
            float(protection["lead_time_point_demand"])
            + float(protection["safety_stock"])
        )
    except (ValueError, KeyError) as error:
        return {
            "run_id": run_id,
            "part_id": part.get("part_id"),
            "orders": pd.DataFrame(),
            "weekly": pd.DataFrame(),
            "safety_stock": np.nan,
            "status": "unresolved",
            "failed_checks": [str(error)],
            "capacity_used": dict(capacity_used or {}),
        }
    selected_column, selected_quantile = select_service_quantile(
        _number(part["service_level_target"])
    )
    stock = available
    critical = str(part.get("criticality")) == "High"
    backlog = int(_number(part.get("backorder_qty"))) if critical else 0
    receipt_queue: dict[int, int] = defaultdict(int)
    for receipt in existing_receipts:
        receipt_queue[receipt["week"]] += receipt["quantity"]
    capacity_state = dict(capacity_used or {})
    order_rows: list[dict[str, Any]] = []
    for planned in preplanned_orders:
        order = dict(planned)
        if str(order.get("part_id")) != str(part.get("part_id")):
            raise ValueError("Preplanned order belongs to a different part")
        arrival_week = int(order["arrival_week"])
        quantity = int(order["order_qty"])
        if arrival_week < 1 or quantity <= 0:
            raise ValueError("Preplanned order has an invalid arrival or quantity")
        order["run_id"] = run_id
        order["region"] = "ALL"
        order["requested_week"] = int(order["planned_for_week"])
        order["reason"] = "capacity_pull_ahead"
        receipt_queue[arrival_week] += quantity
        order_rows.append(order)
    weekly_rows: list[dict[str, Any]] = []
    failed_checks: set[str] = set()
    safety_stock_max = 0.0
    for week in range(1, horizon_weeks + 1):
        forecast_row = forecasts.iloc[week - 1]
        opening_stock = stock
        expected_receipts = receipt_queue.pop(week, 0)
        stock += expected_receipts
        arrival = order_arrival_week(week, lead_days)
        cover_weeks = max(1, arrival - week)
        point_values = pd.to_numeric(
            forecasts["forecast_qty"], errors="coerce"
        ).to_numpy(dtype=float, na_value=np.nan)
        quantile_values = pd.to_numeric(
            forecasts[selected_column], errors="coerce"
        ).to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(point_values).all() or not np.isfinite(
            quantile_values
        ).all():
            return {
                "run_id": run_id,
                "part_id": part.get("part_id"),
                "orders": pd.DataFrame(order_rows),
                "weekly": pd.DataFrame(weekly_rows),
                "safety_stock": np.nan,
                "status": "unresolved",
                "failed_checks": ["missing_or_non_finite_forecast"],
                "capacity_used": capacity_state,
            }
        start = week - 1
        stop = min(horizon_weeks, start + cover_weeks)
        point_demand = float(point_values[start:stop].sum())
        quantile_demand = float(quantile_values[start:stop].sum())
        beyond = max(0, cover_weeks - (stop - start))
        if beyond:
            point_demand += float(point_values[-4:].mean()) * beyond
            quantile_demand += float(quantile_values[-4:].mean()) * beyond
        safety_stock = max(0.0, quantile_demand - point_demand)
        safety_stock_max = max(safety_stock_max, safety_stock)
        future_inbound = sum(
            quantity
            for receipt_week, quantity in receipt_queue.items()
            if week < receipt_week < arrival
        )
        net = calculate_net_requirement(
            point_demand,
            backlog,
            stock,
            future_inbound,
            safety_stock,
        )
        allocation = allocate_supplier_orders(
            part,
            net,
            release_week=week,
            capacity_used=capacity_state,
            horizon_weeks=horizon_weeks,
            capacity_period=(
                pd.Timestamp(forecasts.loc[0, "week_start_date"])
                if pd.notna(forecasts.loc[0, "week_start_date"])
                else None
            ),
            risk_adjusted=risk_adjusted,
        )
        capacity_state = allocation.get("capacity_used", capacity_state)
        failed_checks.update(allocation["failed_checks"])
        if (
            week == 1
            and backlog + float(forecast_row["forecast_qty"]) > stock
            and allocation["orders"]
        ):
            failed_checks.add("release_before_week_1")
        for order in allocation["orders"]:
            order = dict(order)
            order.update(
                {
                    "run_id": run_id,
                    "region": "ALL",
                    "requested_week": week,
                    "net_requirement": net,
                    "safety_stock": safety_stock,
                    "service_level_target": _number(part["service_level_target"]),
                    "criticality": part.get("criticality"),
                    "planning_priority": part.get("planning_priority"),
                    "reason": "lead_time_demand_plus_quantile_safety_stock",
                    "late": (
                        order["arrival_week"] > week
                        and backlog + float(forecast_row["forecast_qty"]) > stock
                    ),
                }
            )
            warnings = []
            if order["arrives_beyond_horizon"]:
                warnings.append("arrives_beyond_horizon")
            if order["late"]:
                warnings.append("late_for_current_week_need")
            order["warnings"] = "|".join(warnings)
            order_rows.append(order)
            receipt_queue[order["arrival_week"]] += int(order["order_qty"])

        current_demand = int(ceil(_number(forecast_row["forecast_qty"])))
        backlog_open = backlog
        backlog_fulfilled = min(stock, backlog) if critical else 0
        stock -= backlog_fulfilled
        backlog_remaining = backlog - backlog_fulfilled if critical else 0
        current_fulfilled = min(stock, current_demand)
        stock -= current_fulfilled
        shortage = current_demand - current_fulfilled
        backlog = backlog_remaining + shortage if critical else 0
        fulfilled = backlog_fulfilled + current_fulfilled
        weekly_rows.append(
            {
                "run_id": run_id,
                "part_id": part.get("part_id"),
                "region": "ALL",
                "week": week,
                "week_start_date": forecast_row["week_start_date"],
                "forecast_qty": float(forecast_row["forecast_qty"]),
                "opening_usable_stock": opening_stock,
                "expected_receipts": expected_receipts,
                "projected_inventory": stock,
                "backlog_qty": backlog,
                "net_requirement": net,
                "safety_stock": safety_stock,
                "reorder_point": reorder_point,
                "below_reorder_point_flag": stock <= reorder_point,
                "supplier_risk_score": supplier_risk_score,
                "risk_adjusted": bool(risk_adjusted),
                "service_level_target": _number(part["service_level_target"]),
                "criticality": part.get("criticality"),
                "fulfilled_qty": fulfilled,
                "lost_sales_qty": shortage if not critical else 0,
                "status": "unresolved" if allocation["failed_checks"] else "firm",
                "failed_checks": "|".join(allocation["failed_checks"]),
                "inventory_conservation": stock
                == opening_stock + expected_receipts - fulfilled,
            }
        )
    orders = pd.DataFrame(order_rows)
    weekly = pd.DataFrame(weekly_rows)
    trigger_week: float = np.nan
    if not weekly.empty:
        below = weekly["below_reorder_point_flag"].astype(bool)
        if below.any():
            trigger_week = float(weekly.loc[below, "week"].iloc[0])
        weekly["reorder_trigger_week"] = trigger_week
    plan_status = "unresolved" if failed_checks else "firm"
    if not weekly.empty:
        weekly["status"] = plan_status
        weekly["failed_checks"] = "|".join(sorted(failed_checks))
    if not orders.empty:
        orders["status"] = plan_status
        orders["failed_checks"] = "|".join(sorted(failed_checks))
    return {
        "run_id": run_id,
        "part_id": part.get("part_id"),
        "orders": orders,
        "weekly": weekly,
        "safety_stock": safety_stock_max,
        "lead_time_days": lead_days,
        "status": plan_status,
        "failed_checks": sorted(failed_checks),
        "sob_target_deviation_reported": bool(order_rows),
        "capacity_used": capacity_state,
        "reorder_point": reorder_point,
        "reorder_trigger_week": trigger_week,
        "supplier_risk_score": supplier_risk_score,
        "risk_adjusted": bool(risk_adjusted),
    }
