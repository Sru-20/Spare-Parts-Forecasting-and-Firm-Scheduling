"""Weekly inventory simulation using the SPEC v4.1 event order."""
from __future__ import annotations

from math import ceil
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_STOCKOUT_MULTIPLIERS = {"High": 10.0, "Medium": 5.0, "Low": 2.0}


def realize_supplier_orders(
    orders: pd.DataFrame,
    supplier_snapshots: Mapping[str, Mapping[str, Any]],
    seed: int = 42,
    rejection_variant: str = "whole_lot",
) -> pd.DataFrame:
    """Apply seeded OTD-delay and rejection events to expected supplier orders."""
    if orders.empty:
        return orders.copy()
    if rejection_variant not in {"whole_lot", "per_unit_binomial"}:
        raise ValueError(f"Unsupported rejection variant: {rejection_variant}")
    rng = np.random.default_rng(seed)
    realized = orders.copy().reset_index(drop=True)
    arrived, received, rejected, rejected_qty, delay_weeks = [], [], [], [], []
    for row in realized.to_dict(orient="records"):
        supplier_id = str(row["supplier_id"])
        if supplier_id not in supplier_snapshots:
            raise ValueError(f"Missing supplier snapshot for {supplier_id}")
        supplier = supplier_snapshots[supplier_id]
        slot = int(row.get("supplier_slot", 1))
        prefix = f"sup{slot}_"
        values = [
            supplier.get(prefix + field)
            for field in ("otd_pct", "avg_delay_days", "reject_pct")
        ]
        if any(value is None or pd.isna(value) for value in values):
            raise ValueError(
                f"Missing delivery or rejection inputs for supplier {supplier_id}"
            )
        on_time, delay_days, reject_pct = map(float, values)
        if (
            not np.isfinite(on_time)
            or not np.isfinite(delay_days)
            or not np.isfinite(reject_pct)
            or not 0 <= on_time <= 100
            or delay_days < 0
            or not 0 <= reject_pct <= 100
        ):
            raise ValueError(
                f"Invalid delivery or rejection inputs for supplier {supplier_id}"
            )
        quantity = int(row["order_qty"])
        if rejection_variant == "per_unit_binomial":
            received_qty = int(
                rng.binomial(quantity, 1.0 - reject_pct / 100.0)
            )
            rejected_quantity = quantity - received_qty
            was_rejected = rejected_quantity > 0
        else:
            was_rejected = rng.random() < reject_pct / 100.0
            received_qty = 0 if was_rejected else quantity
            rejected_quantity = quantity if was_rejected else 0
        was_late = rng.random() >= on_time / 100.0
        extra_weeks = max(1, ceil(delay_days / 7.0)) if was_late and delay_days else 0
        expected = int(row["arrival_week"])
        arrived.append(expected + extra_weeks)
        rejected.append(bool(was_rejected))
        received.append(received_qty)
        rejected_qty.append(rejected_quantity)
        delay_weeks.append(extra_weeks)
    realized["realized_arrival_week"] = arrived
    realized["received_qty"] = received
    realized["rejected"] = rejected
    realized["rejected_qty"] = rejected_qty
    realized["rejection_variant"] = rejection_variant
    realized["delay_weeks"] = delay_weeks
    return realized


def simulate_weekly(
    initial_stock: int,
    demand: Sequence[float],
    receipts: Sequence[Mapping[str, float | int]],
    criticality: str,
    initial_backlog: int = 0,
    unit_cost: float = 0.0,
    carrying_cost_rate: float = 0.25,
    emergency_multiplier: float = 1.5,
    stockout_multipliers: Mapping[str, float] | None = None,
    emergency_rule: bool = True,
) -> pd.DataFrame:
    """Simulate a demand path; receipts arrive at the start of their due week."""
    if initial_stock < 0 or initial_backlog < 0:
        raise ValueError("Initial stock and backlog cannot be negative")
    if any(not np.isfinite(float(value)) or float(value) < 0 for value in demand):
        raise ValueError("Demand must contain only finite, non-negative values")
    critical = criticality == "High"
    opening_backlog = int(initial_backlog) if critical else 0
    backlog_writeoff = int(initial_backlog) if not critical else 0
    receipts_by_week: dict[int, int] = {}
    for receipt in receipts:
        week = int(receipt["week"])
        quantity = int(receipt["quantity"])
        if week < 1 or quantity < 0:
            raise ValueError("Receipt week and quantity are invalid")
        receipts_by_week[week] = receipts_by_week.get(week, 0) + quantity
    stock = int(initial_stock)
    backlog = opening_backlog
    future_emergency: dict[int, int] = {}
    multipliers = dict(DEFAULT_STOCKOUT_MULTIPLIERS)
    if stockout_multipliers:
        multipliers.update(stockout_multipliers)
    rows = []
    for week, demand_qty in enumerate(demand, start=1):
        opening = stock
        receipt_qty = receipts_by_week.get(week, 0) + future_emergency.get(week, 0)
        opening_after_receipt = opening + receipt_qty
        available = opening_after_receipt
        old_backlog = backlog
        backlog_fulfilled = min(available, backlog) if critical else 0
        available -= backlog_fulfilled
        backlog_remaining = backlog - backlog_fulfilled if critical else 0
        current_fulfilled = min(available, int(ceil(float(demand_qty))))
        lost = int(ceil(float(demand_qty))) - current_fulfilled
        available -= current_fulfilled
        backlog = backlog_remaining + lost if critical else 0
        fulfilled = backlog_fulfilled + current_fulfilled
        closing = opening_after_receipt - fulfilled
        if closing < 0:
            raise RuntimeError("Inventory simulation produced negative closing stock")
        stock = closing
        emergency_qty = backlog if critical and emergency_rule else 0
        if emergency_qty:
            future_emergency[week + 1] = (
                future_emergency.get(week + 1, 0) + emergency_qty
            )
        stockout_at_close = backlog if critical else lost
        rows.append(
            {
                "week": week,
                "opening_stock": opening,
                "receipts_qty": receipt_qty,
                "demand_qty": float(demand_qty),
                "opening_backlog_qty": old_backlog,
                "backlog_fulfilled_qty": backlog_fulfilled,
                "current_demand_fulfilled_qty": current_fulfilled,
                "fulfilled_qty": fulfilled,
                "stockout_qty": stockout_at_close,
                "stockout_at_close_qty": stockout_at_close,
                "stockout_at_close_flag": stockout_at_close > 0,
                "unmet_current_demand_qty": lost,
                "lost_sales_qty": lost if not critical else 0,
                "backlog_qty": backlog,
                "closing_stock": closing,
                "emergency_order_qty": emergency_qty,
                "emergency_cost_proxy": emergency_qty
                * float(unit_cost)
                * float(emergency_multiplier),
                "stockout_cost_proxy": lost
                * float(unit_cost)
                * float(multipliers.get(criticality, 0.0)),
                "carrying_cost_proxy": closing
                * float(unit_cost)
                * float(carrying_cost_rate)
                / 52.0,
                "initial_backlog_writeoff_qty": backlog_writeoff if week == 1 else 0,
                "inventory_conservation": closing
                == opening + receipt_qty - fulfilled,
            }
        )
    return pd.DataFrame(rows)
