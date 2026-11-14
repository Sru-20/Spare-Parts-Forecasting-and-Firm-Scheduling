"""Independent hard-constraint checker for generated replenishment schedules."""
from __future__ import annotations

from collections import defaultdict
from math import ceil
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _part_supplier(part: Mapping[str, Any], supplier_id: str) -> dict[str, Any] | None:
    for slot in (1, 2, 3):
        if str(part.get(f"sup{slot}_id") or "") == supplier_id:
            return {
                "type": part.get(f"sup{slot}_type"),
                "lead": part.get(f"sup{slot}_lead_time_days"),
                "moq": part.get(f"sup{slot}_moq"),
                "multiple": part.get(f"sup{slot}_order_multiple"),
                "capacity": part.get(f"sup{slot}_monthly_capacity"),
                "committed": part.get(f"sup{slot}_capacity_committed_pct"),
            }
    return None


def check_schedule(
    schedule: pd.DataFrame,
    parts: Mapping[str, Mapping[str, Any]],
    horizon_weeks: int = 14,
    check_requirements: bool = True,
    requirements: pd.DataFrame | None = None,
) -> dict[str, Any]:
    failures: set[str] = set()
    capacity_totals: dict[tuple[Any, str, Any], float] = defaultdict(float)
    capacity_limits: dict[tuple[Any, str, Any], float] = {}
    if requirements is not None:
        requirement_fields = {"part_id", "release_week", "net_requirement"}
        missing_requirements = requirement_fields - set(requirements.columns)
        if missing_requirements:
            return {
                "feasible": False,
                "failed_checks": [
                    f"missing_requirement_fields:{','.join(sorted(missing_requirements))}"
                ],
                "checked_orders": 0,
                "status": "unresolved",
            }
        requirement_qty = pd.to_numeric(
            requirements["net_requirement"], errors="coerce"
        )
        if (
            not np.isfinite(requirement_qty.to_numpy(dtype=float)).all()
            or requirement_qty.lt(0).any()
        ):
            return {
                "feasible": False,
                "failed_checks": ["invalid_requirement"],
                "checked_orders": 0,
                "status": "unresolved",
            }
    if schedule.empty:
        if (
            requirements is not None
            and not requirements.empty
            and requirement_qty.gt(0).any()
        ):
            return {
                "feasible": False,
                "failed_checks": ["unfilled_requirement"],
                "checked_orders": 0,
                "status": "unresolved",
            }
        return {
            "feasible": True,
            "failed_checks": [],
            "checked_orders": 0,
            "status": "firm",
        }
    required = {
        "part_id", "supplier_id", "release_week", "arrival_week",
        "lead_time_days", "order_qty",
    }
    missing = required - set(schedule.columns)
    if missing:
        return {
            "feasible": False,
            "failed_checks": [f"missing_fields:{','.join(sorted(missing))}"],
            "checked_orders": 0,
            "status": "unresolved",
        }
    for order in schedule.to_dict(orient="records"):
        part_id = str(order["part_id"])
        supplier_id = str(order["supplier_id"])
        part = parts.get(part_id)
        if part is None:
            failures.add("unknown_part")
            continue
        if str(part.get("lifecycle_status", "")).casefold() == "obsolete":
            failures.add("obsolete_part_order")
        supplier = _part_supplier(part, supplier_id)
        if supplier is None:
            failures.add("unknown_supplier")
            continue
        if (
            str(part.get("supply_mode", "")).casefold() == "production"
            and str(supplier["type"]).casefold() == "import"
        ):
            failures.add("production_import_supplier")
        qty = float(order["order_qty"])
        if not np.isfinite(qty) or qty < 0 or not qty.is_integer():
            failures.add("non_integer_or_negative_quantity")
            continue
        moq = float(supplier["moq"])
        multiple = float(supplier["multiple"])
        if qty > 0 and qty < moq:
            failures.add("moq")
        if multiple <= 0 or (qty > 0 and not np.isclose(qty % multiple, 0)):
            failures.add("order_multiple")
        release = int(order["release_week"])
        lead = float(supplier["lead"])
        expected_arrival = release + max(1, ceil(lead / 7.0))
        if release < 1 or int(order["arrival_week"]) != expected_arrival:
            failures.add("lead_time")
        if expected_arrival <= 0:
            failures.add("lead_time")
        if "closing_stock" in order and float(order["closing_stock"]) < 0:
            failures.add("negative_stock")
        if {
            "opening_stock", "receipts_qty", "fulfilled_qty", "closing_stock"
        } <= set(order):
            reconciled = (
                float(order["opening_stock"])
                + float(order["receipts_qty"])
                - float(order["fulfilled_qty"])
            )
            if not np.isclose(reconciled, float(order["closing_stock"])):
                failures.add("inventory_conservation")
        period = order.get("capacity_period", (expected_arrival - 1) // 4)
        simulation_id = order.get("origin_date", "single_schedule")
        key = (simulation_id, supplier_id, period)
        capacity_totals[key] += qty
        limit = (
            float(supplier["capacity"])
            * (1.0 - float(supplier["committed"]) / 100.0)
        )
        capacity_limits[key] = min(capacity_limits.get(key, limit), limit)
    for key, total in capacity_totals.items():
        if total > capacity_limits[key] + 1e-9:
            failures.add("supplier_capacity")
    if requirements is not None and check_requirements:
        key_fields = ["part_id"]
        if "run_id" in requirements and "run_id" in schedule:
            key_fields.append("run_id")
        key_fields.append("release_week")
        actual = schedule.copy()
        requirement_week = (
            actual["requested_week"]
            if "requested_week" in actual
            else actual["release_week"]
        )
        actual["release_week"] = requirement_week.astype(int)
        actual_totals = (
            actual.groupby(key_fields, dropna=False, observed=True)["order_qty"]
            .sum()
            .to_dict()
        )
        for requirement in requirements.to_dict(orient="records"):
            key = tuple(requirement[field] for field in key_fields)
            if float(actual_totals.get(key, 0)) + 1e-9 < float(
                requirement["net_requirement"]
            ):
                failures.add("unfilled_requirement")
                break
    elif check_requirements and "net_requirement" in schedule:
        order_groups = ["part_id"]
        if "run_id" in schedule:
            order_groups.append("run_id")
        if "release_week" in schedule:
            order_groups.append("release_week")
        for _, group in schedule.groupby(order_groups, dropna=False, observed=True):
            required_qty = float(group["net_requirement"].iloc[0])
            if float(group["order_qty"].sum()) + 1e-9 < required_qty:
                failures.add("unfilled_requirement")
    failed = sorted(failures)
    return {
        "feasible": not failed,
        "failed_checks": failed,
        "checked_orders": int(len(schedule)),
        "status": "firm" if not failed else "unresolved",
    }


def check_inventory_ledger(
    ledger: pd.DataFrame,
    opening_column: str,
    receipts_column: str,
    fulfilled_column: str,
    closing_column: str,
) -> dict[str, Any]:
    """Independently reconcile weekly inventory rows."""
    required = {
        opening_column,
        receipts_column,
        fulfilled_column,
        closing_column,
    }
    missing = required - set(ledger.columns)
    if missing:
        return {
            "feasible": False,
            "failed_checks": [f"missing_inventory_fields:{','.join(sorted(missing))}"],
            "checked_weeks": 0,
        }
    if ledger.empty:
        return {"feasible": True, "failed_checks": [], "checked_weeks": 0}
    values = ledger[
        [opening_column, receipts_column, fulfilled_column, closing_column]
    ].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        return {
            "feasible": False,
            "failed_checks": ["invalid_inventory_value"],
            "checked_weeks": int(len(ledger)),
        }
    reconciled = (
        values[opening_column]
        + values[receipts_column]
        - values[fulfilled_column]
    )
    failures = []
    if not np.isclose(
        reconciled.to_numpy(dtype=float),
        values[closing_column].to_numpy(dtype=float),
        atol=1e-9,
        rtol=0,
    ).all():
        failures.append("inventory_conservation")
    if values[closing_column].lt(0).any():
        failures.append("negative_stock")
    return {
        "feasible": not failures,
        "failed_checks": failures,
        "checked_weeks": int(len(ledger)),
    }
