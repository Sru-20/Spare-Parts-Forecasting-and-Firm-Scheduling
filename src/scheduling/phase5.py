"""Validation-only execution of the Phase 5 MRP and simulator core."""
from __future__ import annotations

import json
from collections import defaultdict
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.data.loader import load_scheduling
from src.scheduling.checker import check_inventory_ledger, check_schedule
from src.scheduling.engine import (
    allocate_supplier_orders,
    build_mrp_schedule,
    order_arrival_week,
    snapshot_receipts,
    supplier_options,
)
from src.simulation.weekly import realize_supplier_orders, simulate_weekly


ROOT = Path(__file__).resolve().parents[2]
HARD_CONSTRAINTS = {
    "moq",
    "order_multiple",
    "supplier_capacity",
    "lead_time",
    "negative_stock",
    "inventory_conservation",
    "obsolete_part_order",
    "production_import_supplier",
    "non_integer_or_negative_quantity",
    "unknown_part",
    "unknown_supplier",
}
FORECAST_KEYS = [
    "origin_date",
    "part_id",
    "region",
    "week_start_date",
    "horizon_step",
]


def load_validation_paths(root: Path = ROOT) -> pd.DataFrame:
    """Combine saved Track A predictions and observed held-validation labels."""
    source = root / "artifacts" / "track_a_backtest_forecasts_v41.parquet"
    if not source.is_file():
        raise FileNotFoundError(f"Missing saved validation forecast artifact: {source}")
    forecasts = pd.read_parquet(source)
    if forecasts["evaluation_split"].ne("evaluation").any():
        raise ValueError("Phase 5 accepts held-validation forecast rows only")
    dates = pd.to_datetime(forecasts["week_start_date"])
    if dates.max() > pd.Timestamp("2026-03-23"):
        raise ValueError("Phase 5 cannot read forecast targets after train-validation")
    if forecasts["origin_date"].nunique() != 4:
        raise ValueError("Expected the four saved held-validation origins")

    point = forecasts.loc[
        forecasts["model"].eq("lightgbm_tweedie"),
        FORECAST_KEYS + ["forecast_qty", "actual_qty"],
    ].copy()
    quantile = forecasts.loc[
        forecasts["model"].eq("lightgbm_quantile"),
        FORECAST_KEYS + ["forecast_p90", "forecast_p95", "forecast_p99"],
    ].copy()
    joined = point.merge(
        quantile,
        on=FORECAST_KEYS,
        how="inner",
        validate="one_to_one",
    )
    if joined.empty:
        raise ValueError("No aligned champion point and quantile forecasts found")
    rows = []
    for keys, group in joined.groupby(
        ["origin_date", "part_id", "week_start_date", "horizon_step"],
        sort=True,
        observed=True,
    ):
        labels = pd.to_numeric(group["actual_qty"], errors="coerce")
        complete_actual = bool(labels.notna().all())
        rows.append(
            {
                "origin_date": keys[0],
                "part_id": keys[1],
                "week_start_date": keys[2],
                "horizon_step": int(keys[3]),
                "week": int(keys[3]),
                "regions_included": int(group["region"].nunique()),
                "complete_actual": complete_actual,
                "actual_qty": float(labels.sum()) if complete_actual else np.nan,
                "forecast_qty": float(group["forecast_qty"].sum()),
                "forecast_p90": float(group["forecast_p90"].sum()),
                "forecast_p95": float(group["forecast_p95"].sum()),
                "forecast_p99": float(group["forecast_p99"].sum()),
            }
        )
    paths = pd.DataFrame(rows)
    complete_keys = (
        paths.groupby(["origin_date", "part_id"], observed=True)
        .agg(week_count=("week", "nunique"), complete=("complete_actual", "all"))
        .reset_index()
    )
    complete_keys = complete_keys.loc[
        complete_keys["week_count"].eq(14) & complete_keys["complete"]
    ]
    paths = paths.merge(
        complete_keys[["origin_date", "part_id"]],
        on=["origin_date", "part_id"],
        how="inner",
        validate="many_to_one",
    )
    if paths.empty:
        raise ValueError("No complete 14-week part-origin actual demand paths")
    if paths["actual_qty"].isna().any():
        raise RuntimeError("Incomplete actual labels passed the path-completeness gate")
    return paths.sort_values(
        ["origin_date", "part_id", "week"], kind="stable"
    ).reset_index(drop=True)


def _supplier_snapshot(part: dict[str, Any]) -> dict[str, dict[str, Any]]:
    suppliers: dict[str, dict[str, Any]] = {}
    for slot in range(1, 4):
        supplier_id = part.get(f"sup{slot}_id")
        if supplier_id is None or pd.isna(supplier_id):
            continue
        suppliers[str(supplier_id)] = {
            f"sup{slot}_otd_pct": part.get(f"sup{slot}_otd_pct"),
            f"sup{slot}_avg_delay_days": part.get(
                f"sup{slot}_avg_delay_days"
            ),
            f"sup{slot}_reject_pct": part.get(f"sup{slot}_reject_pct"),
        }
    return suppliers


def _scaled_snapshot(
    snapshot: dict[str, Any], multiplier: float
) -> dict[str, Any]:
    scaled = dict(snapshot)
    usable = int(np.floor(float(snapshot["available_qty"]) * multiplier))
    scaled["available_qty"] = usable
    scaled["on_hand_qty"] = (
        usable + int(snapshot["reserved_qty"]) + int(snapshot["blocked_qty"])
    )
    return scaled


def _generate_origin_plans(
    origin: Any,
    origin_paths: pd.DataFrame,
    parts: dict[str, dict[str, Any]],
    capacity_seed: dict[Any, float],
    preplanned_by_part: dict[str, list[dict[str, Any]]],
    initial_stock_multiplier: float,
    due_later_week: int,
) -> tuple[list[dict[str, Any]], dict[Any, float]]:
    plans = []
    shared_capacity = dict(capacity_seed)
    for part_id, forecast_group in origin_paths.groupby(
        "part_id", sort=True, observed=True
    ):
        raw_snapshot = parts.get(str(part_id))
        if raw_snapshot is None:
            continue
        snapshot = _scaled_snapshot(raw_snapshot, initial_stock_multiplier)
        run_id = f"phase5:{pd.Timestamp(origin).date().isoformat()}:{part_id}"
        plan = build_mrp_schedule(
            snapshot,
            forecast_group,
            run_id=run_id,
            horizon_weeks=14,
            capacity_used=shared_capacity,
            due_later_week=due_later_week,
            initial_stock_multiplier=1.0,
            preplanned_orders=preplanned_by_part.get(str(part_id), []),
        )
        shared_capacity = plan.get("capacity_used", shared_capacity)
        orders = plan["orders"].copy()
        if not orders.empty:
            orders["origin_date"] = origin
            orders["region"] = "ALL"
            orders["forecast_model"] = (
                "lightgbm_tweedie_point+lightgbm_quantile_ss"
            )
        plans.append(
            {
                "run_id": run_id,
                "part_id": str(part_id),
                "snapshot": snapshot,
                "forecast_group": forecast_group,
                "plan": plan,
                "orders": orders,
            }
        )
    return plans, shared_capacity


def _propose_pull_ahead_orders(
    origin: Any,
    plans: list[dict[str, Any]],
    baseline_capacity: dict[Any, float],
) -> dict[str, list[dict[str, Any]]]:
    """Use unused earlier calendar-month capacity for a future requirement once."""
    capacity_state = dict(baseline_capacity)
    candidates = []
    for item in plans:
        plan = item["plan"]
        if "supplier_capacity" not in plan["failed_checks"]:
            continue
        weekly = plan["weekly"]
        existing = item["orders"]
        for row in weekly.to_dict(orient="records"):
            week = int(row["week"])
            ordered = 0
            if not existing.empty:
                order_week = existing["release_week"].astype(int).eq(week)
                ordered = int(existing.loc[order_week, "order_qty"].sum())
            shortfall = max(0, int(ceil(float(row["net_requirement"]) - ordered)))
            if shortfall:
                candidates.append(
                    {
                        "item": item,
                        "week": week,
                        "week_start_date": pd.Timestamp(row["week_start_date"]),
                        "shortfall": shortfall,
                        "priority": {
                            "High": 0,
                            "Medium": 1,
                            "Low": 2,
                        }.get(str(row["criticality"]), 3),
                    }
                )
    candidates.sort(
        key=lambda row: (
            row["priority"],
            row["week"],
            row["item"]["part_id"],
        )
    )
    pulled: dict[str, list[dict[str, Any]]] = defaultdict(list)
    origin_week_start = min(
        pd.to_datetime(item["forecast_group"]["week_start_date"]).min()
        for item in plans
    )
    for candidate in candidates:
        item = candidate["item"]
        snapshot = item["snapshot"]
        options = supplier_options(snapshot)
        remaining = candidate["shortfall"]
        for release_week in range(candidate["week"] - 1, 0, -1):
            eligible_slots = set()
            for option in options:
                target_arrival = order_arrival_week(
                    candidate["week"], option["lead_time_days"]
                )
                target_month = (
                    candidate["week_start_date"]
                    + pd.Timedelta(days=7 * (target_arrival - candidate["week"]))
                ).strftime("%Y-%m")
                early_arrival = order_arrival_week(
                    release_week, option["lead_time_days"]
                )
                early_month = (
                    origin_week_start
                    + pd.Timedelta(days=7 * (early_arrival - 1))
                ).strftime("%Y-%m")
                if early_month < target_month:
                    eligible_slots.add(int(option["slot"]))
            if not eligible_slots:
                continue
            eligible_part = dict(snapshot)
            for slot in (1, 2, 3):
                if slot not in eligible_slots:
                    for field in (
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
                    ):
                        eligible_part[f"sup{slot}_{field}"] = None
            allocation = allocate_supplier_orders(
                eligible_part,
                remaining,
                release_week=release_week,
                capacity_used=capacity_state,
                horizon_weeks=14,
                capacity_period=origin_week_start,
            )
            capacity_state = allocation.get("capacity_used", capacity_state)
            allocated = 0
            for order in allocation["orders"]:
                allocated += int(order["order_qty"])
                order.update(
                    {
                        "run_id": item["run_id"],
                        "origin_date": origin,
                        "region": "ALL",
                        "planned_for_week": candidate["week"],
                        "requested_qty": candidate["shortfall"],
                        "forecast_model": (
                            "lightgbm_tweedie_point+lightgbm_quantile_ss"
                        ),
                        "pulled_ahead": True,
                    }
                )
                pulled[item["part_id"]].append(order)
            remaining = max(0, remaining - allocated)
            if remaining == 0:
                break
    return dict(pulled)


def _capacity_reservations(
    pulled_by_part: dict[str, list[dict[str, Any]]]
) -> dict[tuple[str, Any], float]:
    reservations: dict[tuple[str, Any], float] = defaultdict(float)
    for orders in pulled_by_part.values():
        for order in orders:
            key = (str(order["supplier_id"]), order["capacity_period"])
            reservations[key] += int(order["order_qty"])
    return dict(reservations)


def _collect_plan_shortfalls(
    plans: list[dict[str, Any]], stage: str
) -> list[dict[str, Any]]:
    rows = []
    for item in plans:
        plan = item["plan"]
        orders = item["orders"]
        for weekly in plan["weekly"].to_dict(orient="records"):
            requested_week = int(weekly["week"])
            if orders.empty:
                relevant_orders = pd.DataFrame()
                ordered = 0
            else:
                relevant_orders = orders.loc[
                    orders["requested_week"].astype(int).eq(requested_week)
                ]
                ordered = int(relevant_orders["order_qty"].sum())
            shortfall = max(
                0,
                int(ceil(float(weekly["net_requirement"]) - ordered)),
            )
            if not shortfall:
                continue
            slot_names = (
                sorted(
                    {
                        str(int(slot))
                        for slot in relevant_orders["supplier_slot"].dropna()
                    }
                )
                if not relevant_orders.empty
                else []
            )
            rows.append(
                {
                    "stage": stage,
                    "run_id": item["run_id"],
                    "part_id": item["part_id"],
                    "criticality": weekly["criticality"],
                    "requirement_week": requested_week,
                    "week_start_date": weekly["week_start_date"],
                    "requirement_month": pd.Timestamp(
                        weekly["week_start_date"]
                    ).strftime("%Y-%m"),
                    "net_requirement": weekly["net_requirement"],
                    "ordered_qty": ordered,
                    "shortfall_qty": shortfall,
                    "shortfall_band": (
                        "lt_100"
                        if shortfall < 100
                        else "100_999"
                        if shortfall < 1000
                        else "1000_9999"
                        if shortfall < 10000
                        else "gte_10000"
                    ),
                    "allocated_supplier_slots": ",".join(slot_names)
                    if slot_names
                    else "none",
                    "plan_failure_cause": "|".join(plan["failed_checks"]),
                }
            )
    return rows


def _lead_time_coverage(
    paths: pd.DataFrame, parts: dict[str, dict[str, Any]]
) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    over_99 = sorted(
        part_id
        for part_id, part in parts.items()
        if float(part["service_level_target"]) > 0.99
    )
    for (origin, part_id), group in paths.groupby(
        ["origin_date", "part_id"], sort=True, observed=True
    ):
        if str(part_id) not in parts:
            continue
        part = parts[str(part_id)]
        options = supplier_options(part)
        total_share = sum(row["sob_pct"] for row in options)
        if not options or total_share <= 0:
            continue
        lead_days = sum(
            row["sob_pct"] * row["lead_time_days"] for row in options
        ) / total_share
        lead_weeks = max(1, int(ceil(lead_days / 7.0)))
        target = float(part["service_level_target"])
        quantile_column = next(
            (
                col
                for col, level in (
                    ("forecast_p90", 0.90),
                    ("forecast_p95", 0.95),
                    ("forecast_p99", 0.99),
                )
                if level >= target
            ),
            "forecast_p99",
        )
        point = group["forecast_qty"].astype(float).to_numpy()
        quantile = group[quantile_column].astype(float).to_numpy()
        actual = group["actual_qty"].astype(float).to_numpy()
        bounded = min(lead_weeks, len(group))
        point_demand = float(point[:bounded].sum())
        quantile_demand = float(quantile[:bounded].sum())
        actual_demand = float(actual[:bounded].sum())
        extension = max(0, lead_weeks - bounded)
        if extension:
            point_demand += float(point[-4:].mean()) * extension
            quantile_demand += float(quantile[-4:].mean()) * extension
            actual_demand += float(actual[-4:].mean()) * extension
        safety_stock = max(0.0, quantile_demand - point_demand)
        rows.append(
            {
                "origin_date": origin,
                "part_id": str(part_id),
                "criticality": part["criticality"],
                "service_level_target": target,
                "lead_time_days": lead_days,
                "lead_time_weeks": lead_weeks,
                "selected_quantile": quantile_column,
                "point_lead_time_demand": point_demand,
                "safety_stock": safety_stock,
                "actual_lead_time_demand": actual_demand,
                "covered": actual_demand <= point_demand + safety_stock,
                "regions": int(group["regions_included"].max()),
            }
        )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("criticality", observed=True)
        .agg(
            part_origins=("covered", "size"),
            coverage_rate=("covered", "mean"),
            mean_target=("service_level_target", "mean"),
            min_target=("service_level_target", "min"),
            max_target=("service_level_target", "max"),
        )
        .reset_index()
    )
    return summary, over_99


def _path_population_audit(
    root: Path, paths: pd.DataFrame, scheduling: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    source = root / "artifacts" / "track_a_backtest_forecasts_v41.parquet"
    forecasts = pd.read_parquet(source)
    point = forecasts.loc[
        forecasts["model"].eq("lightgbm_tweedie"),
        FORECAST_KEYS + ["actual_qty"],
    ]
    quantile = forecasts.loc[
        forecasts["model"].eq("lightgbm_quantile"), FORECAST_KEYS
    ]
    regional = point.merge(
        quantile, on=FORECAST_KEYS, how="inner", validate="one_to_one"
    )
    weekly = (
        regional.groupby(
            ["origin_date", "part_id", "horizon_step"], observed=True
        )
        .agg(
            regions=("region", "nunique"),
            complete_actual=("actual_qty", lambda value: value.notna().all()),
        )
        .reset_index()
    )
    catalog = pd.read_csv(
        root / "data" / "03_spare_parts_catalogue.csv",
        usecols=[
            "part_id",
            "criticality",
            "lifecycle_status",
            "abc_class",
            "part_type",
        ],
    )
    origins = sorted(pd.to_datetime(regional["origin_date"]).unique())
    universe = pd.MultiIndex.from_product(
        [origins, catalog["part_id"].astype(str).unique()],
        names=["origin_date", "part_id"],
    ).to_frame(index=False)
    status = (
        weekly.groupby(["origin_date", "part_id"], observed=True)
        .agg(
            horizon_weeks=("horizon_step", "nunique"),
            incomplete_actual_weeks=(
                "complete_actual",
                lambda values: int(values.eq(False).sum()),
            ),
        )
        .reset_index()
    )
    universe = universe.merge(
        status, on=["origin_date", "part_id"], how="left", validate="one_to_one"
    )
    universe["exclusion_reason"] = np.select(
        [
            universe["horizon_weeks"].isna(),
            universe["horizon_weeks"].notna()
            & universe["horizon_weeks"].ne(14),
            universe["horizon_weeks"].eq(14)
            & universe["incomplete_actual_weeks"].gt(0),
        ],
        [
            "no_forecast_rows_for_part_origin",
            "incomplete_14_week_forecast_horizon",
            "missing_observed_regional_label",
        ],
        default="complete",
    )
    segments = pd.read_parquet(root / "artifacts" / "series_segments.parquet")
    part_segments = (
        segments.groupby("part_id", observed=True)
        .agg(
            sbc_class=(
                "sbc_class",
                lambda values: (
                    values.mode().dropna().iloc[0]
                    if not values.mode().dropna().empty
                    else "Unclassified"
                ),
            ),
        )
        .reset_index()
    )
    universe = universe.merge(
        part_segments, on="part_id", how="left", validate="many_to_one"
    )
    universe["sbc_class"] = universe["sbc_class"].fillna("No segment artifact")
    universe = universe.merge(catalog, on="part_id", how="left", validate="many_to_one")
    reasons = universe["exclusion_reason"].value_counts().to_dict()
    universe["excluded"] = universe["exclusion_reason"].ne("complete")
    criticality_rates = (
        universe.groupby("criticality", observed=True)
        .agg(
            possible_paths=("excluded", "size"),
            excluded_paths=("excluded", "sum"),
            exclusion_rate=("excluded", "mean"),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    segment_rates = (
        universe.groupby("sbc_class", observed=True)
        .agg(
            possible_paths=("excluded", "size"),
            excluded_paths=("excluded", "sum"),
            exclusion_rate=("excluded", "mean"),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    joined = paths.groupby(["origin_date", "part_id"], observed=True).agg(
        actual_14_week_demand=("actual_qty", "sum")
    ).reset_index()
    initial_cover = joined.merge(
        scheduling[
            [
                "part_id",
                "criticality",
                "available_qty",
                "backorder_qty",
            ]
        ],
        on="part_id",
        how="inner",
        validate="many_to_one",
    )
    initial_cover["initial_cover_weeks"] = np.where(
        initial_cover["actual_14_week_demand"].gt(0),
        14.0
        * initial_cover["available_qty"]
        / initial_cover["actual_14_week_demand"],
        np.inf,
    )
    initial_cover["basis"] = "simulated_synthetic_initialization"
    return (
        universe,
        initial_cover,
        {
            "reason_counts": reasons,
            "exclusion_rate_by_criticality": criticality_rates,
            "exclusion_rate_by_segment": segment_rates,
        },
    )


def run_phase5_core(
    root: Path = ROOT,
    *,
    initial_stock_multiplier: float | None = None,
    due_later_week: int | None = None,
    rejection_variant: str | None = None,
    use_pull_ahead: bool | None = None,
    write_outputs: bool = True,
) -> dict[str, Any]:
    """Run the Phase 5 MRP and simulator on saved held-validation paths."""
    config = yaml.safe_load((root / "configs" / "project.yaml").read_text(encoding="utf-8"))
    business = yaml.safe_load((root / "configs" / "business.yaml").read_text(encoding="utf-8"))
    assumptions = config["scheduling"]["simulator_assumptions"]
    initial_stock_multiplier = (
        float(assumptions["initial_inventory"]["initial_stock_multiplier"])
        if initial_stock_multiplier is None
        else float(initial_stock_multiplier)
    )
    due_later_week = (
        int(assumptions["inbound_receipts"]["due_later_week"])
        if due_later_week is None
        else int(due_later_week)
    )
    rejection_variant = (
        str(assumptions["supplier_uncertainty"]["default_rejection_variant"])
        if rejection_variant is None
        else str(rejection_variant)
    )
    use_pull_ahead = (
        bool(assumptions["capacity"]["pull_ahead"]["enabled"])
        if use_pull_ahead is None
        else bool(use_pull_ahead)
    )
    if initial_stock_multiplier < 0 or due_later_week < 1:
        raise ValueError("Scenario inventory multiplier or receipt week is invalid")
    if rejection_variant not in {"whole_lot", "per_unit_binomial"}:
        raise ValueError(f"Unsupported rejection variant: {rejection_variant}")
    seed = int(assumptions["seed"])
    scheduling = load_scheduling(root / "data" / "02_scheduling_inputs.csv")
    parts = {
        str(row["part_id"]): row
        for row in scheduling.to_dict(orient="records")
    }
    paths = load_validation_paths(root)
    stockout_costs = business["scheduling"]["costs"]["stockout_penalty"][
        "criticality_multipliers"
    ]
    emergency_multiplier = float(
        business["scheduling"]["costs"]["emergency_procurement"][
            "unit_cost_multiplier"
        ]
    )
    weekly_rows: list[pd.DataFrame] = []
    plan_weekly_rows: list[pd.DataFrame] = []
    order_rows: list[pd.DataFrame] = []
    checker_rows: list[dict[str, Any]] = []
    plan_items: list[dict[str, Any]] = []
    baseline_unresolved = 0
    shortfall_stage_rows: list[dict[str, Any]] = []
    pull_ahead_count = 0
    pull_ahead_qty = 0
    run_count = 0
    for origin, origin_paths in paths.groupby(
        "origin_date", sort=True, observed=True
    ):
        baseline_plans, baseline_capacity = _generate_origin_plans(
            origin,
            origin_paths,
            parts,
            {},
            {},
            initial_stock_multiplier,
            due_later_week,
        )
        baseline_unresolved += sum(
            item["plan"]["status"] != "firm" for item in baseline_plans
        )
        shortfall_stage_rows.extend(
            _collect_plan_shortfalls(baseline_plans, "before_pull_ahead")
        )
        pulled_by_part = (
            _propose_pull_ahead_orders(
                origin, baseline_plans, baseline_capacity
            )
            if use_pull_ahead
            else {}
        )
        pull_ahead_count += sum(len(rows) for rows in pulled_by_part.values())
        pull_ahead_qty += sum(
            int(order["order_qty"])
            for rows in pulled_by_part.values()
            for order in rows
        )
        final_plans, _ = _generate_origin_plans(
            origin,
            origin_paths,
            parts,
            _capacity_reservations(pulled_by_part),
            pulled_by_part,
            initial_stock_multiplier,
            due_later_week,
        )
        run_count += len(final_plans)
        plans = final_plans
        shortfall_stage_rows.extend(
            _collect_plan_shortfalls(plans, "after_pull_ahead")
        )
        for item in plans:
            if not item["plan"]["weekly"].empty:
                plan_weekly = item["plan"]["weekly"].copy()
                plan_weekly["origin_date"] = origin
                plan_weekly["plan_status"] = item["plan"]["status"]
                plan_weekly["plan_failed_checks"] = "|".join(
                    item["plan"]["failed_checks"]
                )
                plan_weekly_rows.append(plan_weekly)
        origin_orders = (
            pd.concat(
                [item["orders"] for item in plans if not item["orders"].empty],
                ignore_index=True,
            )
            if any(not item["orders"].empty for item in plans)
            else pd.DataFrame()
        )
        origin_check = check_schedule(
            origin_orders,
            parts,
            horizon_weeks=14,
            check_requirements=False,
        )
        for item in plans:
            part_id = str(item["part_id"])
            part_orders = item["orders"]
            requirements = item["plan"]["weekly"][
                ["run_id", "part_id", "week", "net_requirement"]
            ].rename(columns={"week": "release_week"})
            part_check = check_schedule(
                part_orders,
                {part_id: item["snapshot"]},
                horizon_weeks=14,
                requirements=requirements,
            )
            failures = sorted(
                set(item["plan"]["failed_checks"])
                | set(part_check["failed_checks"])
                | (
                    set(origin_check["failed_checks"])
                    if not origin_check["feasible"]
                    else set()
                )
            )
            final_status = (
                "firm"
                if item["plan"]["status"] == "firm" and not failures
                else "unresolved"
            )
            checker_rows.append(
                {
                    "run_id": item["run_id"],
                    "part_id": part_id,
                    "origin_date": origin,
                    "plan_status": item["plan"]["status"],
                    "checker_status": (
                        "firm" if part_check["feasible"] and origin_check["feasible"]
                        else "unresolved"
                    ),
                    "plan_failed_checks": "|".join(
                        item["plan"]["failed_checks"]
                    ),
                    "independent_checker_failed_checks": "|".join(
                        sorted(
                            set(part_check["failed_checks"])
                            | set(origin_check["failed_checks"])
                        )
                    ),
                    "feasible": final_status == "firm",
                    "failed_checks": "|".join(failures),
                }
            )
            item["final_status"] = final_status
            plan_items.append(item)
            if not part_orders.empty:
                part_orders["status"] = final_status
                part_orders["plan_status"] = item["plan"]["status"]
                part_orders["independent_checker_status"] = (
                    "firm" if part_check["feasible"] and origin_check["feasible"]
                    else "unresolved"
                )
                part_orders["independent_checker_failed_checks"] = "|".join(
                    sorted(
                        set(part_check["failed_checks"])
                        | set(origin_check["failed_checks"])
                    )
                )
                order_rows.append(part_orders)

            snapshot = item["snapshot"]
            forecast_group = item["forecast_group"]
            realized = realize_supplier_orders(
                part_orders,
                _supplier_snapshot(snapshot),
                seed=seed,
                rejection_variant=rejection_variant,
            )
            receipts = snapshot_receipts(snapshot, due_later_week)
            for receipt in realized.to_dict(orient="records"):
                if int(receipt["realized_arrival_week"]) <= 14:
                    receipts.append(
                        {
                            "week": int(receipt["realized_arrival_week"]),
                            "quantity": int(receipt["received_qty"]),
                        }
                    )
            carrying_rate = snapshot.get("carrying_cost_rate")
            if carrying_rate is None or pd.isna(carrying_rate):
                carrying_rate = assumptions["carrying_cost"]["fallback_annual_rate"]
            simulated = simulate_weekly(
                initial_stock=int(snapshot["available_qty"]),
                demand=forecast_group["actual_qty"].astype(float).tolist(),
                receipts=receipts,
                criticality=str(snapshot["criticality"]),
                initial_backlog=int(snapshot["backorder_qty"] or 0),
                unit_cost=float(snapshot["unit_cost"]),
                carrying_cost_rate=float(carrying_rate),
                emergency_multiplier=emergency_multiplier,
                stockout_multipliers={
                    key: float(value) for key, value in stockout_costs.items()
                },
                emergency_rule=True,
            )
            simulated["run_id"] = item["run_id"]
            simulated["origin_date"] = origin
            simulated["part_id"] = part_id
            simulated["region"] = "ALL"
            simulated["basis"] = "simulated"
            weekly_rows.append(simulated)

    orders_out = (
        pd.concat(order_rows, ignore_index=True)
        if order_rows
        else pd.DataFrame()
    )
    simulation_out = (
        pd.concat(weekly_rows, ignore_index=True)
        if weekly_rows
        else pd.DataFrame()
    )
    checks_out = pd.DataFrame(checker_rows)
    plan_weekly_out = (
        pd.concat(plan_weekly_rows, ignore_index=True)
        if plan_weekly_rows
        else pd.DataFrame()
    )
    if not simulation_out.empty and not plan_weekly_out.empty:
        simulation_out = simulation_out.merge(
            plan_weekly_out[
                ["run_id", "week", "safety_stock"]
            ].drop_duplicates(["run_id", "week"]),
            on=["run_id", "week"],
            how="left",
            validate="one_to_one",
        )
        simulation_out["excess_inventory_units"] = (
            simulation_out["closing_stock"] - simulation_out["safety_stock"]
        ).clip(lower=0)
    mrp_ledger_failures = 0
    simulator_ledger_failures = 0
    for item in plan_items:
        run_id = item["run_id"]
        mrp_check = check_inventory_ledger(
            item["plan"]["weekly"],
            "opening_usable_stock",
            "expected_receipts",
            "fulfilled_qty",
            "projected_inventory",
        )
        sim_ledger = simulation_out.loc[simulation_out["run_id"].eq(run_id)]
        sim_check = check_inventory_ledger(
            sim_ledger,
            "opening_stock",
            "receipts_qty",
            "fulfilled_qty",
            "closing_stock",
        )
        mrp_ledger_failures += int(not mrp_check["feasible"])
        simulator_ledger_failures += int(not sim_check["feasible"])
        ledger_failures = sorted(
            set(mrp_check["failed_checks"]) | set(sim_check["failed_checks"])
        )
        if ledger_failures:
            index = checks_out.index[checks_out["run_id"].eq(run_id)]
            for column in (
                "independent_checker_failed_checks",
                "failed_checks",
            ):
                existing = str(checks_out.loc[index[0], column] or "")
                checks_out.loc[index, column] = "|".join(
                    sorted(
                        set(filter(None, existing.split("|")))
                        | set(ledger_failures)
                    )
                )
            checks_out.loc[index, "checker_status"] = "unresolved"
            checks_out.loc[index, "feasible"] = False
    if not orders_out.empty:
        statuses = checks_out.set_index("run_id")["feasible"].map(
            {True: "firm", False: "unresolved"}
        )
        orders_out["status"] = orders_out["run_id"].map(statuses)
    emergency_orders = simulation_out.loc[
        simulation_out["emergency_order_qty"].gt(0),
        [
            "run_id",
            "origin_date",
            "part_id",
            "region",
            "week",
            "emergency_order_qty",
            "emergency_cost_proxy",
        ],
    ].copy()
    if not emergency_orders.empty:
        emergency_orders = emergency_orders.rename(
            columns={
                "week": "order_week",
                "emergency_order_qty": "order_qty",
                "emergency_cost_proxy": "cost_proxy",
            }
        )
        emergency_orders["receipt_week"] = emergency_orders["order_week"] + 1
        emergency_orders["schedule_checker_included"] = False
        emergency_orders["basis"] = "proxy"
    if not checks_out.empty:
        checks_out["criticality"] = checks_out["part_id"].map(
            {part_id: part["criticality"] for part_id, part in parts.items()}
        )
    shortfalls_out = pd.DataFrame(shortfall_stage_rows)
    sob_rows = []
    for item in plan_items:
        snapshot = item["snapshot"]
        options = supplier_options(snapshot)
        if str(snapshot.get("supply_mode", "")).casefold() == "production":
            options = [
                row
                for row in options
                if str(row["type"]).casefold() != "import"
            ]
        eligible_share_total = sum(row["sob_pct"] for row in options)
        orders = item["orders"]
        total_order_qty = int(orders["order_qty"].sum()) if not orders.empty else 0
        for option in options:
            supplier_orders = (
                int(
                    orders.loc[
                        orders["supplier_id"].astype(str).eq(
                            str(option["id"])
                        ),
                        "order_qty",
                    ].sum()
                )
                if not orders.empty
                else 0
            )
            target_share = (
                100.0 * option["sob_pct"] / eligible_share_total
                if eligible_share_total
                else 0.0
            )
            actual_share = (
                100.0 * supplier_orders / total_order_qty
                if total_order_qty
                else 0.0
            )
            sob_rows.append(
                {
                    "run_id": item["run_id"],
                    "origin_date": item["forecast_group"][
                        "origin_date"
                    ].iloc[0],
                    "part_id": item["part_id"],
                    "supplier_id": str(option["id"]),
                    "supplier_slot": int(option["slot"]),
                    "order_qty": supplier_orders,
                    "sob_target_pct": target_share,
                    "actual_share_pct": actual_share,
                    "sob_deviation_pp": actual_share - target_share,
                    "status": item["final_status"],
                }
            )
    sob_out = pd.DataFrame(sob_rows)
    coverage_out, over_99_parts = _lead_time_coverage(paths, parts)
    population_out, initial_cover_out, excluded_reasons = (
        _path_population_audit(root, paths, scheduling)
    )
    cover_summary = {}
    for criticality, group in initial_cover_out.groupby(
        "criticality", observed=True
    ):
        finite = group["initial_cover_weeks"].replace(np.inf, np.nan).dropna()
        cover_summary[str(criticality)] = {
            "part_origins": int(len(group)),
            "zero_demand_part_origins": int(
                group["initial_cover_weeks"].eq(np.inf).sum()
            ),
            "median_weeks": float(finite.median()) if not finite.empty else None,
            "p10_weeks": float(finite.quantile(0.10)) if not finite.empty else None,
            "p90_weeks": float(finite.quantile(0.90)) if not finite.empty else None,
            "less_than_one_week_pct": float(
                group["initial_cover_weeks"].lt(1).mean() * 100
            ),
            "at_least_14_weeks_pct": float(
                group["initial_cover_weeks"].ge(14).mean() * 100
            ),
        }
    total_demand = float(simulation_out["demand_qty"].sum())
    total_current_fulfilled = float(
        simulation_out["current_demand_fulfilled_qty"].sum()
    )
    final_shortfalls = (
        shortfalls_out.loc[
            shortfalls_out["stage"].eq("after_pull_ahead"), "shortfall_qty"
        ]
        if not shortfalls_out.empty
        else pd.Series(dtype=float)
    )
    shortfall_total = float(final_shortfalls.sum())
    summary = {
        "status": "success",
        "assumption_version": assumptions["version"],
        "seed": seed,
        "scenario": {
            "initial_stock_multiplier": initial_stock_multiplier,
            "due_later_week": due_later_week,
            "rejection_variant": rejection_variant,
            "pull_ahead_enabled": use_pull_ahead,
        },
        "held_validation_origins": sorted(
            pd.to_datetime(paths["origin_date"]).dt.date.astype(str).unique().tolist()
        ),
        "complete_part_origin_paths": run_count,
        "actual_path_weeks": int(len(paths)),
        "orders": int(len(orders_out)),
        "mrp_plan_weeks": int(len(plan_weekly_out)),
        "simulation_weeks": int(len(simulation_out)),
        "schedule_checks": int(len(checks_out)),
        "schedule_feasible_runs": int(checks_out["feasible"].sum()),
        "schedule_unresolved_runs": int((~checks_out["feasible"]).sum()),
        "baseline_unresolved_before_pull_ahead": int(baseline_unresolved),
        "pull_ahead_orders": int(pull_ahead_count),
        "pull_ahead_quantity": int(pull_ahead_qty),
        "independent_checker_feasible_runs": int(
            checks_out["checker_status"].eq("firm").sum()
        ),
        "plan_unresolved_runs": int(
            checks_out["plan_status"].eq("unresolved").sum()
        ),
        "independent_checker_hard_violations": int(
            checks_out["independent_checker_failed_checks"].apply(
                lambda value: bool(
                    HARD_CONSTRAINTS.intersection(
                        str(value).split("|") if pd.notna(value) else []
                    )
                )
            ).sum()
        ),
        "unfilled_requirement_runs": int(
            checks_out["independent_checker_failed_checks"].str.contains(
                "unfilled_requirement", regex=False
            ).sum()
        ),
        "unfilled_requirement_shortfall_units": shortfall_total,
        "mrp_inventory_ledger_failed_runs": mrp_ledger_failures,
        "simulator_inventory_ledger_failed_runs": simulator_ledger_failures,
        "service_fill_rate": (
            total_current_fulfilled / total_demand if total_demand else None
        ),
        "stockout_part_week_pct": float(
            simulation_out["stockout_at_close_flag"].mean() * 100
        ),
        "emergency_order_count": int(len(emergency_orders)),
        "emergency_order_quantity": int(
            simulation_out["emergency_order_qty"].sum()
        ),
        "emergency_cost_proxy": float(
            simulation_out["emergency_cost_proxy"].sum()
        ),
        "excess_inventory_units_proxy": float(
            simulation_out["excess_inventory_units"].sum()
        ),
        "carrying_cost_proxy": float(
            simulation_out["carrying_cost_proxy"].sum()
        ),
        "excluded_640_path_analysis": excluded_reasons,
        "initial_cover_by_criticality": cover_summary,
        "lead_time_coverage_by_criticality": coverage_out.to_dict(
            orient="records"
        ),
        "service_targets_above_p99": over_99_parts,
        "tuning_quantile_forecasts_saved": False,
        "calibration_multiplier_fit_skipped": (
            "No saved tuning-origin quantile forecasts; multiplier remains off."
        ),
        "final_test_opened": False,
        "forecast_artifact_modified": False,
    }
    if write_outputs:
        results = root / "results"
        results.mkdir(parents=True, exist_ok=True)
        paths.to_csv(results / "phase5_validation_paths.csv", index=False)
        orders_out.to_csv(results / "phase5_mrp_orders.csv", index=False)
        plan_weekly_out.to_csv(results / "phase5_mrp_weekly.csv", index=False)
        simulation_out.to_csv(
            results / "phase5_simulation_weekly.csv", index=False
        )
        checks_out.to_csv(results / "phase5_schedule_checks.csv", index=False)
        emergency_orders.to_csv(
            results / "phase5_emergency_orders.csv", index=False
        )
        shortfalls_out.to_csv(
            results / "phase5_unresolved_shortfalls.csv", index=False
        )
        sob_out.to_csv(results / "phase5_sob_deviations.csv", index=False)
        coverage_out.to_csv(
            results / "phase5_lead_time_coverage.csv", index=False
        )
        population_out.to_csv(
            results / "phase5_path_population.csv", index=False
        )
        initial_cover_out.to_csv(
            results / "phase5_initial_cover.csv", index=False
        )
        (results / "phase5_run_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
    return summary


def run_phase5_sensitivities(root: Path = ROOT) -> pd.DataFrame:
    """Run configured one-factor scenarios without changing frozen defaults."""
    config = yaml.safe_load(
        (root / "configs" / "project.yaml").read_text(encoding="utf-8")
    )
    switches = config["scheduling"]["simulator_assumptions"][
        "sensitivity_scenarios"
    ]
    assumptions = config["scheduling"]["simulator_assumptions"]
    scenarios = [
        {
            "name": "baseline",
            "initial_stock_multiplier": float(
                assumptions["initial_inventory"]["initial_stock_multiplier"]
            ),
            "due_later_week": int(
                assumptions["inbound_receipts"]["due_later_week"]
            ),
            "rejection_variant": str(
                assumptions["supplier_uncertainty"][
                    "default_rejection_variant"
                ]
            ),
        },
        {
            "name": "initial_stock_0_5",
            "initial_stock_multiplier": float(
                switches["initial_stock_multiplier"]
            ),
            "due_later_week": int(
                assumptions["inbound_receipts"]["due_later_week"]
            ),
            "rejection_variant": str(
                assumptions["supplier_uncertainty"][
                    "default_rejection_variant"
                ]
            ),
        },
    ]
    scenarios.extend(
        {
            "name": f"due_later_week_{int(week)}",
            "initial_stock_multiplier": float(
                assumptions["initial_inventory"]["initial_stock_multiplier"]
            ),
            "due_later_week": int(week),
            "rejection_variant": str(
                assumptions["supplier_uncertainty"][
                    "default_rejection_variant"
                ]
            ),
        }
        for week in switches["due_later_weeks"]
    )
    scenarios.append(
        {
            "name": f"rejection_{switches['rejection_variant']}",
            "initial_stock_multiplier": float(
                assumptions["initial_inventory"]["initial_stock_multiplier"]
            ),
            "due_later_week": int(
                assumptions["inbound_receipts"]["due_later_week"]
            ),
            "rejection_variant": str(switches["rejection_variant"]),
        }
    )
    rows = []
    for scenario in scenarios:
        summary = run_phase5_core(
            root,
            initial_stock_multiplier=scenario["initial_stock_multiplier"],
            due_later_week=scenario["due_later_week"],
            rejection_variant=scenario["rejection_variant"],
            write_outputs=False,
        )
        rows.append(
            {
                "scenario": scenario["name"],
                **scenario,
                "service_fill_rate": summary["service_fill_rate"],
                "stockout_part_week_pct": summary["stockout_part_week_pct"],
                "emergency_order_count": summary["emergency_order_count"],
                "emergency_order_quantity": summary[
                    "emergency_order_quantity"
                ],
                "excess_inventory_units_proxy": summary[
                    "excess_inventory_units_proxy"
                ],
                "carrying_cost_proxy": summary["carrying_cost_proxy"],
                "unresolved_schedules": summary["schedule_unresolved_runs"],
                "unresolved_shortfall_units": summary[
                    "unfilled_requirement_shortfall_units"
                ],
                "hard_constraint_violations": summary[
                    "independent_checker_hard_violations"
                ],
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(root / "results" / "phase5_sensitivity.csv", index=False)
    return result


if __name__ == "__main__":
    print(run_phase5_core())
