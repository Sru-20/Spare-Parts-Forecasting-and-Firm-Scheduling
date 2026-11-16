"""Validation-only execution of the Phase 5 MRP and simulator core."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from math import ceil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from src.data.loader import load_scheduling
from src.features.segmentation import classify_sbc_series
from src.scheduling.checker import check_inventory_ledger, check_schedule
from src.scheduling.engine import (
    allocate_supplier_orders,
    build_mrp_schedule,
    order_arrival_week,
    snapshot_receipts,
    supplier_options,
    sob_weighted_lead_time,
)
from src.simulation.weekly import realize_supplier_orders, simulate_weekly


ROOT = Path(__file__).resolve().parents[2]
HISTORY_WEEKS = 13
MIN_COMPLETE_HISTORY_WEEKS = 8
PRIMARY_INITIALIZATION = "origin_visible"
D2_STRESS_INITIALIZATION = "d2_snapshot_stress"
RESULT_VERSION = "v5"
ANALOGUE_EXCEPTION_VERSION = "v1"
PLANNER_SUMMARY_VERSION = "v6"
HARD_CONSTRAINTS = {
    "moq",
    "order_multiple",
    "supplier_capacity",
    "lead_time",
    "release_before_week_1",
    "moq_multiple_rounding",
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


class _UniqueKeySafeLoader(yaml.SafeLoader):
    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        self.flatten_mapping(node)
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"duplicate key {key!r}",
                    key_node.start_mark,
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def _load_yaml_unique(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.load(stream, Loader=_UniqueKeySafeLoader)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def load_origin_visible_history(
    root: Path,
    origins: list[Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Aggregate only observed D1 labels dated before each validation origin."""
    path = root / "data" / "01_demand_forecasting.csv"
    origin_dates = sorted(
        {pd.Timestamp(value).normalize() for value in origins}
    )
    if not origin_dates:
        return {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        headers = next(reader, None)
        if headers is None:
            raise ValueError(f"Demand history is empty: {path}")
        if len(headers) != len(set(headers)):
            raise ValueError(f"Demand history has duplicate columns: {path}")
        columns = {
            name: headers.index(name)
            for name in (
                "week_start_date",
                "part_id",
                "region",
                "split_role",
                "demand_qty",
                "is_missing_week",
            )
        }
        history: dict[
            pd.Timestamp,
            dict[str, dict[pd.Timestamp, dict[str, float | None]]],
        ] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for row_number, row in enumerate(reader, start=2):
            week = pd.Timestamp(row[columns["week_start_date"]]).normalize()
            eligible_origins = [
                origin for origin in origin_dates if week < origin
            ]
            if not eligible_origins:
                continue
            if row[columns["split_role"]] not in {"warmup", "train_val"}:
                continue
            part_id = row[columns["part_id"]]
            region = row[columns["region"]]
            missing = row[columns["is_missing_week"]] == "1"
            target_text = row[columns["demand_qty"]]
            target: float | None = None
            if not missing and target_text != "":
                try:
                    target = float(target_text)
                except ValueError as error:
                    raise ValueError(
                        f"Invalid observed demand at {path}:{row_number}"
                    ) from error
                if not np.isfinite(target) or target < 0:
                    raise ValueError(
                        f"Invalid observed demand at {path}:{row_number}"
                    )
            for origin in eligible_origins:
                regions = history[origin][part_id][week]
                if region in regions:
                    raise ValueError(
                        f"Duplicate part-region-week history at {path}:{row_number}"
                    )
                regions[region] = None if missing or target is None else target

    output: dict[tuple[str, str], dict[str, Any]] = {}
    for origin in origin_dates:
        origin_key = origin.date().isoformat()
        window = {
            origin - pd.Timedelta(weeks=offset)
            for offset in range(1, HISTORY_WEEKS + 1)
        }
        for part_id, weeks in history[origin].items():
            expected_regions = {
                region for rows in weeks.values() for region in rows
            }
            observed_weekly_totals = []
            for week in sorted(window):
                regional = weeks.get(week, {})
                if (
                    expected_regions
                    and set(regional) == expected_regions
                    and all(value is not None for value in regional.values())
                ):
                    observed_weekly_totals.append(
                        float(sum(value for value in regional.values() if value is not None))
                    )
            complete_weeks = len(observed_weekly_totals)
            mean_demand = (
                float(np.mean(observed_weekly_totals))
                if complete_weeks
                else None
            )
            region_classes = []
            for region in expected_regions:
                region_history = [
                    rows[region]
                    for _, rows in sorted(weeks.items())
                    if region in rows
                ]
                classification = classify_sbc_series(
                    pd.Series(region_history, dtype="float64")
                )
                label = classification["sbc_class"]
                if isinstance(label, str):
                    region_classes.append(label)
            origin_sbc_class = (
                pd.Series(region_classes).mode().iloc[0]
                if region_classes
                else "Unknown/ShortHistory"
            )
            output[(origin_key, str(part_id))] = {
                "complete_observed_weeks": complete_weeks,
                "trailing_mean_weekly_demand": mean_demand,
                "origin_visible_sbc_class": str(origin_sbc_class),
                "history_status": (
                    "eligible"
                    if complete_weeks >= MIN_COMPLETE_HISTORY_WEEKS
                    else "insufficient_origin_visible_history"
                ),
            }
    return output


def origin_visible_opening_stock(
    part: Mapping[str, Any], history: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the v4.2 trailing-history and SOB-weighted lead-time formula."""
    complete_weeks = int(history["complete_observed_weeks"])
    if complete_weeks < MIN_COMPLETE_HISTORY_WEEKS:
        raise ValueError(
            "insufficient_origin_visible_history:"
            f"{complete_weeks}_of_{MIN_COMPLETE_HISTORY_WEEKS}"
        )
    mean_demand = history["trailing_mean_weekly_demand"]
    if mean_demand is None or not np.isfinite(float(mean_demand)):
        raise ValueError("origin_visible_history_mean_is_unavailable")
    lead_days = sob_weighted_lead_time(part)
    lead_weeks = int(ceil(lead_days / 7.0))
    rounded_weekly_mean = int(ceil(float(mean_demand)))
    return {
        "opening_available_qty": rounded_weekly_mean * (lead_weeks + 1),
        "trailing_mean_weekly_demand": float(mean_demand),
        "rounded_weekly_mean_demand": rounded_weekly_mean,
        "sob_weighted_lead_time_days": float(lead_days),
        "lead_time_weeks": lead_weeks,
        "complete_observed_weeks": complete_weeks,
    }


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
                "origin_date": pd.Timestamp(str(keys[0])),
                "part_id": keys[1],
                "week_start_date": pd.Timestamp(str(keys[2])),
                "horizon_step": int(str(keys[3])),
                "week": int(str(keys[3])),
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
    initialization_scenario: str,
    origin_history: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[Any, float], list[dict[str, Any]]]:
    plans = []
    exclusions = []
    shared_capacity = dict(capacity_seed)
    origin_key = pd.Timestamp(str(origin)).date().isoformat()
    for part_id, forecast_group in origin_paths.groupby(
        "part_id", sort=True, observed=True
    ):
        raw_snapshot = parts.get(str(part_id))
        if raw_snapshot is None:
            exclusions.append(
                {
                    "origin_date": origin_key,
                    "part_id": str(part_id),
                    "exclusion_reason": "missing_scheduling_snapshot",
                }
            )
            continue
        history = origin_history.get((origin_key, str(part_id)), {})
        if initialization_scenario == PRIMARY_INITIALIZATION:
            try:
                stock = origin_visible_opening_stock(raw_snapshot, history)
            except ValueError as error:
                exclusions.append(
                    {
                        "origin_date": origin_key,
                        "part_id": str(part_id),
                        "exclusion_reason": str(error),
                        "complete_observed_weeks": history.get(
                            "complete_observed_weeks", 0
                        ),
                    }
                )
                continue
            opening_qty = int(
                np.floor(
                    stock["opening_available_qty"] * initial_stock_multiplier
                )
            )
            snapshot = dict(raw_snapshot)
            snapshot.update(
                {
                    "available_qty": opening_qty,
                    "on_hand_qty": opening_qty,
                    "reserved_qty": 0,
                    "blocked_qty": 0,
                    "backorder_qty": 0,
                    "in_transit_qty": 0,
                    "open_po_qty": 0,
                    "open_po_due_this_month_qty": 0,
                    "open_po_due_later_qty": 0,
                }
            )
            initial_stock_details = {
                **stock,
                "opening_available_qty": opening_qty,
                "initial_stock_multiplier": initial_stock_multiplier,
            }
        elif initialization_scenario == D2_STRESS_INITIALIZATION:
            snapshot = _scaled_snapshot(raw_snapshot, initial_stock_multiplier)
            initial_stock_details = {
                "opening_available_qty": int(snapshot["available_qty"]),
                "complete_observed_weeks": history.get(
                    "complete_observed_weeks", 0
                ),
                "trailing_mean_weekly_demand": history.get(
                    "trailing_mean_weekly_demand"
                ),
                "initial_stock_multiplier": initial_stock_multiplier,
            }
        else:
            raise ValueError(
                f"Unknown initialization scenario: {initialization_scenario}"
            )
        run_id = f"phase5:{pd.Timestamp(str(origin)).date().isoformat()}:{part_id}"
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
                "initialization_scenario": initialization_scenario,
                "snapshot": snapshot,
                "initial_stock_details": initial_stock_details,
                "forecast_group": forecast_group,
                "plan": plan,
                "orders": orders,
            }
        )
    return plans, shared_capacity, exclusions


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
                    "initialization_scenario": item[
                        "initialization_scenario"
                    ],
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
    root: Path,
    paths: pd.DataFrame,
    scheduling: pd.DataFrame,
    initialization_scenario: str,
    origin_history: dict[tuple[str, str], dict[str, Any]],
    initial_stock_multiplier: float,
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
    universe["sbc_class"] = [
        origin_history.get(
            (
                pd.Timestamp(str(origin)).date().isoformat(),
                str(part_id),
            ),
            {},
        ).get("origin_visible_sbc_class", "Unknown/ShortHistory")
        for origin, part_id in zip(universe["origin_date"], universe["part_id"])
    ]
    universe = universe.merge(catalog, on="part_id", how="left", validate="many_to_one")
    reasons = universe["exclusion_reason"].value_counts().to_dict()
    universe["excluded"] = universe["exclusion_reason"].ne("complete")
    universe["initialization_scenario"] = initialization_scenario
    universe["complete_observed_history_weeks"] = [
        origin_history.get(
            (pd.Timestamp(str(origin)).date().isoformat(), str(part_id)), {}
        ).get("complete_observed_weeks", 0)
        for origin, part_id in zip(universe["origin_date"], universe["part_id"])
    ]
    universe["initialization_exclusion_reason"] = ""
    complete_mask = universe["exclusion_reason"].eq("complete")
    no_snapshot_mask = ~universe["part_id"].astype(str).isin(
        set(scheduling["part_id"].astype(str))
    )
    universe.loc[
        complete_mask & no_snapshot_mask, "initialization_exclusion_reason"
    ] = "missing_scheduling_snapshot"
    if initialization_scenario == PRIMARY_INITIALIZATION:
        insufficient = (
            complete_mask
            & ~no_snapshot_mask
            & universe["complete_observed_history_weeks"].lt(
                MIN_COMPLETE_HISTORY_WEEKS
            )
        )
        universe.loc[
            insufficient, "initialization_exclusion_reason"
        ] = "insufficient_origin_visible_history"
    initialization_reason_counts = (
        universe.loc[
            universe["initialization_exclusion_reason"].ne(""),
            "initialization_exclusion_reason",
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    initialization_exclusions_by_criticality = (
        universe.loc[universe["initialization_exclusion_reason"].ne("")]
        .groupby(
            ["criticality", "initialization_exclusion_reason"],
            dropna=False,
            observed=True,
        )
        .size()
        .rename("excluded_paths")
        .reset_index()
        .to_dict(orient="records")
    )
    initialization_exclusions_by_segment = (
        universe.loc[universe["initialization_exclusion_reason"].ne("")]
        .groupby(
            ["sbc_class", "initialization_exclusion_reason"],
            dropna=False,
            observed=True,
        )
        .size()
        .rename("excluded_paths")
        .reset_index()
        .to_dict(orient="records")
    )
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
    joined = paths.groupby(["origin_date", "part_id"], observed=True).size().reset_index(
        name="validation_weeks"
    )
    initial_cover = joined.merge(
        scheduling[["part_id", "criticality", "available_qty"]],
        on="part_id",
        how="inner",
        validate="many_to_one",
    )
    history_rows = [
        origin_history.get(
            (
                pd.Timestamp(str(row.origin_date)).date().isoformat(),
                str(row.part_id),
            ),
            {},
        )
        for row in initial_cover.itertuples(index=False)
    ]
    initial_cover["complete_observed_history_weeks"] = [
        int(item.get("complete_observed_weeks", 0)) for item in history_rows
    ]
    initial_cover["trailing_mean_weekly_demand"] = [
        item.get("trailing_mean_weekly_demand") for item in history_rows
    ]
    if initialization_scenario == PRIMARY_INITIALIZATION:
        stock_rows = []
        stock_exclusion_reasons = []
        for row, history_row in zip(
            initial_cover.to_dict(orient="records"), history_rows
        ):
            try:
                part_details = {
                    str(key): value
                    for key, value in scheduling.loc[
                        scheduling["part_id"].astype(str).eq(str(row["part_id"]))
                    ]
                    .iloc[0]
                    .to_dict()
                    .items()
                }
                stock_info = origin_visible_opening_stock(
                    part_details,
                    history_row,
                )
                stock_rows.append(
                    int(
                        np.floor(
                            stock_info["opening_available_qty"]
                            * initial_stock_multiplier
                        )
                    )
                )
                stock_exclusion_reasons.append("")
            except (ValueError, KeyError) as error:
                stock_rows.append(np.nan)
                stock_exclusion_reasons.append(str(error))
        initial_cover["opening_available_qty"] = stock_rows
        initial_cover["initialization_exclusion_reason"] = stock_exclusion_reasons
    else:
        initial_cover["opening_available_qty"] = np.floor(
            initial_cover["available_qty"] * initial_stock_multiplier
        ).astype("Int64")
        initial_cover["initialization_exclusion_reason"] = np.where(
            initial_cover["complete_observed_history_weeks"].lt(
                MIN_COMPLETE_HISTORY_WEEKS
            ),
            "cover_unavailable_insufficient_origin_visible_history",
            "",
        )
    initial_cover["initial_cover_weeks"] = np.where(
        initial_cover["trailing_mean_weekly_demand"].fillna(0).gt(0)
        & initial_cover["complete_observed_history_weeks"].ge(
            MIN_COMPLETE_HISTORY_WEEKS
        ),
        initial_cover["opening_available_qty"]
        / initial_cover["trailing_mean_weekly_demand"],
        np.inf,
    )
    initial_cover.loc[
        initial_cover["initialization_exclusion_reason"].ne(""),
        "initial_cover_weeks",
    ] = np.nan
    initial_cover["initialization_scenario"] = initialization_scenario
    initial_cover["initial_stock_multiplier"] = initial_stock_multiplier
    initial_cover["basis"] = (
        "origin_visible_initialization"
        if initialization_scenario == PRIMARY_INITIALIZATION
        else "d2_snapshot_stress"
    )
    return (
        universe,
        initial_cover,
        {
            "reason_counts": reasons,
            "exclusion_rate_by_criticality": criticality_rates,
            "exclusion_rate_by_segment": segment_rates,
            "initialization_exclusion_reason_counts": initialization_reason_counts,
            "initialization_exclusions_by_criticality": (
                initialization_exclusions_by_criticality
            ),
            "initialization_exclusions_by_segment": (
                initialization_exclusions_by_segment
            ),
        },
    )


def _write_unresolved_reports(
    checks: pd.DataFrame,
    shortfalls: pd.DataFrame,
    results: Path,
    output_prefix: str,
) -> None:
    if checks.empty:
        _write_frame_without_overwrite(
            pd.DataFrame(
                columns=["criticality", "failure_reason", "unresolved_schedules"]
            ),
            results / f"{output_prefix}_unresolved_by_reason_criticality.csv",
        )
        _write_frame_without_overwrite(
            pd.DataFrame(
                columns=[
                    "requirement_month",
                    "criticality",
                    "failure_reason",
                    "unresolved_schedules",
                    "shortfall_rows",
                    "shortfall_units",
                ]
            ),
            results / f"{output_prefix}_unresolved_by_month.csv",
        )
        return
    failure_rows = []
    for row in checks.loc[checks["status"].eq("unresolved")].to_dict(
        orient="records"
    ):
        for reason in filter(None, str(row.get("failed_checks", "")).split("|")):
            failure_rows.append(
                {
                    "run_id": row["run_id"],
                    "criticality": row.get("criticality"),
                    "failure_reason": reason,
                }
            )
    by_reason = pd.DataFrame(failure_rows)
    if by_reason.empty:
        by_reason = pd.DataFrame(
            columns=["criticality", "failure_reason", "unresolved_schedules"]
        )
    else:
        by_reason = (
            by_reason.groupby(
                ["criticality", "failure_reason"], dropna=False, observed=True
            )
            .agg(unresolved_schedules=("run_id", "nunique"))
            .reset_index()
        )
    _write_frame_without_overwrite(
        by_reason,
        results / f"{output_prefix}_unresolved_by_reason_criticality.csv",
    )

    month_rows = []
    check_reasons = checks.set_index("run_id")["failed_checks"].to_dict()
    unresolved_ids = set(
        checks.loc[checks["status"].eq("unresolved"), "run_id"]
    )
    if not shortfalls.empty:
        final_shortfalls = shortfalls.loc[
            shortfalls["stage"].eq("after_pull_ahead")
            & shortfalls["run_id"].isin(unresolved_ids)
        ]
        for row in final_shortfalls.to_dict(orient="records"):
            reasons = str(check_reasons.get(row["run_id"], "") or "")
            for reason in filter(None, reasons.split("|")):
                month_rows.append(
                    {
                        "requirement_month": row["requirement_month"],
                        "criticality": row["criticality"],
                        "failure_reason": reason,
                        "run_id": row["run_id"],
                        "shortfall_qty": row["shortfall_qty"],
                    }
                )
    for row in checks.loc[
        checks["status"].eq("unresolved")
        & checks["failed_checks"].str.contains(
            "release_before_week_1", regex=False, na=False
        )
    ].to_dict(orient="records"):
        month_rows.append(
            {
                "requirement_month": pd.Timestamp(
                    str(row["week_1_start_date"])
                ).strftime("%Y-%m"),
                "criticality": row["criticality"],
                "failure_reason": "release_before_week_1",
                "run_id": row["run_id"],
                "shortfall_qty": 0,
            }
        )
    by_month = pd.DataFrame(month_rows)
    if by_month.empty:
        by_month = pd.DataFrame(
            columns=[
                "requirement_month",
                "criticality",
                "failure_reason",
                "unresolved_schedules",
                "shortfall_rows",
                "shortfall_units",
            ]
        )
    else:
        by_month = (
            by_month.groupby(
                ["requirement_month", "criticality", "failure_reason"],
                dropna=False,
                observed=True,
            )
            .agg(
                unresolved_schedules=("run_id", "nunique"),
                shortfall_rows=("run_id", "size"),
                shortfall_units=("shortfall_qty", "sum"),
            )
            .reset_index()
        )
    _write_frame_without_overwrite(
        by_month, results / f"{output_prefix}_unresolved_by_month.csv"
    )


def _write_frame_without_overwrite(frame: pd.DataFrame, path: Path) -> None:
    content = frame.to_csv(index=False, lineterminator="\n")
    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            return
        raise FileExistsError(
            f"Refusing to overwrite existing result with different content: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="")


def _write_json_without_overwrite(value: dict[str, Any], path: Path) -> None:
    content = json.dumps(value, indent=2, default=str) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            return
        raise FileExistsError(
            f"Refusing to overwrite existing result with different content: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="")


def run_phase5_core(
    root: Path = ROOT,
    *,
    initial_stock_multiplier: float | None = None,
    due_later_week: int | None = None,
    rejection_variant: str | None = None,
    use_pull_ahead: bool | None = None,
    initialization_scenario: str = PRIMARY_INITIALIZATION,
    write_outputs: bool = True,
) -> dict[str, Any]:
    """Run the Phase 5 MRP and simulator on saved held-validation paths."""
    config = _load_yaml_unique(root / "configs" / "project.yaml")
    business = _load_yaml_unique(root / "configs" / "business.yaml")
    if initialization_scenario not in {
        PRIMARY_INITIALIZATION,
        D2_STRESS_INITIALIZATION,
    }:
        raise ValueError(f"Unknown initialization scenario: {initialization_scenario}")
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
    if (
        not np.isfinite(initial_stock_multiplier)
        or initial_stock_multiplier < 0
        or due_later_week < 1
    ):
        raise ValueError("Scenario inventory multiplier or receipt week is invalid")
    if rejection_variant not in {"whole_lot", "per_unit_binomial"}:
        raise ValueError(f"Unsupported rejection variant: {rejection_variant}")
    seed = int(assumptions["seed"])
    scheduling = load_scheduling(root / "data" / "02_scheduling_inputs.csv")
    parts = {
        str(row["part_id"]): {
            str(key): value for key, value in row.items()
        }
        for row in scheduling.to_dict(orient="records")
    }
    paths = load_validation_paths(root)
    origins = pd.to_datetime(paths["origin_date"]).drop_duplicates().tolist()
    origin_history = load_origin_visible_history(root, origins)
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
    initialization_exclusions: list[dict[str, Any]] = []
    shortfall_stage_rows: list[dict[str, Any]] = []
    pull_ahead_count = 0
    pull_ahead_qty = 0
    run_count = 0
    for origin, origin_paths in paths.groupby(
        "origin_date", sort=True, observed=True
    ):
        baseline_plans, baseline_capacity, _ = _generate_origin_plans(
            origin,
            origin_paths,
            parts,
            {},
            {},
            initial_stock_multiplier,
            due_later_week,
            initialization_scenario,
            origin_history,
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
        final_plans, _, origin_exclusions = _generate_origin_plans(
            origin,
            origin_paths,
            parts,
            _capacity_reservations(pulled_by_part),
            pulled_by_part,
            initial_stock_multiplier,
            due_later_week,
            initialization_scenario,
            origin_history,
        )
        initialization_exclusions.extend(origin_exclusions)
        run_count += len(final_plans)
        plans = final_plans
        shortfall_stage_rows.extend(
            _collect_plan_shortfalls(plans, "after_pull_ahead")
        )
        for item in plans:
            if not item["plan"]["weekly"].empty:
                plan_weekly = item["plan"]["weekly"].copy()
                plan_weekly["origin_date"] = origin
                plan_weekly["initialization_scenario"] = initialization_scenario
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
        origin_capacity_failures = (
            set(origin_check["failed_checks"]) & {"supplier_capacity"}
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
                | origin_capacity_failures
            )
            final_status = (
                "unresolved"
                if item["plan"]["status"] != "firm" or failures
                else "not_ordered"
                if part_orders.empty
                else "firm"
            )
            checker_rows.append(
                {
                    "run_id": item["run_id"],
                    "part_id": part_id,
                    "origin_date": origin,
                    "week_1_start_date": pd.to_datetime(
                        item["forecast_group"]["week_start_date"]
                    ).min(),
                    "initialization_scenario": initialization_scenario,
                    "initial_available_qty": item["initial_stock_details"][
                        "opening_available_qty"
                    ],
                    "trailing_mean_weekly_demand": item[
                        "initial_stock_details"
                    ].get("trailing_mean_weekly_demand"),
                    "complete_observed_history_weeks": item[
                        "initial_stock_details"
                    ].get("complete_observed_weeks"),
                    "plan_status": item["plan"]["status"],
                    "status": final_status,
                    "checker_status": (
                        "firm"
                        if part_check["feasible"] and not origin_capacity_failures
                        else "unresolved"
                    ),
                    "plan_failed_checks": "|".join(
                        item["plan"]["failed_checks"]
                    ),
                    "independent_checker_failed_checks": "|".join(
                        sorted(
                            set(part_check["failed_checks"])
                            | origin_capacity_failures
                        )
                    ),
                    "feasible": final_status != "unresolved",
                    "failed_checks": "|".join(failures),
                }
            )
            item["final_status"] = final_status
            plan_items.append(item)
            if not part_orders.empty:
                part_orders["status"] = final_status
                part_orders["plan_status"] = item["plan"]["status"]
                part_orders["independent_checker_status"] = (
                    "firm"
                    if part_check["feasible"] and not origin_capacity_failures
                    else "unresolved"
                )
                part_orders["independent_checker_failed_checks"] = "|".join(
                    sorted(
                        set(part_check["failed_checks"])
                        | origin_capacity_failures
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
            simulated["initialization_scenario"] = initialization_scenario
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
            checks_out.loc[index, "status"] = "unresolved"
            checks_out.loc[index, "feasible"] = False
    if not orders_out.empty:
        statuses = checks_out.set_index("run_id")["status"]
        orders_out["status"] = orders_out["run_id"].map(statuses)
        orders_out["initialization_scenario"] = initialization_scenario
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
        _path_population_audit(
            root,
            paths,
            scheduling,
            initialization_scenario,
            origin_history,
            initial_stock_multiplier,
        )
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
    reason_counts = defaultdict(int)
    for row in initialization_exclusions:
        reason_counts[row["exclusion_reason"].split(":")[0]] += 1
    checker_hard_failures = checks_out[
        "independent_checker_failed_checks"
    ].apply(
        lambda value: bool(
            HARD_CONSTRAINTS.intersection(
                str(value).split("|") if pd.notna(value) else []
            )
        )
    )
    firm_order_hard_failures = int(
        (
            checker_hard_failures
            & checks_out["status"].eq("firm")
        ).sum()
    )
    unresolved_checker_hard_failures = int(
        (
            checker_hard_failures
            & checks_out["status"].eq("unresolved")
        ).sum()
    )
    summary = {
        "status": "success",
        "assumption_version": assumptions["version"],
        "config_sha256": {
            "project.yaml": hashlib.sha256(
                (root / "configs" / "project.yaml").read_bytes()
            ).hexdigest(),
            "business.yaml": hashlib.sha256(
                (root / "configs" / "business.yaml").read_bytes()
            ).hexdigest(),
        },
        "seed": seed,
        "initialization_scenario": initialization_scenario,
        "scenario": {
            "initial_stock_multiplier": initial_stock_multiplier,
            "due_later_week": due_later_week,
            "rejection_variant": rejection_variant,
            "pull_ahead_enabled": use_pull_ahead,
        },
        "held_validation_origins": sorted(
            str(value)
            for value in pd.to_datetime(paths["origin_date"])
            .dt.date.astype(str)
            .unique()
            .tolist()
        ),
        "complete_part_origin_paths": run_count,
        "initialization_excluded_paths": len(initialization_exclusions),
        "initialization_exclusion_reason_counts": dict(sorted(reason_counts.items())),
        "actual_path_weeks": int(len(paths)),
        "orders": int(len(orders_out)),
        "mrp_plan_weeks": int(len(plan_weekly_out)),
        "simulation_weeks": int(len(simulation_out)),
        "schedule_checks": int(len(checks_out)),
        "schedule_firm_runs": int(checks_out["status"].eq("firm").sum()),
        "schedule_feasible_runs": int(
            checks_out["checker_status"].eq("firm").sum()
        ),
        "schedule_unresolved_runs": int(
            checks_out["status"].eq("unresolved").sum()
        ),
        "schedule_not_ordered_runs": int(
            checks_out["status"].eq("not_ordered").sum()
        ),
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
            checker_hard_failures.sum()
        ),
        "firm_order_hard_constraint_violations": firm_order_hard_failures,
        "unresolved_checker_hard_failure_runs": unresolved_checker_hard_failures,
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
        "this_run_accessed_sealed_test_labels": False,
        "forecast_artifact_modified": False,
    }
    if write_outputs:
        results = root / "results"
        results.mkdir(parents=True, exist_ok=True)
        output_prefix = (
            f"phase5_primary_{RESULT_VERSION}"
            if initialization_scenario == PRIMARY_INITIALIZATION
            else f"phase5_d2_snapshot_stress_{RESULT_VERSION}"
        )
        _write_frame_without_overwrite(
            paths.assign(initialization_scenario=initialization_scenario),
            results / f"{output_prefix}_validation_paths.csv",
        )
        _write_frame_without_overwrite(
            orders_out, results / f"{output_prefix}_mrp_orders.csv"
        )
        _write_frame_without_overwrite(
            plan_weekly_out, results / f"{output_prefix}_mrp_weekly.csv"
        )
        _write_frame_without_overwrite(
            simulation_out, results / f"{output_prefix}_simulation_weekly.csv"
        )
        _write_frame_without_overwrite(
            checks_out, results / f"{output_prefix}_schedule_checks.csv"
        )
        _write_frame_without_overwrite(
            emergency_orders, results / f"{output_prefix}_emergency_orders.csv"
        )
        _write_frame_without_overwrite(
            shortfalls_out, results / f"{output_prefix}_unresolved_shortfalls.csv"
        )
        _write_frame_without_overwrite(
            sob_out, results / f"{output_prefix}_sob_deviations.csv"
        )
        _write_frame_without_overwrite(
            coverage_out, results / f"{output_prefix}_lead_time_coverage.csv"
        )
        _write_frame_without_overwrite(
            population_out, results / f"{output_prefix}_path_population.csv"
        )
        _write_frame_without_overwrite(
            initial_cover_out, results / f"{output_prefix}_initial_cover.csv"
        )
        _write_frame_without_overwrite(
            pd.DataFrame(initialization_exclusions),
            results / f"{output_prefix}_initialization_exclusions.csv",
        )
        _write_unresolved_reports(
            checks_out, shortfalls_out, results, output_prefix
        )
        high_critical_shortages = (
            shortfalls_out.loc[shortfalls_out["criticality"].eq("High")].sort_values(
                ["week_start_date", "shortfall_qty"],
                ascending=[True, False],
                kind="stable",
            )
            if "criticality" in shortfalls_out
            else shortfalls_out
        )
        _write_frame_without_overwrite(
            high_critical_shortages,
            results / f"{output_prefix}_critical_shortages.csv",
        )
        _write_json_without_overwrite(
            summary, results / f"{output_prefix}_run_summary.json"
        )
    return summary


def run_phase5_sensitivities(root: Path = ROOT) -> pd.DataFrame:
    """Run configured one-factor scenarios without changing frozen defaults."""
    config = _load_yaml_unique(root / "configs" / "project.yaml")
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
    ]
    scenarios.extend(
        {
            "name": f"initial_stock_{str(multiplier).replace('.', '_')}",
            "initial_stock_multiplier": float(multiplier),
            "due_later_week": int(
                assumptions["inbound_receipts"]["due_later_week"]
            ),
            "rejection_variant": str(
                assumptions["supplier_uncertainty"][
                    "default_rejection_variant"
                ]
            ),
        }
        for multiplier in switches["initial_stock_multipliers"]
    )
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
            initialization_scenario=PRIMARY_INITIALIZATION,
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
    _write_frame_without_overwrite(
        result, root / "results" / f"phase5_primary_sensitivity_{RESULT_VERSION}.csv"
    )
    return result


def _write_text_without_overwrite(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            return
        raise FileExistsError(
            f"Refusing to overwrite existing result with different content: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="")


def write_analogue_fallback_diagnostic(root: Path = ROOT) -> dict[str, Any]:
    """Record unforecastable short series without inventing forecast values."""
    catalog = pd.read_csv(
        root / "data" / "03_spare_parts_catalogue.csv",
        usecols=["part_id", "component", "vehicle_type"],
        dtype={"part_id": "string"},
    )
    if catalog["part_id"].astype(str).duplicated().any():
        raise ValueError("Catalogue contains duplicate part IDs")
    segments = pd.read_parquet(
        root / "artifacts" / "series_segments.parquet",
        columns=[
            "part_id",
            "region",
            "classification_status",
            "observed_weeks",
        ],
    )
    short_series = segments.loc[
        segments["classification_status"].astype("string").str.startswith(
            "unclassified", na=False
        )
    ].copy()
    if short_series.duplicated(["part_id", "region"]).any():
        raise ValueError("Segmentation artifact contains duplicate series keys")

    configuration = _load_yaml_unique(root / "configs" / "project.yaml")
    origins = configuration.get("phase4", {}).get(
        "tier2_random_forest", {}
    ).get("held_validation_origins", [])
    if not origins:
        raise ValueError("No frozen held-validation origins are configured")

    catalog = catalog.set_index("part_id", drop=False)
    if not short_series["part_id"].astype(str).isin(catalog.index.astype(str)).all():
        raise ValueError("A short series has no matching catalogue part")

    exception_rows: list[dict[str, Any]] = []
    for series in short_series.sort_values(["part_id", "region"]).itertuples(
        index=False
    ):
        part_id = str(series.part_id)
        metadata = catalog.loc[part_id]
        component = metadata["component"]
        vehicle_type = metadata["vehicle_type"]
        if pd.isna(component) or pd.isna(vehicle_type):
            eligible_donor_count = 0
        else:
            eligible = catalog.loc[
                catalog["component"].eq(component).fillna(False)
                & catalog["vehicle_type"].eq(vehicle_type).fillna(False)
                & catalog["part_id"].ne(part_id)
            ]
            eligible_donor_count = int(eligible["part_id"].nunique())
        if eligible_donor_count:
            raise RuntimeError(
                "Exact component/vehicle-type donors exist, but fallback "
                "forecast generation is not implemented; refusing to label "
                "the series as unforecastable."
            )
        reason = "no_eligible_same_component_vehicle_type_donor"
        if series.classification_status == "unclassified_no_development_observations":
            reason = (
                "no_development_observations;"
                "no_validated_launch_exposure_scaling;"
                "no_eligible_same_component_vehicle_type_donor"
            )
        for origin in sorted({str(value) for value in origins}):
            exception_rows.append(
                {
                    "origin_date": origin,
                    "part_id": part_id,
                    "region": str(series.region),
                    "classification_status": str(series.classification_status),
                    "observed_development_weeks": int(series.observed_weeks),
                    "component": component,
                    "vehicle_type": vehicle_type,
                    "eligible_donor_count": eligible_donor_count,
                    "status": "UNFORECASTED_NO_DEFENSIBLE_ANALOGUE",
                    "forecast_emitted": False,
                    "reason": reason,
                }
            )

    exceptions = pd.DataFrame(exception_rows)
    if exceptions.empty:
        raise ValueError("No unclassified short series were found to document")
    reason_counts = {
        str(reason): int(count)
        for reason, count in exceptions["reason"].value_counts().items()
    }
    exception_csv = (
        root
        / "results"
        / f"phase5_analogue_exceptions_{ANALOGUE_EXCEPTION_VERSION}.csv"
    )
    exception_json = (
        root
        / "results"
        / f"phase5_analogue_exceptions_{ANALOGUE_EXCEPTION_VERSION}.json"
    )
    exception_report = (
        root
        / "results"
        / f"phase5_analogue_exception_summary_{ANALOGUE_EXCEPTION_VERSION}.md"
    )
    summary = {
        "status": "approved_data_limitation_exception",
        "donor_definition": (
            "exact same component and vehicle_type, excluding the target part"
        ),
        "short_series": int(len(short_series)),
        "part_ids": sorted(short_series["part_id"].astype(str).unique()),
        "validation_origins": sorted({str(value) for value in origins}),
        "series_origin_records": int(len(exceptions)),
        "forecasted_series": 0,
        "unforecasted_series": int(len(short_series)),
        "eligible_donor_count_by_series_origin": sorted(
            int(value) for value in exceptions["eligible_donor_count"].unique()
        ),
        "status_by_series_origin": "UNFORECASTED_NO_DEFENSIBLE_ANALOGUE",
        "reason_counts_by_series_origin": reason_counts,
        "forecast_values_emitted": False,
        "forecast_qty_substituted_with_zero": False,
        "p0091_reason": (
            "no development observations and no validated launch/exposure "
            "scaling"
        ),
        "evidence_csv": str(exception_csv.relative_to(root)),
    }
    _write_frame_without_overwrite(exceptions, exception_csv)
    _write_json_without_overwrite(summary, exception_json)
    _write_text_without_overwrite(
        exception_report,
        "\n".join(
            [
                "# Phase 5 short-series analogue exception",
                "",
                "Approved data-limitation exception; no fallback forecast values were generated.",
                "",
                f"- Affected series: {len(short_series)}.",
                f"- Forecasted short series: 0; unforecasted short series: {len(short_series)}.",
                f"- Held validation origins: {', '.join(summary['validation_origins'])}.",
                f"- Series-origin exception records: {len(exceptions)}.",
                "- Exact same-component/same-vehicle-type donors: 0 for every recorded series-origin.",
                "- Forecast rows emitted: 0. No missing forecast was replaced by zero demand.",
                "- Status: `UNFORECASTED_NO_DEFENSIBLE_ANALOGUE`.",
                "- P0091 remains unforecasted: no development observations and no validated launch/exposure scaling.",
                f"- Detail: `{exception_csv.relative_to(root)}`.",
                "",
            ]
        ),
    )
    return {
        **summary,
        "eligible_donor_parts": 0,
        "confidence": "low",
        "safety_stock_policy": "point_forecast_only",
        "frozen_forecasts_modified": False,
        "reason": (
            "No eligible exact-rule donors exist. Each affected series-origin "
            "is explicitly unforecasted; no quantity was fabricated."
        ),
    }


def _format_example(row: pd.Series | None) -> str:
    if row is None:
        return "No qualifying example was produced."
    return ", ".join(f"{key}={value}" for key, value in row.items())


def run_phase5_mvp(root: Path = ROOT) -> dict[str, Any]:
    """Run both initialization cases, required sensitivity, and safe evidence."""
    primary = run_phase5_core(
        root, initialization_scenario=PRIMARY_INITIALIZATION
    )
    stress = run_phase5_core(
        root, initialization_scenario=D2_STRESS_INITIALIZATION
    )
    sensitivity = run_phase5_sensitivities(root)
    analogue = write_analogue_fallback_diagnostic(root)
    repeated_primary = run_phase5_core(
        root, initialization_scenario=PRIMARY_INITIALIZATION
    )
    primary_repeat_identical = json.dumps(
        primary, sort_keys=True, default=str
    ) == json.dumps(repeated_primary, sort_keys=True, default=str)
    if not primary_repeat_identical:
        raise AssertionError("Primary Phase 5 repeat run was not identical")

    comparison = pd.DataFrame(
        [
            {
                "initialization_scenario": PRIMARY_INITIALIZATION,
                "schedule_checks": primary["schedule_checks"],
                "firm_runs": primary["schedule_firm_runs"],
                "unresolved_runs": primary["schedule_unresolved_runs"],
                "not_ordered_runs": primary["schedule_not_ordered_runs"],
                "hard_constraint_violations": primary[
                    "independent_checker_hard_violations"
                ],
                "service_fill_rate": primary["service_fill_rate"],
                "stockout_part_week_pct": primary["stockout_part_week_pct"],
            },
            {
                "initialization_scenario": D2_STRESS_INITIALIZATION,
                "schedule_checks": stress["schedule_checks"],
                "firm_runs": stress["schedule_firm_runs"],
                "unresolved_runs": stress["schedule_unresolved_runs"],
                "not_ordered_runs": stress["schedule_not_ordered_runs"],
                "hard_constraint_violations": stress[
                    "independent_checker_hard_violations"
                ],
                "service_fill_rate": stress["service_fill_rate"],
                "stockout_part_week_pct": stress["stockout_part_week_pct"],
            },
        ]
    )
    _write_frame_without_overwrite(
        comparison,
        root / "results" / f"phase5_scenario_comparison_{RESULT_VERSION}.csv",
    )
    results = root / "results"
    output_prefix = f"phase5_primary_{RESULT_VERSION}"
    firm_orders = pd.read_csv(results / f"{output_prefix}_mrp_orders.csv")
    firm_orders = firm_orders.loc[
        firm_orders["status"].eq("firm") & firm_orders["order_qty"].gt(0)
    ]
    unresolved_checks = pd.read_csv(
        results / f"{output_prefix}_schedule_checks.csv"
    )
    unresolved_checks = unresolved_checks.loc[
        unresolved_checks["status"].eq("unresolved")
    ]
    firm_example = (
        _format_example(firm_orders.iloc[0]) if not firm_orders.empty else None
    )
    unresolved_example = (
        _format_example(unresolved_checks.iloc[0])
        if not unresolved_checks.empty
        else None
    )
    report_lines = [
        "# Phase 5 simulated planning demonstration",
        "",
        "Method validated on simulated data; this is not evidence of factory savings or a supplier commitment.",
        "",
        "## Initialization and schedule evidence",
        "",
        f"- Primary initialization: {primary['initialization_scenario']}; {primary['schedule_firm_runs']} firm, {primary['schedule_unresolved_runs']} unresolved, {primary['schedule_not_ordered_runs']} not ordered; {primary['schedule_feasible_runs']} checker-clean including not-ordered runs.",
        f"- D2 stress initialization: {stress['initialization_scenario']}; {stress['schedule_firm_runs']} firm, {stress['schedule_unresolved_runs']} unresolved, {stress['schedule_not_ordered_runs']} not ordered; {stress['schedule_feasible_runs']} checker-clean including not-ordered runs.",
        f"- Primary firm-order hard-constraint violations: {primary['firm_order_hard_constraint_violations']}; unresolved runs with checker hard failures: {primary['unresolved_checker_hard_failure_runs']}; MRP ledger failures: {primary['mrp_inventory_ledger_failed_runs']}; simulator ledger failures: {primary['simulator_inventory_ledger_failed_runs']}.",
        f"- Initial cover by criticality: `{json.dumps(primary['initial_cover_by_criticality'], sort_keys=True)}`.",
        f"- D2 stress initial cover by criticality: `{json.dumps(stress['initial_cover_by_criticality'], sort_keys=True)}`.",
        f"- Sensitivity scenarios recorded: {', '.join(sensitivity['scenario'].astype(str))}.",
        "",
        "## Order examples",
        "",
        f"- Firm order: {firm_example or 'No qualifying firm order was produced.'}",
        f"- Unresolved schedule: {unresolved_example or 'No unresolved schedule was produced.'}",
        "",
        "## Short-series analogue outcome",
        "",
        f"- Forecasted short series: {analogue['forecasted_series']}; unforecasted short series: {analogue['unforecasted_series']} across {analogue['series_origin_records']} series-origin records.",
        "- Series without an eligible exact-rule donor are recorded as `UNFORECASTED_NO_DEFENSIBLE_ANALOGUE`; no forecast quantity is emitted or replaced by zero demand.",
        "- P0091 remains unforecasted because it has no development observations and no validated launch/exposure scaling.",
        f"- Versioned exception evidence: `{Path(analogue['evidence_csv']).as_posix()}` and `results/phase5_analogue_exceptions_{ANALOGUE_EXCEPTION_VERSION}.json`.",
        "The dataset validator was intentionally not rerun. A previous validator execution read sealed final-test target values for integrity checks, not prediction evaluation; this session used held-validation forecast artifacts only.",
        "",
        "Reproduce with `python -m src.scheduling.phase5 --demo`.",
    ]
    _write_text_without_overwrite(
        results / f"phase5_mvp_planner_summary_{PLANNER_SUMMARY_VERSION}.md",
        "\n".join(report_lines) + "\n",
    )
    repeat_evidence = {
        "repeat_run_summary_identical": primary_repeat_identical,
        "repeat_run_output_identical": True,
        "verification_method": (
            "The repeated run writes to the same result paths through a "
            "no-overwrite writer that accepts only byte-identical content."
        ),
        "initialization_scenario": PRIMARY_INITIALIZATION,
    }
    _write_json_without_overwrite(
        repeat_evidence,
        results / f"phase5_repeat_run_evidence_{RESULT_VERSION}.json",
    )
    return {
        "primary": primary,
        "d2_snapshot_stress": stress,
        "sensitivity_scenarios": int(len(sensitivity)),
        "analogue_fallback": analogue,
        "repeat_run_identical": primary_repeat_identical,
        "report": str(
            results / f"phase5_mvp_planner_summary_{PLANNER_SUMMARY_VERSION}.md"
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the primary and D2 stress scenarios, sensitivities, and reports",
    )
    arguments = parser.parse_args()
    if arguments.demo:
        print(json.dumps(run_phase5_mvp(), indent=2, default=str))
    else:
        print(json.dumps(run_phase5_core(), indent=2, default=str))
