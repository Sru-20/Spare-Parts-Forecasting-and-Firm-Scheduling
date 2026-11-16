"""Frozen-validation scheduler comparison and provisional schedule generation."""
from __future__ import annotations

import hashlib
import json
from math import ceil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from src.data.loader import load_scheduling
from src.scheduling.checker import check_inventory_ledger, check_schedule
from src.scheduling.engine import (
    allocate_supplier_orders,
    build_mrp_schedule,
    calculate_usable_stock,
    snapshot_receipts,
    supplier_options,
)
from src.scheduling.phase5 import (
    D2_STRESS_INITIALIZATION,
    PRIMARY_INITIALIZATION,
    _supplier_snapshot,
    load_origin_visible_history,
    origin_visible_opening_stock,
)
from src.simulation.weekly import realize_supplier_orders, simulate_weekly


ROOT = Path(__file__).resolve().parents[2]
RESULT_VERSION = "v1"
POINT_CANDIDATES = (
    "lightgbm_tweedie::random_1",
    "lightgbm_quantile::random_1",
)
SCHEDULERS = ("traditional_baseline", "mrp", "risk_adjusted_mrp")
INITIALIZATIONS = (PRIMARY_INITIALIZATION, D2_STRESS_INITIALIZATION)
KEYS = ["origin_date", "part_id", "week"]
OUTPUTS = (
    "results/phase6_scheduler_comparison_v1.csv",
    "results/phase6_kpi_comparison_v1.csv",
    "results/phase6_weight_sensitivity_v1.csv",
    "results/phase6_paired_bootstrap_v1.csv",
    "results/phase6_cost_sensitivity_v1.csv",
    "results/phase6_initialization_comparison_v1.csv",
    "results/phase6_exclusions_v1.csv",
    "results/phase6_checker_summary_v1.csv",
    "results/phase6_provisional_firm_schedule_v1.csv",
    "reports/phase6_comparison_report.md",
)


def _config(root: Path) -> dict[str, Any]:
    with (root / "configs" / "project.yaml").open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("project.yaml must contain a mapping")
    weights = value["scoring"]["tracks"]["scheduling"]["weights"]
    if not np.isclose(sum(float(item) for item in weights.values()), 1.0):
        raise ValueError("Track B weights must sum to 1.0")
    return value


def _write_new_csv(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing Phase 6 output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_new_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing Phase 6 output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _aggregate_model(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    selected = frame.loc[frame["model"].eq(model)].copy()
    if selected.empty:
        raise ValueError(f"Missing saved validation forecasts for {model}")
    selected["week_start_date"] = pd.to_datetime(selected["week_start_date"])
    grouped = selected.groupby(
        ["origin_date", "part_id", "week_start_date", "horizon_step"],
        as_index=False,
        sort=True,
        observed=True,
    ).agg(
        forecast_qty=("forecast_qty", lambda values: values.sum(min_count=1)),
        actual_qty=("actual_qty", lambda values: values.sum(min_count=len(values))),
        forecast_p90=("forecast_p90", lambda values: values.sum(min_count=len(values))),
        forecast_p95=("forecast_p95", lambda values: values.sum(min_count=len(values))),
        forecast_p99=("forecast_p99", lambda values: values.sum(min_count=len(values))),
        criticality=("criticality", "first"),
    )
    grouped["week"] = grouped["horizon_step"].astype(int)
    grouped["origin_date"] = pd.to_datetime(grouped["origin_date"])
    return grouped


def load_comparison_paths(root: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load only saved evaluation forecasts and retain matched complete paths."""
    forecast_path = root / "artifacts" / "track_a_backtest_forecasts_v41.parquet"
    selection_path = root / "results" / "track_a_v41_selection_matrix.csv"
    forecasts = pd.read_parquet(forecast_path)
    if forecasts["evaluation_split"].ne("evaluation").any():
        raise ValueError("Phase 6 accepts held-validation rows only")
    if pd.to_datetime(forecasts["week_start_date"]).max() > pd.Timestamp(
        "2026-03-23"
    ):
        raise ValueError("Validation forecast artifact extends beyond train-validation")
    selection = pd.read_csv(selection_path)
    eligible = set(
        selection.loc[
            selection["eligible_for_weighted_selection"].astype(bool),
            "candidate_id",
        ].astype(str)
    )
    if not set(POINT_CANDIDATES).issubset(eligible):
        raise ValueError("Frozen comparison candidates are not both eligible")

    model_names = {
        POINT_CANDIDATES[0]: "lightgbm_tweedie",
        POINT_CANDIDATES[1]: "lightgbm_quantile",
    }
    average = _aggregate_model(forecasts, "historical_average").rename(
        columns={
            "forecast_qty": "historical_average_qty",
            "forecast_p90": "historical_average_p90",
            "forecast_p95": "historical_average_p95",
            "forecast_p99": "historical_average_p99",
        }
    )
    quantiles = _aggregate_model(forecasts, "lightgbm_quantile")[
        KEYS + ["forecast_p90", "forecast_p95", "forecast_p99"]
    ]
    paths: dict[str, pd.DataFrame] = {}
    exclusions: list[dict[str, Any]] = []
    path_keys = ["origin_date", "part_id"]
    for candidate_id, model in model_names.items():
        candidate = _aggregate_model(forecasts, model).rename(
            columns={"forecast_qty": "candidate_forecast_qty"}
        ).drop(columns=["forecast_p90", "forecast_p95", "forecast_p99"])
        joined = candidate.merge(
            quantiles,
            on=KEYS,
            how="left",
            validate="one_to_one",
        ).merge(
            average[KEYS + ["historical_average_qty"]],
            on=KEYS,
            how="left",
            validate="one_to_one",
        )
        joined["forecast_qty"] = joined["candidate_forecast_qty"]
        counts = joined.groupby(path_keys, observed=True).agg(
            weeks=("week", "nunique"),
            actual_complete=("actual_qty", lambda values: values.notna().all()),
            candidate_complete=("candidate_forecast_qty", lambda values: values.notna().all()),
            quantiles_complete=("forecast_p99", lambda values: values.notna().all()),
            baseline_complete=("historical_average_qty", lambda values: values.notna().all()),
        )
        valid = (
            counts["weeks"].eq(14)
            & counts["actual_complete"]
            & counts["candidate_complete"]
            & counts["quantiles_complete"]
            & counts["baseline_complete"]
        )
        for (origin, part_id), row in counts.loc[~valid].iterrows():
            reasons = []
            if row["weeks"] != 14:
                reasons.append("incomplete_14_week_path")
            if not row["actual_complete"]:
                reasons.append("missing_actual")
            if not row["candidate_complete"]:
                reasons.append("missing_candidate_prediction")
            if not row["quantiles_complete"]:
                reasons.append("missing_quantile_prediction")
            if not row["baseline_complete"]:
                reasons.append("missing_historical_average_prediction")
            exclusions.append(
                {
                    "candidate_id": candidate_id,
                    "origin_date": origin,
                    "part_id": part_id,
                    "exclusion_reason": "|".join(reasons),
                }
            )
        valid_keys = counts.loc[valid].reset_index()[path_keys]
        paths[candidate_id] = (
            joined.merge(valid_keys, on=path_keys, how="inner", validate="many_to_one")
            .sort_values(KEYS, kind="stable")
            .reset_index(drop=True)
        )
    key_sets = {
        candidate_id: set(
            map(
                tuple,
                frame[path_keys].itertuples(index=False, name=None),
            )
        )
        for candidate_id, frame in paths.items()
    }
    common = set.intersection(*key_sets.values()) if key_sets else set()
    for candidate_id, frame in paths.items():
        candidate_keys = key_sets[candidate_id]
        for origin, part_id in candidate_keys - common:
            exclusions.append(
                {
                    "candidate_id": candidate_id,
                    "origin_date": origin,
                    "part_id": part_id,
                    "exclusion_reason": "not_in_common_candidate_cohort",
                }
            )
        common_frame = frame.loc[
            [
                key in common
                for key in frame[path_keys].itertuples(index=False, name=None)
            ],
            path_keys,
        ].drop_duplicates()
        paths[candidate_id] = frame.merge(
            common_frame, on=path_keys, how="inner", validate="many_to_one"
        )
    return paths, pd.DataFrame(exclusions)


def _initial_snapshot(
    part: dict[str, Any],
    origin: pd.Timestamp,
    histories: Mapping[tuple[str, str], Mapping[str, Any]],
    initialization: str,
) -> dict[str, Any]:
    snapshot = dict(part)
    if initialization == PRIMARY_INITIALIZATION:
        key = (origin.date().isoformat(), str(part["part_id"]))
        stock = origin_visible_opening_stock(part, histories[key])
        available = int(stock["opening_available_qty"])
        snapshot.update(
            {
                "on_hand_qty": available,
                "reserved_qty": 0,
                "blocked_qty": 0,
                "available_qty": available,
                "backorder_qty": 0,
                "in_transit_qty": 0,
                "open_po_qty": 0,
                "open_po_due_this_month_qty": 0,
                "open_po_due_later_qty": 0,
            }
        )
    elif initialization != D2_STRESS_INITIALIZATION:
        raise ValueError(f"Unknown initialization: {initialization}")
    calculate_usable_stock(snapshot)
    return snapshot


def _checker_parts(
    part: dict[str, Any], risk_adjusted: bool
) -> dict[str, dict[str, Any]]:
    checked = dict(part)
    if risk_adjusted:
        for option in supplier_options(part, risk_adjusted=True):
            slot = int(option["slot"])
            checked[f"sup{slot}_lead_time_days"] = option[
                "effective_lead_time_days"
            ]
    return {str(part["part_id"]): checked}


def _traditional_baseline(
    part: dict[str, Any],
    path: pd.DataFrame,
    *,
    run_id: str,
    horizon: int,
    due_later_week: int,
    capacity_used: Mapping[Any, float] | None = None,
) -> dict[str, Any]:
    """D2 ROP policy; order to D2 ROP when projected lead-time position breaches it."""
    stock = calculate_usable_stock(part)
    lead_days = sum(
        float(option["sob_pct"]) * float(option["lead_time_days"])
        for option in supplier_options(part)
    ) / 100.0
    lead_weeks = max(1, int(ceil(lead_days / 7.0)))
    reorder_point = int(part["current_reorder_point_qty"])
    receipts: dict[int, int] = {}
    for receipt in snapshot_receipts(part, due_later_week):
        receipts[receipt["week"]] = receipts.get(receipt["week"], 0) + receipt["quantity"]
    queue: dict[int, int] = {}
    capacity: dict[Any, float] = dict(capacity_used or {})
    orders: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    backlog = int(part["backorder_qty"]) if part["criticality"] == "High" else 0
    for item in path.sort_values("week").itertuples(index=False):
        week = int(item.week)
        opening = stock
        arrived = receipts.get(week, 0) + queue.pop(week, 0)
        stock += arrived
        demand_window = path.loc[
            path["week"].between(week, min(horizon, week + lead_weeks - 1)),
            "historical_average_qty",
        ].astype(float)
        future_receipts = sum(
            amount for due, amount in queue.items() if week < due <= week + lead_weeks
        )
        projected_position = stock + future_receipts - float(demand_window.sum())
        net = max(0, reorder_point - projected_position)
        allocation = allocate_supplier_orders(
            part,
            int(ceil(net)),
            release_week=week,
            capacity_used=capacity,
            horizon_weeks=horizon,
            capacity_period=pd.Timestamp(item.week_start_date),
            risk_adjusted=False,
        )
        capacity = allocation.get("capacity_used", capacity)
        for order in allocation["orders"]:
            row = {
                **order,
                "run_id": run_id,
                "region": "ALL",
                "requested_week": week,
                "net_requirement": net,
                "reason": "historical_average_forecast_d2_reorder_point",
                "late": order["arrival_week"] > week + lead_weeks,
                "warnings": "",
            }
            orders.append(row)
            queue[row["arrival_week"]] = queue.get(row["arrival_week"], 0) + int(
                row["order_qty"]
            )
        demand = int(ceil(float(item.historical_average_qty)))
        backlog_fulfilled = min(stock, backlog) if part["criticality"] == "High" else 0
        stock -= backlog_fulfilled
        remaining_backlog = backlog - backlog_fulfilled
        fulfilled = min(stock, demand)
        stock -= fulfilled
        shortage = demand - fulfilled
        backlog = remaining_backlog + shortage if part["criticality"] == "High" else 0
        weekly_rows.append(
            {
                "run_id": run_id,
                "part_id": part["part_id"],
                "region": "ALL",
                "week": week,
                "week_start_date": item.week_start_date,
                "forecast_qty": float(item.historical_average_qty),
                "opening_usable_stock": opening,
                "expected_receipts": arrived,
                "projected_inventory": stock,
                "backlog_qty": backlog,
                "net_requirement": net,
                "safety_stock": 0.0,
                "reorder_point": reorder_point,
                "service_level_target": float(part["service_level_target"]),
                "criticality": part["criticality"],
                "fulfilled_qty": backlog_fulfilled + fulfilled,
                "lost_sales_qty": shortage if part["criticality"] != "High" else 0,
                "inventory_conservation": stock
                == opening + arrived - backlog_fulfilled - fulfilled,
                "status": "unresolved" if allocation["failed_checks"] else "firm",
                "failed_checks": "|".join(allocation["failed_checks"]),
                "reorder_trigger_week": week if net else np.nan,
            }
        )
    return {
        "orders": pd.DataFrame(orders),
        "weekly": pd.DataFrame(weekly_rows),
        "status": (
            "unresolved"
            if any(row["failed_checks"] for row in weekly_rows)
            else "firm"
        ),
        "failed_checks": sorted(
            {
                reason
                for row in weekly_rows
                for reason in str(row["failed_checks"]).split("|")
                if reason
            }
        ),
        "reorder_point": reorder_point,
        "capacity_used": capacity,
    }


def _run_plan(
    candidate_id: str,
    scheduler: str,
    initialization: str,
    path: pd.DataFrame,
    part: dict[str, Any],
    histories: Mapping[tuple[str, str], Mapping[str, Any]],
    config: dict[str, Any],
    capacity_used: Mapping[Any, float] | None = None,
    horizon: int = 14,
    run_id_prefix: str = "phase6",
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any], dict[Any, float]]:
    origin = pd.Timestamp(path["origin_date"].iloc[0])
    snapshot = _initial_snapshot(part, origin, histories, initialization)
    if horizon < 1:
        raise ValueError("Planning horizon must be positive")
    run_id = f"{run_id_prefix}:{candidate_id}:{scheduler}:{initialization}:{origin:%Y-%m-%d}:{part['part_id']}"
    due_later_week = int(
        config["scheduling"]["simulator_assumptions"]["inbound_receipts"][
            "due_later_week"
        ]
    )
    if scheduler == "traditional_baseline":
        plan = _traditional_baseline(
            snapshot,
            path,
            run_id=run_id,
            horizon=horizon,
            due_later_week=due_later_week,
            capacity_used=capacity_used,
        )
    else:
        forecasts = path[
            [
                "week",
                "week_start_date",
                "forecast_qty",
                "forecast_p90",
                "forecast_p95",
                "forecast_p99",
            ]
        ].copy()
        plan = build_mrp_schedule(
            snapshot,
            forecasts,
            run_id=run_id,
            horizon_weeks=horizon,
            due_later_week=due_later_week,
            risk_adjusted=scheduler == "risk_adjusted_mrp",
            capacity_used=capacity_used,
        )
    orders = plan["orders"].copy()
    if not orders.empty:
        orders["origin_date"] = origin
        orders["requested_week"] = orders.get(
            "requested_week", orders.get("release_week")
        )
    weekly = plan["weekly"].copy()
    requirements = (
        weekly.loc[:, ["part_id", "run_id", "week", "net_requirement"]]
        .rename(columns={"week": "release_week"})
        if not weekly.empty
        else pd.DataFrame(columns=["part_id", "run_id", "release_week", "net_requirement"])
    )
    checker = check_schedule(
        orders,
        _checker_parts(snapshot, scheduler == "risk_adjusted_mrp"),
        horizon_weeks=horizon,
        requirements=requirements,
    )
    suppliers = _supplier_snapshot(snapshot)
    realized_orders = realize_supplier_orders(
        orders,
        suppliers,
        seed=int(config["scheduling"]["simulator_assumptions"]["seed"]),
    )
    receipts = []
    if initialization == D2_STRESS_INITIALIZATION:
        receipts.extend(snapshot_receipts(snapshot, due_later_week))
    for order in realized_orders.to_dict(orient="records"):
        if order["received_qty"] > 0:
            receipts.append(
                {
                    "week": int(order["realized_arrival_week"]),
                    "quantity": int(order["received_qty"]),
                }
            )
    simulator = config["scheduling"]["simulator_assumptions"]
    carrying = snapshot.get("carrying_cost_rate")
    if carrying is None or pd.isna(carrying):
        carrying = simulator["carrying_cost"]["fallback_annual_rate"]
    simulation = simulate_weekly(
        initial_stock=int(snapshot["available_qty"]),
        demand=path.sort_values("week")["actual_qty"].astype(float).tolist(),
        receipts=receipts,
        criticality=str(snapshot["criticality"]),
        initial_backlog=int(snapshot["backorder_qty"]),
        unit_cost=float(snapshot["unit_cost"]),
        carrying_cost_rate=float(carrying),
        emergency_multiplier=float(
            simulator["emergency_procurement"]["unit_cost_multiplier"]
        ),
        stockout_multipliers=simulator["stockout_cost"]["criticality_multiplier"],
    )
    simulation = simulation.merge(
        weekly[
            ["week", "safety_stock", "reorder_point", "inventory_conservation"]
        ].rename(columns={"inventory_conservation": "mrp_inventory_conservation"}),
        on="week",
        how="left",
        validate="one_to_one",
    )
    simulation["excess_inventory_units"] = (
        simulation["closing_stock"] - simulation["safety_stock"].fillna(0)
    ).clip(lower=0)
    simulation["run_id"] = run_id
    simulation["origin_date"] = origin
    simulation["part_id"] = str(part["part_id"])
    simulation["candidate_id"] = candidate_id
    simulation["scheduler_id"] = scheduler
    simulation["initialization"] = initialization
    simulation["criticality"] = str(part["criticality"])
    simulation["unit_cost"] = float(part["unit_cost"])
    ledger = check_inventory_ledger(
        simulation,
        "opening_stock",
        "receipts_qty",
        "fulfilled_qty",
        "closing_stock",
    )
    failures = set(plan.get("failed_checks", []))
    failures.update(checker["failed_checks"])
    failures.update(ledger["failed_checks"])
    failures = sorted(failures)
    status = (
        "unresolved"
        if failures or not checker["feasible"] or not ledger["feasible"]
        else "firm"
    )
    if not weekly.empty:
        weekly["status"] = status
        weekly["failed_checks"] = "|".join(failures)
        weekly["origin_date"] = origin
        weekly["candidate_id"] = candidate_id
        weekly["scheduler_id"] = scheduler
        weekly["initialization"] = initialization
    actual = path.sort_values("week")["actual_qty"].astype(float).to_numpy()
    predicted = (
        path.sort_values("week")["historical_average_qty"].astype(float).to_numpy()
        if scheduler == "traditional_baseline"
        else path.sort_values("week")["forecast_qty"].astype(float).to_numpy()
    )
    forecast_abs_error = np.abs(predicted - actual)
    actual_sum = float(actual.sum())
    forecast_metrics = {
        "forecast_wape": (
            float(forecast_abs_error.sum() / actual_sum)
            if actual_sum > 0
            else np.nan
        ),
        "forecast_accuracy_pct": (
            float(100 * (1 - forecast_abs_error.sum() / actual_sum))
            if actual_sum > 0
            else np.nan
        ),
        "forecast_bias": (
            float((predicted - actual).sum() / actual_sum)
            if actual_sum > 0
            else np.nan
        ),
        "forecast_variance_signed_units": float((predicted - actual).sum()),
        "forecast_variance_absolute_units": float(forecast_abs_error.sum()),
        "forecast_bias_abs": abs(
            float((predicted - actual).sum() / actual_sum)
        )
        if actual_sum > 0
        else np.nan,
    }
    demand_value = actual_sum * float(snapshot["unit_cost"])
    avg_inventory_value = float(simulation["closing_stock"].mean()) * float(
        snapshot["unit_cost"]
    )
    total_orders = float(orders["order_qty"].sum()) if not orders.empty else 0.0
    on_time_qty = (
        float(
            realized_orders.loc[
                realized_orders["realized_arrival_week"].le(
                    realized_orders["arrival_week"]
                ),
                "received_qty",
            ].sum()
        )
        if not realized_orders.empty
        else 0.0
    )
    current_fulfilled = int(simulation["current_demand_fulfilled_qty"].sum())
    row = {
        "run_id": run_id,
        "candidate_id": candidate_id,
        "scheduler_id": scheduler,
        "initialization": initialization,
        "origin_date": origin,
        "part_id": str(part["part_id"]),
        "criticality": str(part["criticality"]),
        "firm": status == "firm",
        "status": status,
        "failed_checks": "|".join(failures),
        "checker_feasible": bool(checker["feasible"]),
        "hard_constraint_violation": bool(
            status == "firm" and not (checker["feasible"] and ledger["feasible"])
        ),
        "checker_failed_checks": "|".join(checker["failed_checks"]),
        "ledger_feasible": bool(ledger["feasible"]),
        "ledger_failed_checks": "|".join(ledger["failed_checks"]),
        "actual_demand_units": actual_sum,
        "fulfilled_units": current_fulfilled,
        "stockout_part_weeks": int(simulation["stockout_at_close_flag"].sum()),
        "part_weeks": int(len(simulation)),
        "emergency_order_count": int(simulation["emergency_order_qty"].gt(0).sum()),
        "emergency_quantity": int(simulation["emergency_order_qty"].sum()),
        "emergency_cost_proxy": float(simulation["emergency_cost_proxy"].sum()),
        "excess_inventory_units": float(simulation["excess_inventory_units"].sum()),
        "carrying_cost_proxy": float(simulation["carrying_cost_proxy"].sum()),
        "stockout_cost_proxy": float(simulation["stockout_cost_proxy"].sum()),
        "critical_available_weeks": (
            int((~simulation["stockout_at_close_flag"]).sum())
            if part["criticality"] == "High"
            else np.nan
        ),
        "critical_total_weeks": len(simulation) if part["criticality"] == "High" else np.nan,
        "scheduled_order_qty": total_orders,
        "on_time_received_qty": on_time_qty,
        "turnover_14_week": (
            demand_value / avg_inventory_value
            if avg_inventory_value > 0
            else np.nan
        ),
        "phase_out_stock_value_before": (
            float(snapshot["available_qty"]) * float(snapshot["unit_cost"])
            if str(snapshot["lifecycle_status"]).casefold() == "phase-out"
            else 0.0
        ),
        "phase_out_stock_value_after": (
            float(simulation["closing_stock"].iloc[-1]) * float(snapshot["unit_cost"])
            if str(snapshot["lifecycle_status"]).casefold() == "phase-out"
            else 0.0
        ),
        "phase_out_stock_qty_before": (
            float(snapshot["available_qty"])
            if str(snapshot["lifecycle_status"]).casefold() == "phase-out"
            else 0.0
        ),
        "phase_out_stock_qty_after": (
            float(simulation["closing_stock"].iloc[-1])
            if str(snapshot["lifecycle_status"]).casefold() == "phase-out"
            else 0.0
        ),
        "phase_out_average_weekly_demand": (
            actual_sum / horizon
            if str(snapshot["lifecycle_status"]).casefold() == "phase-out"
            else 0.0
        ),
        **forecast_metrics,
    }
    return row, orders, simulation, checker, plan.get("capacity_used", {})


def _weighted_score(rows: pd.DataFrame, weights: Mapping[str, float]) -> pd.Series:
    metrics = {
        "critical_part_availability": ("critical_availability", True),
        "overall_fill_rate_and_stockout": ("fill_rate", True),
        "emergency_procurement": ("emergency_cost_proxy", False),
        "excess_inventory_and_carrying_cost": ("inventory_cost", False),
        "schedule_adherence_and_stability": ("schedule_adherence", True),
        "inventory_turnover": ("turnover_14_week", True),
    }
    scores = pd.DataFrame(index=rows.index)
    for name, (column, higher_better) in metrics.items():
        values = pd.to_numeric(rows[column], errors="coerce")
        ranks = values.rank(method="average", ascending=not higher_better, pct=True)
        scores[name] = ranks.fillna(0.5)
    applicable = [name for name in metrics if name in weights]
    denominator = sum(float(weights[name]) for name in applicable)
    return sum(scores[name] * float(weights[name]) for name in applicable) / denominator


def _kpi_rows(summary: pd.DataFrame) -> pd.DataFrame:
    definitions = {
        "forecast_wape": ("sum absolute forecast errors / sum actual", "ratio", "measured", "actual demand units"),
        "forecast_accuracy_pct": ("100 x (1 - WAPE)", "percent", "measured", "actual demand units"),
        "forecast_variance_signed_units": ("sum forecast - actual", "units", "measured", "matched part-origin-weeks"),
        "forecast_variance_absolute_units": ("sum absolute forecast - actual", "units", "measured", "matched part-origin-weeks"),
        "forecast_bias": ("sum forecast - actual / sum actual", "ratio", "measured", "actual demand units"),
        "forecast_bias_abs": ("absolute signed forecast bias", "ratio", "measured", "actual demand units"),
        "critical_part_availability": ("High-criticality available part-weeks / High-criticality part-weeks", "ratio", "simulated", "High-criticality part-weeks"),
        "stock_out_pct": ("part-weeks with shortage / eligible part-weeks", "ratio", "simulated", "eligible part-weeks"),
        "fill_rate": ("fulfilled current demand / actual demand", "ratio", "simulated", "actual demand units"),
        "schedule_adherence": ("received order units arriving by planned week / planned order units", "ratio", "simulated", "planned order units"),
        "emergency_order_count": ("number of simulated emergency order events", "count", "simulated", "emergency order events"),
        "emergency_cost_proxy": ("emergency units x unit cost x configured multiplier", "currency", "proxy", "emergency order units"),
        "excess_inventory_units": ("sum max(closing stock - safety stock, 0)", "units", "simulated", "part-weeks"),
        "carrying_cost_proxy": ("sum closing inventory value x annual rate / 52", "currency", "proxy", "part-weeks"),
        "turnover_14_week": ("14-week COGS / average closing inventory value", "ratio", "simulated", "average inventory value"),
        "emergency_procurement_reduction": ("traditional baseline emergency cost proxy - scheduler emergency cost proxy", "currency", "proxy", "emergency procurement cost proxy"),
        "excess_inventory_reduction": ("traditional baseline excess inventory units - scheduler excess inventory units", "units", "simulated", "part-weeks"),
        "carrying_cost_reduction": ("traditional baseline carrying cost proxy - scheduler carrying cost proxy", "currency", "proxy", "part-weeks"),
        "phase_out_stock_value_before": ("sum phase-out usable opening stock x unit cost", "currency", "measured", "phase-out usable opening stock"),
        "phase_out_stock_value_after": ("sum phase-out closing stock x unit cost at horizon end", "currency", "simulated", "phase-out closing stock"),
        "phase_out_weeks_cover_before": ("phase-out usable opening stock / mean weekly actual demand", "weeks", "measured", "phase-out mean weekly demand"),
        "phase_out_weeks_cover_after": ("phase-out closing stock / mean weekly actual demand", "weeks", "simulated", "phase-out mean weekly demand"),
    }
    records = []
    grouping = ["candidate_id", "scheduler_id", "initialization"]
    for keys, group in summary.groupby(grouping, observed=True):
        total_actual = float(group["actual_demand_units"].sum())
        values = {
            "forecast_wape": (
                float(group["forecast_variance_absolute_units"].sum() / total_actual)
                if total_actual
                else np.nan
            ),
            "forecast_accuracy_pct": (
                float(100 * (1 - group["forecast_variance_absolute_units"].sum() / total_actual))
                if total_actual
                else np.nan
            ),
            "forecast_variance_signed_units": float(group["forecast_variance_signed_units"].sum()),
            "forecast_variance_absolute_units": float(
                group["forecast_variance_absolute_units"].sum()
            ),
            "forecast_bias": (
                float(
                    (group["forecast_bias"] * group["actual_demand_units"]).sum()
                    / total_actual
                )
                if total_actual
                else np.nan
            ),
            "forecast_bias_abs": (
                abs(float((group["forecast_bias"] * group["actual_demand_units"]).sum() / total_actual))
                if total_actual
                else np.nan
            ),
            "critical_part_availability": (
                float(group["critical_available_weeks"].sum() / group["critical_total_weeks"].sum())
                if group["critical_total_weeks"].sum() > 0
                else np.nan
            ),
            "stock_out_pct": float(group["stockout_part_weeks"].sum() / group["part_weeks"].sum()),
            "fill_rate": float(group["fulfilled_units"].sum() / total_actual) if total_actual else np.nan,
            "schedule_adherence": (
                float(group["on_time_received_qty"].sum() / group["scheduled_order_qty"].sum())
                if group["scheduled_order_qty"].sum()
                else np.nan
            ),
            "emergency_order_count": float(group["emergency_order_count"].sum()),
            "emergency_cost_proxy": float(group["emergency_cost_proxy"].sum()),
            "excess_inventory_units": float(group["excess_inventory_units"].sum()),
            "carrying_cost_proxy": float(group["carrying_cost_proxy"].sum()),
            "turnover_14_week": float(group["turnover_14_week"].mean()),
            "phase_out_stock_value_before": float(
                group["phase_out_stock_value_before"].sum()
            ),
            "phase_out_stock_value_after": float(
                group["phase_out_stock_value_after"].sum()
            ),
            "phase_out_weeks_cover_before": (
                float(
                    group["phase_out_stock_qty_before"].sum()
                    / group["phase_out_average_weekly_demand"].sum()
                )
                if group["phase_out_average_weekly_demand"].sum() > 0
                else np.nan
            ),
            "phase_out_weeks_cover_after": (
                float(
                    group["phase_out_stock_qty_after"].sum()
                    / group["phase_out_average_weekly_demand"].sum()
                )
                if group["phase_out_average_weekly_demand"].sum() > 0
                else np.nan
            ),
        }
        base = {
            "candidate_id": keys[0],
            "scheduler_id": keys[1],
            "initialization": keys[2],
        }
        for metric, value in values.items():
            formula, unit, basis, denominator = definitions[metric]
            records.append(
                {
                    **base,
                    "kpi": metric,
                    "formula": formula,
                    "unit": unit,
                    "basis": basis,
                    "denominator": denominator,
                    "value": value,
                }
            )
    result = pd.DataFrame(records)
    obsolete_missing_records = []
    for base in summary[grouping].drop_duplicates().to_dict("records"):
        for metric, unit in (
            ("obsolete_stock_value_before", "currency"),
            ("obsolete_stock_value_after", "currency"),
            ("obsolete_weeks_cover_before", "weeks"),
            ("obsolete_weeks_cover_after", "weeks"),
        ):
            obsolete_missing_records.append(
                {
                    **base,
                    "kpi": metric,
                    "formula": (
                        "not measurable: obsolete parts P0046 and P0087 have no D2 scheduling snapshot"
                    ),
                    "unit": unit,
                    "basis": "not_measurable",
                    "denominator": "obsolete parts absent from D2",
                    "value": np.nan,
                }
            )
    result = pd.concat(
        [result, pd.DataFrame(obsolete_missing_records)], ignore_index=True
    )
    reduction_specs = (
        ("emergency_cost_proxy", "emergency_procurement_reduction"),
        ("excess_inventory_units", "excess_inventory_reduction"),
        ("carrying_cost_proxy", "carrying_cost_reduction"),
    )
    reduction_records = []
    for source_kpi, reduction_kpi in reduction_specs:
        source = result.loc[result["kpi"].eq(source_kpi)].set_index(
            ["candidate_id", "initialization", "scheduler_id"]
        )["value"]
        for row in result.loc[result["kpi"].eq(source_kpi)].to_dict("records"):
            baseline_value = source.get(
                (row["candidate_id"], row["initialization"], "traditional_baseline"),
                np.nan,
            )
            reduction_records.append(
                {
                    "candidate_id": row["candidate_id"],
                    "scheduler_id": row["scheduler_id"],
                    "initialization": row["initialization"],
                    "kpi": reduction_kpi,
                    "formula": definitions[reduction_kpi][0],
                    "unit": definitions[reduction_kpi][1],
                    "basis": definitions[reduction_kpi][2],
                    "denominator": definitions[reduction_kpi][3],
                    "value": baseline_value - row["value"]
                    if pd.notna(baseline_value) and pd.notna(row["value"])
                    else np.nan,
                }
            )
    result = pd.concat(
        [result, pd.DataFrame(reduction_records)], ignore_index=True
    )
    keys = ["candidate_id", "initialization", "kpi"]
    baseline = result.loc[
        result["scheduler_id"].eq("traditional_baseline"),
        keys + ["value"],
    ].rename(columns={"value": "baseline_value"})
    result = result.merge(
        baseline,
        on=keys,
        how="left",
        validate="many_to_one",
    )
    result["absolute_difference"] = result["value"] - result["baseline_value"]
    result["percentage_difference"] = np.where(
        result["baseline_value"].notna()
        & result["baseline_value"].ne(0),
        result["absolute_difference"] / result["baseline_value"] * 100.0,
        np.nan,
    )
    result["percentage_difference_status"] = np.where(
        result["baseline_value"].eq(0),
        "not_defined_zero_baseline",
        np.where(result["baseline_value"].notna(), "defined", "not_defined_missing_baseline"),
    )
    lower_better = {
        "forecast_wape",
        "forecast_variance_absolute_units",
        "forecast_bias_abs",
        "stock_out_pct",
        "emergency_order_count",
        "emergency_cost_proxy",
        "excess_inventory_units",
        "carrying_cost_proxy",
    }
    higher_better = {
        "forecast_accuracy_pct",
        "critical_part_availability",
        "fill_rate",
        "schedule_adherence",
        "turnover_14_week",
        "emergency_procurement_reduction",
        "excess_inventory_reduction",
        "carrying_cost_reduction",
    }
    result["worse_than_baseline"] = np.where(
        result["kpi"].isin(lower_better),
        result["absolute_difference"].gt(0),
        np.where(
            result["kpi"].isin(higher_better),
            result["absolute_difference"].lt(0),
            False,
        ),
    )
    return result


def _paired_bootstrap(
    summary: pd.DataFrame, seed: int, replicates: int = 2000
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    metrics = (
        "fill_rate",
        "critical_availability",
        "stock_out_pct",
        "forecast_wape",
        "forecast_bias",
        "schedule_adherence",
        "emergency_cost_proxy",
        "excess_inventory_units",
        "carrying_cost_proxy",
        "turnover_14_week",
    )
    records = []
    for (candidate, initialization), cohort in summary.groupby(
        ["candidate_id", "initialization"], observed=True
    ):
        cohort = cohort.copy()
        cohort["part_origin_id"] = (
            pd.to_datetime(cohort["origin_date"]).dt.strftime("%Y-%m-%d")
            + ":"
            + cohort["part_id"].astype(str)
        )
        pivoted = {
            metric: cohort.pivot(
                index="part_origin_id", columns="scheduler_id", values=metric
            )
            for metric in metrics
        }
        for metric, pivot in pivoted.items():
            if "traditional_baseline" not in pivot:
                continue
            for scheduler in ("mrp", "risk_adjusted_mrp"):
                if scheduler not in pivot:
                    continue
                paired = pivot[["traditional_baseline", scheduler]].dropna()
                if paired.empty:
                    continue
                differences = (
                    paired[scheduler] - paired["traditional_baseline"]
                ).to_numpy(dtype=float)
                draws = rng.integers(0, len(differences), size=(replicates, len(differences)))
                means = differences[draws].mean(axis=1)
                records.append(
                    {
                        "candidate_id": candidate,
                        "initialization": initialization,
                        "scheduler_id": scheduler,
                        "baseline_scheduler": "traditional_baseline",
                        "kpi": metric,
                        "paired_part_origins": len(differences),
                        "mean_difference": float(differences.mean()),
                        "ci95_low": float(np.quantile(means, 0.025)),
                        "ci95_high": float(np.quantile(means, 0.975)),
                        "seed": seed,
                        "replicates": replicates,
                    }
                )
    return pd.DataFrame(records)


def _sensitivity_scores(
    summary: pd.DataFrame, weights: Mapping[str, float]
) -> pd.DataFrame:
    cohort = summary.groupby(
        ["candidate_id", "scheduler_id", "initialization"], as_index=False,
        observed=True,
    ).agg(
        critical_availability=("critical_available_weeks", "sum"),
        critical_weeks=("critical_total_weeks", "sum"),
        fulfilled_units=("fulfilled_units", "sum"),
        actual_demand_units=("actual_demand_units", "sum"),
        emergency_cost_proxy=("emergency_cost_proxy", "sum"),
        excess_inventory_units=("excess_inventory_units", "sum"),
        carrying_cost_proxy=("carrying_cost_proxy", "sum"),
        on_time_received_qty=("on_time_received_qty", "sum"),
        scheduled_order_qty=("scheduled_order_qty", "sum"),
        turnover_14_week=("turnover_14_week", "mean"),
        firm_paths=("firm", "sum"),
        total_paths=("firm", "size"),
        hard_constraint_violation=(
            "hard_constraint_violation",
            "any",
        ),
    )
    cohort["critical_availability"] = cohort["critical_availability"] / cohort[
        "critical_weeks"
    ]
    cohort["fill_rate"] = cohort["fulfilled_units"] / cohort["actual_demand_units"]
    cohort["schedule_adherence"] = cohort["on_time_received_qty"] / cohort[
        "scheduled_order_qty"
    ].replace(0, np.nan)
    cohort["inventory_cost"] = cohort["excess_inventory_units"] + cohort[
        "carrying_cost_proxy"
    ]
    cohort["score"] = _weighted_score(cohort, weights)
    # Unresolved paths remain visible but do not invalidate feasible firm orders.
    cohort.loc[cohort["hard_constraint_violation"], "score"] = -np.inf
    cohort["rank"] = cohort.groupby("initialization")["score"].rank(
        method="min", ascending=False
    )
    return cohort


def run_phase6(root: Path = ROOT) -> dict[str, Any]:
    """Run the saved-validation comparison and write new, no-overwrite outputs."""
    for relative in OUTPUTS:
        if (root / relative).exists():
            raise FileExistsError(
                f"Phase 6 output already exists; inspect before proceeding: {root / relative}"
            )
    config = _config(root)
    config_hash = _sha256(root / "configs" / "project.yaml")
    paths_by_candidate, exclusions = load_comparison_paths(root)
    scheduling = load_scheduling(root / "data" / "02_scheduling_inputs.csv")
    parts = {
        str(row["part_id"]): row for row in scheduling.to_dict(orient="records")
    }
    origins = sorted(
        pd.Timestamp(value)
        for frame in paths_by_candidate.values()
        for value in frame["origin_date"].drop_duplicates()
    )
    histories = load_origin_visible_history(root, origins)
    if paths_by_candidate:
        cohort = paths_by_candidate[POINT_CANDIDATES[0]][
            ["origin_date", "part_id"]
        ].drop_duplicates()
    else:
        cohort = pd.DataFrame(columns=["origin_date", "part_id"])
    valid_path_keys: set[tuple[pd.Timestamp, str]] = set()
    for row in cohort.itertuples(index=False):
        origin = pd.Timestamp(row.origin_date)
        part_id = str(row.part_id)
        reasons = []
        if part_id not in parts:
            reasons.append("missing_D2_scheduling_snapshot")
        history = histories.get((origin.date().isoformat(), part_id))
        if history is None:
            reasons.append("missing_origin_visible_history")
        elif int(history["complete_observed_weeks"]) < 8:
            reasons.append("insufficient_origin_visible_history")
        if reasons:
            for candidate_id in POINT_CANDIDATES:
                exclusions.loc[len(exclusions)] = {
                    "candidate_id": candidate_id,
                    "origin_date": origin,
                    "part_id": part_id,
                    "exclusion_reason": "|".join(reasons),
                }
        else:
            valid_path_keys.add((origin, part_id))
    for candidate_id, frame in paths_by_candidate.items():
        paths_by_candidate[candidate_id] = frame.loc[
            [
                (pd.Timestamp(origin), str(part_id)) in valid_path_keys
                for origin, part_id in zip(frame["origin_date"], frame["part_id"])
            ]
        ].copy()
    result_rows: list[dict[str, Any]] = []
    order_frames: list[pd.DataFrame] = []
    simulation_frames: list[pd.DataFrame] = []
    checker_rows: list[dict[str, Any]] = []
    path_membership: set[tuple[str, str, str]] = set()
    for candidate_id, candidate_paths in paths_by_candidate.items():
        for origin, origin_paths in candidate_paths.groupby(
            "origin_date", sort=True, observed=True
        ):
            for initialization in INITIALIZATIONS:
                for scheduler in SCHEDULERS:
                    capacity: dict[Any, float] = {}
                    origin_order_frames: list[pd.DataFrame] = []
                    origin_row_indexes: list[int] = []
                    for part_id, path in origin_paths.groupby(
                        "part_id", sort=True, observed=True
                    ):
                        path_membership.add(
                            (
                                candidate_id,
                                pd.Timestamp(origin).date().isoformat(),
                                str(part_id),
                            )
                        )
                        row, orders, simulation, checker, capacity = _run_plan(
                            candidate_id,
                            scheduler,
                            initialization,
                            path,
                            parts[str(part_id)],
                            histories,
                            config,
                            capacity,
                        )
                        row["config_sha256"] = config_hash
                        result_rows.append(row)
                        origin_row_indexes.append(len(result_rows) - 1)
                        if not orders.empty:
                            order_frames.append(orders)
                            origin_order_frames.append(orders)
                        simulation_frames.append(simulation)
                        checker_rows.append(
                            {
                                "candidate_id": candidate_id,
                                "scheduler_id": scheduler,
                                "initialization": initialization,
                                "run_id": row["run_id"],
                                "part_id": part_id,
                                "status": row["status"],
                                "firm": row["firm"],
                                "hard_constraint_violation": row[
                                    "hard_constraint_violation"
                                ],
                                "checker_feasible": row["checker_feasible"],
                                "checker_failed_checks": row["checker_failed_checks"],
                                "ledger_feasible": row["ledger_feasible"],
                                "ledger_failed_checks": row["ledger_failed_checks"],
                            }
                        )
                    combined_orders = (
                        pd.concat(origin_order_frames, ignore_index=True)
                        if origin_order_frames
                        else pd.DataFrame()
                    )
                    checker_parts = {
                        part_id: _checker_parts(
                            part, scheduler == "risk_adjusted_mrp"
                        )[part_id]
                        for part_id, part in parts.items()
                    }
                    global_checker = check_schedule(
                        combined_orders,
                        checker_parts,
                        horizon_weeks=14,
                        check_requirements=False,
                    )
                    checker_rows.append(
                        {
                            "candidate_id": candidate_id,
                            "scheduler_id": scheduler,
                            "initialization": initialization,
                            "run_id": f"phase6:global:{candidate_id}:{scheduler}:{initialization}:{pd.Timestamp(origin):%Y-%m-%d}",
                            "part_id": "ALL",
                            "status": global_checker["status"],
                            "firm": global_checker["status"] == "firm",
                            "checker_feasible": global_checker["feasible"],
                            "checker_failed_checks": "|".join(
                                global_checker["failed_checks"]
                            ),
                            "ledger_feasible": True,
                            "ledger_failed_checks": "",
                        }
                    )
                    if not global_checker["feasible"]:
                        for index in origin_row_indexes:
                            result_rows[index]["firm"] = False
                            result_rows[index]["status"] = "unresolved"
                            result_rows[index]["hard_constraint_violation"] = False
                            failures = set(
                                filter(
                                    None,
                                    str(result_rows[index]["failed_checks"]).split("|"),
                                )
                            )
                            failures.update(global_checker["failed_checks"])
                            result_rows[index]["failed_checks"] = "|".join(
                                sorted(failures)
                            )
                            for checker_row in checker_rows:
                                if checker_row["run_id"] == result_rows[index]["run_id"]:
                                    checker_row["status"] = "unresolved"
                                    checker_row["firm"] = False
                                    checker_row["checker_feasible"] = False
                                    checker_row["hard_constraint_violation"] = False
                                    checker_row["checker_failed_checks"] = "|".join(
                                        sorted(
                                            set(
                                                filter(
                                                    None,
                                                    str(
                                                        checker_row[
                                                            "checker_failed_checks"
                                                        ]
                                                    ).split("|"),
                                                )
                                            )
                                            | set(global_checker["failed_checks"])
                                        )
                                    )
    summary = pd.DataFrame(result_rows)
    if summary.empty:
        raise ValueError("No complete, schedulable Phase 6 paths")
    summary["critical_availability"] = np.where(
        summary["critical_total_weeks"].gt(0),
        summary["critical_available_weeks"] / summary["critical_total_weeks"],
        np.nan,
    )
    summary["fill_rate"] = summary["fulfilled_units"] / summary[
        "actual_demand_units"
    ].replace(0, np.nan)
    summary["schedule_adherence"] = summary["on_time_received_qty"] / summary[
        "scheduled_order_qty"
    ].replace(0, np.nan)
    summary["stock_out_pct"] = summary["stockout_part_weeks"] / summary[
        "part_weeks"
    ]
    summary["inventory_cost"] = summary["excess_inventory_units"] + summary[
        "carrying_cost_proxy"
    ]
    weights = config["scoring"]["tracks"]["scheduling"]["weights"]
    cohort = _sensitivity_scores(summary, weights)
    summary = summary.merge(
        cohort[
            [
                "candidate_id",
                "scheduler_id",
                "initialization",
                "score",
                "rank",
            ]
        ],
        on=["candidate_id", "scheduler_id", "initialization"],
        how="left",
        validate="many_to_one",
    )
    kpis = _kpi_rows(summary)
    paired = _paired_bootstrap(
        summary,
        seed=int(config["scheduling"]["simulator_assumptions"]["seed"]),
    )
    primary = cohort.loc[
        cohort["initialization"].eq(PRIMARY_INITIALIZATION)
        & ~cohort["hard_constraint_violation"]
    ]
    if primary.empty:
        raise RuntimeError(
            "No Phase 6 scheduler passed the independent checker under primary initialization"
        )
    winner = primary.sort_values(
        ["rank", "candidate_id", "scheduler_id"], kind="stable"
    ).iloc[0]
    sensitivity_rows = []
    perturb = float(config["scoring"].get("weight_sensitivity_points", 10)) / 100.0
    for name, base_weight in weights.items():
        for direction in (-1, 1):
            changed = dict(weights)
            changed[name] = max(0.0, float(base_weight) + direction * perturb)
            total = sum(changed.values())
            changed = {key: value / total for key, value in changed.items()}
            rescored = _sensitivity_scores(summary, changed)
            eligible = rescored.loc[
                rescored["initialization"].eq(PRIMARY_INITIALIZATION)
                & ~rescored["hard_constraint_violation"]
            ].sort_values(["score", "candidate_id", "scheduler_id"], ascending=[False, True, True])
            sensitivity_rows.append(
                {
                    "perturbed_weight": name,
                    "direction": direction,
                    "winner_candidate_id": (
                        eligible.iloc[0]["candidate_id"] if not eligible.empty else None
                    ),
                    "winner_scheduler_id": (
                        eligible.iloc[0]["scheduler_id"] if not eligible.empty else None
                    ),
                    "base_winner_candidate_id": winner["candidate_id"],
                    "base_winner_scheduler_id": winner["scheduler_id"],
                    "winner_changed": bool(
                        eligible.empty
                        or eligible.iloc[0]["candidate_id"] != winner["candidate_id"]
                        or eligible.iloc[0]["scheduler_id"] != winner["scheduler_id"]
                    ),
                    "weights": json.dumps(changed, sort_keys=True),
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    costs = []
    cost_config = config["scheduling"]["simulator_assumptions"]
    emergency_range = cost_config["emergency_procurement"]["sensitivity_range"]
    stockout_base = cost_config["stockout_cost"]["criticality_multiplier"]
    stockout_scale = float(cost_config["stockout_cost"]["relative_sensitivity"])
    for emergency_multiplier in emergency_range:
        for stockout_multiplier in (1 - stockout_scale, 1 + stockout_scale):
            for keys, _group in summary.groupby(
                ["candidate_id", "scheduler_id", "initialization"], observed=True
            ):
                cost = 0.0
                for sim in simulation_frames:
                    matching = sim.loc[
                        sim["candidate_id"].eq(keys[0])
                        & sim["scheduler_id"].eq(keys[1])
                        & sim["initialization"].eq(keys[2])
                    ]
                    if matching.empty:
                        continue
                    cost += float(
                        (
                            matching["emergency_order_qty"]
                            * matching["unit_cost"]
                            * float(emergency_multiplier)
                        ).sum()
                    )
                    cost += float(
                        (
                            matching["unmet_current_demand_qty"]
                            * matching["unit_cost"]
                            * matching["criticality"].map(stockout_base)
                            * stockout_multiplier
                        ).sum()
                    )
                    cost += float(matching["carrying_cost_proxy"].sum())
                costs.append(
                    {
                        "candidate_id": keys[0],
                        "scheduler_id": keys[1],
                        "initialization": keys[2],
                        "emergency_cost_multiplier": emergency_multiplier,
                        "stockout_penalty_multiplier": stockout_multiplier,
                        "combined_proxy_cost": cost,
                        "basis": "proxy",
                    }
                )
    cost_frame = pd.DataFrame(costs)
    initialization = cohort.groupby(
        ["candidate_id", "scheduler_id", "initialization"], as_index=False,
        observed=True,
    ).agg(
        score=("score", "mean"),
        rank=("rank", "min"),
        firm_paths=("firm_paths", "sum"),
        hard_constraint_violation=("hard_constraint_violation", "any"),
        total_paths=("total_paths", "sum"),
    )
    initialization["firm_path_rate"] = (
        initialization["firm_paths"] / initialization["total_paths"]
    )
    checker_summary = pd.DataFrame(checker_rows)
    simulation_frame = pd.concat(simulation_frames, ignore_index=True)
    schedule_orders = pd.concat(order_frames, ignore_index=True) if order_frames else pd.DataFrame()
    forecast_future = pd.read_csv(
        root / "results" / "forecast_future_14w_v41.csv",
        parse_dates=["week_start_date"],
    )
    provisional = _build_provisional_schedule(
        forecast_future,
        str(winner["candidate_id"]),
        str(winner["scheduler_id"]),
        parts,
        config,
        root,
    )
    provisional_checker = provisional.attrs["independent_checker"]
    base = summary.copy()
    base["simulator_outcome_rows"] = len(simulation_frame)
    base["path_coverage"] = len(path_membership)
    base["run_id"] = base["run_id"].astype(str)
    report = _report_text(
        summary,
        winner,
        sensitivity,
        checker_summary,
        exclusions,
        config_hash,
        provisional,
        provisional_checker,
    )
    _write_new_csv(base, root / OUTPUTS[0])
    _write_new_csv(kpis, root / OUTPUTS[1])
    _write_new_csv(sensitivity, root / OUTPUTS[2])
    _write_new_csv(paired, root / OUTPUTS[3])
    _write_new_csv(cost_frame, root / OUTPUTS[4])
    _write_new_csv(initialization, root / OUTPUTS[5])
    _write_new_csv(exclusions, root / OUTPUTS[6])
    _write_new_csv(checker_summary, root / OUTPUTS[7])
    _write_new_csv(provisional, root / OUTPUTS[8])
    _write_new_text(root / OUTPUTS[9], report)
    return {
        "status": "success",
        "candidate_ids": list(POINT_CANDIDATES),
        "scheduler_ids": list(SCHEDULERS),
        "initializations": list(INITIALIZATIONS),
        "comparison_rows": int(len(summary)),
        "winner_candidate_id": str(winner["candidate_id"]),
        "winner_scheduler_id": str(winner["scheduler_id"]),
        "winner_score": float(winner["score"]),
        "phase5_status": "PARTIAL",
        "provisional_schedule_rows": int(len(provisional)),
    }


def _build_provisional_schedule(
    future: pd.DataFrame,
    candidate_id: str,
    scheduler: str,
    parts: Mapping[str, dict[str, Any]],
    config: dict[str, Any],
    root: Path,
) -> pd.DataFrame:
    model = {
        POINT_CANDIDATES[0]: "lightgbm_tweedie",
        POINT_CANDIDATES[1]: "lightgbm_quantile",
    }[candidate_id]
    candidate = future.loc[future["model"].eq(model)].copy()
    quantile = future.loc[future["model"].eq("lightgbm_quantile")].copy()
    if candidate.empty or quantile.empty:
        raise ValueError(f"Missing provisional forecasts for {candidate_id}")
    catalog = pd.read_csv(
        root / "data" / "03_spare_parts_catalogue.csv",
        usecols=["part_id", "part_name"],
        dtype={"part_id": "string", "part_name": "string"},
    ).drop_duplicates("part_id")
    candidate = candidate.merge(
        quantile[
            [
                "part_id",
                "region",
                "week_start_date",
                "forecast_p90",
                "forecast_p95",
                "forecast_p99",
            ]
        ].rename(
            columns={
                "forecast_p90": "safety_forecast_p90",
                "forecast_p95": "safety_forecast_p95",
                "forecast_p99": "safety_forecast_p99",
            }
        ),
        on=["part_id", "region", "week_start_date"],
        how="left",
        validate="one_to_one",
    )
    candidate = candidate.merge(catalog, on="part_id", how="left", validate="many_to_one")
    rows = []
    due_later_week = int(
        config["scheduling"]["simulator_assumptions"]["inbound_receipts"][
            "due_later_week"
        ]
    )
    frozen_weeks = int(
        config["scheduling"]["simulator_assumptions"]
        .get("frozen_window", {})
        .get("weeks", 2)
    )
    capacity: dict[Any, float] = {}
    provisional_orders: list[pd.DataFrame] = []
    provisional_parts: dict[str, dict[str, Any]] = {}
    ordered_groups = sorted(
        candidate.groupby("part_id", sort=True, observed=True),
        key=lambda item: (
            {"High": 0, "Medium": 1, "Low": 2}.get(
                str(parts.get(str(item[0]), {}).get("criticality")), 3
            ),
            str(item[0]),
        ),
    )
    for part_id, group in ordered_groups:
        part = parts.get(str(part_id))
        if part is None:
            continue
        grouped = group.groupby(
            ["week_start_date", "forecast_target_step"], as_index=False, sort=True
        ).agg(
            forecast_qty=("forecast_qty", "sum"),
            forecast_p90=("safety_forecast_p90", "sum"),
            forecast_p95=("safety_forecast_p95", "sum"),
            forecast_p99=("safety_forecast_p99", "sum"),
        ).rename(columns={"forecast_target_step": "week"})
        grouped["week"] = grouped["week"].astype(int)
        if len(grouped) != 14 or grouped[
            ["forecast_qty", "forecast_p90", "forecast_p95", "forecast_p99"]
        ].isna().any().any():
            continue
        capacity_before = dict(capacity)
        plan = build_mrp_schedule(
            part,
            grouped,
            run_id=f"phase6:provisional:{candidate_id}:{scheduler}:{part_id}",
            horizon_weeks=14,
            due_later_week=due_later_week,
            risk_adjusted=scheduler == "risk_adjusted_mrp",
            capacity_used=capacity,
        )
        capacity = plan.get("capacity_used", capacity)
        orders = plan["orders"]
        requirement_frame = plan["weekly"][
            ["part_id", "run_id", "week", "net_requirement"]
        ].rename(columns={"week": "release_week"})
        checker = check_schedule(
            orders,
            _checker_parts(part, scheduler == "risk_adjusted_mrp"),
            horizon_weeks=14,
            requirements=requirement_frame,
        )
        final_status = (
            "firm"
            if plan["status"] == "firm" and checker["feasible"]
            else "unresolved"
        )
        final_failures = set(plan.get("failed_checks", []))
        final_failures.update(checker["failed_checks"])
        if final_status == "firm" and not orders.empty and provisional_orders:
            global_parts = dict(provisional_parts)
            global_parts[str(part_id)] = part
            global_checker = check_schedule(
                pd.concat([*provisional_orders, orders], ignore_index=True),
                {
                    key: _checker_parts(
                        value, scheduler == "risk_adjusted_mrp"
                    )[key]
                    for key, value in global_parts.items()
                },
                horizon_weeks=14,
                check_requirements=False,
            )
            if not global_checker["feasible"]:
                final_status = "unresolved"
                final_failures.update(global_checker["failed_checks"])
        if final_status == "firm":
            capacity = plan.get("capacity_used", capacity)
            if not orders.empty:
                provisional_orders.append(orders)
                provisional_parts[str(part_id)] = part
        else:
            capacity = capacity_before
        if not plan["weekly"].empty:
            weekly = plan["weekly"].copy()
            for item in weekly.to_dict(orient="records"):
                week = int(item["week"])
                order_rows = (
                    orders.loc[orders["requested_week"].eq(week)].to_dict("records")
                    if not orders.empty
                    else []
                )
                for order in order_rows or [{}]:
                    release_week = order.get("release_week")
                    release_date = (
                        pd.Timestamp(group["week_start_date"].min())
                        + pd.Timedelta(days=7 * (int(release_week) - 1))
                        if release_week is not None
                        else pd.NaT
                    )
                    due_date = (
                        pd.Timestamp(group["week_start_date"].min())
                        + pd.Timedelta(days=7 * (int(order["arrival_week"]) - 1))
                        if order.get("arrival_week") is not None
                        else pd.NaT
                    )
                    rows.append(
                        {
                        "run_id": item["run_id"],
                        "candidate_id": candidate_id,
                        "scheduler_id": scheduler,
                        "forecast_status": "PROVISIONAL",
                        "part_id": str(part_id),
                        "part_name": group["part_name"].iloc[0],
                        "region_plant": "ALL",
                        "week": week,
                        "forecast_qty": item["forecast_qty"],
                        "opening_usable_stock": item["opening_usable_stock"],
                        "expected_receipts": item["expected_receipts"],
                        "projected_inventory": item["projected_inventory"],
                        "backlog_qty": item["backlog_qty"],
                        "net_requirement": item["net_requirement"],
                        "safety_stock": item["safety_stock"],
                        "reorder_point": item["reorder_point"],
                        "service_level_target": item["service_level_target"],
                        "supply_mode": part["supply_mode"],
                        "supplier_id": order.get("supplier_id"),
                        "supplier_risk_flag": None,
                        "order_qty": order.get("order_qty"),
                        "release_week": release_week,
                        "release_date": release_date,
                        "due_date": due_date,
                        "window": (
                            "frozen"
                            if release_week is not None and int(release_week) <= frozen_weeks
                            else "flexible" if release_week is not None else None
                        ),
                        "moq_multiple_adjustment": (
                            int(order["order_qty"] - order["net_requirement"])
                            if order.get("order_qty") is not None
                            else None
                        ),
                        "capacity_allocation": order.get("order_qty"),
                        "sob_target": order.get("sob_target_pct"),
                        "criticality": part["criticality"],
                        "priority": part["planning_priority"],
                        "reason": order.get("reason"),
                        "warnings": order.get("warnings"),
                        "status": final_status,
                        "failed_checks": "|".join(
                            sorted(
                                final_failures
                                | set(
                                    filter(
                                        None,
                                        str(item["failed_checks"]).split("|"),
                                    )
                                )
                            )
                        ),
                        }
                    )
    provisional = pd.DataFrame(rows)
    combined_orders = (
        pd.concat(provisional_orders, ignore_index=True)
        if provisional_orders
        else pd.DataFrame()
    )
    final_checker = check_schedule(
        combined_orders,
        {
            part_id: _checker_parts(part, scheduler == "risk_adjusted_mrp")[part_id]
            for part_id, part in provisional_parts.items()
        },
        horizon_weeks=14,
        check_requirements=False,
    )
    if not final_checker["feasible"]:
        raise RuntimeError(
            "Final provisional firm-order set failed the independent checker: "
            + "|".join(final_checker["failed_checks"])
        )
    provisional.attrs["independent_checker"] = final_checker
    return provisional


def _report_text(
    summary: pd.DataFrame,
    winner: pd.Series,
    sensitivity: pd.DataFrame,
    checker: pd.DataFrame,
    exclusions: pd.DataFrame,
    config_hash: str,
    provisional: pd.DataFrame,
    provisional_checker: Mapping[str, Any],
) -> str:
    primary = summary.loc[summary["initialization"].eq(PRIMARY_INITIALIZATION)]
    stress = summary.loc[summary["initialization"].eq(D2_STRESS_INITIALIZATION)]
    sensitivity_changes = int(sensitivity["winner_changed"].sum())
    exclusion_reasons = (
        exclusions["exclusion_reason"].value_counts().to_dict()
        if not exclusions.empty
        else {}
    )
    checker_reasons = (
        checker.loc[
            ~checker["checker_feasible"], "checker_failed_checks"
        ].value_counts().to_dict()
        if not checker.empty
        else {}
    )
    proposed = summary.loc[
        summary["initialization"].eq(PRIMARY_INITIALIZATION)
        & ~summary["scheduler_id"].eq("traditional_baseline")
    ]
    forecast_service_cost = proposed.groupby(
        ["candidate_id", "scheduler_id"], as_index=False, observed=True
    ).agg(
        absolute_error=("forecast_variance_absolute_units", "sum"),
        actual_demand=("actual_demand_units", "sum"),
        fulfilled=("fulfilled_units", "sum"),
        stockout_part_weeks=("stockout_part_weeks", "sum"),
        part_weeks=("part_weeks", "sum"),
        emergency_cost=("emergency_cost_proxy", "sum"),
        carrying_cost=("carrying_cost_proxy", "sum"),
    )
    forecast_service_cost["weekly_wape"] = (
        forecast_service_cost["absolute_error"]
        / forecast_service_cost["actual_demand"].replace(0, np.nan)
    )
    forecast_service_cost["fill_rate"] = (
        forecast_service_cost["fulfilled"]
        / forecast_service_cost["actual_demand"].replace(0, np.nan)
    )
    lowest_error = forecast_service_cost.sort_values(
        ["weekly_wape", "candidate_id", "scheduler_id"], kind="stable"
    ).iloc[0]
    best_service = forecast_service_cost.sort_values(
        ["fill_rate", "candidate_id", "scheduler_id"],
        ascending=[False, True, True],
        kind="stable",
    ).iloc[0]
    lowest_emergency_cost = forecast_service_cost.sort_values(
        ["emergency_cost", "candidate_id", "scheduler_id"], kind="stable"
    ).iloc[0]
    lowest_carrying_cost = forecast_service_cost.sort_values(
        ["carrying_cost", "candidate_id", "scheduler_id"], kind="stable"
    ).iloc[0]
    return "\n".join(
        [
            "# Phase 6 scheduler comparison",
            "",
            "**Basis:** method validated on simulated data. The 14-week future schedule is provisional; no factory savings or supplier commitment is claimed.",
            "**Phase 5 remains PARTIAL.** This Phase 6 work proceeds under the user-approved sequencing deviation; it does not satisfy the v4.2 sequential Phase 5 gate.",
            "",
            f"- Config SHA-256: `{config_hash}`",
            f"- Winning validation pair (primary initialization): `{winner['candidate_id']}` + `{winner['scheduler_id']}`; score {winner['score']:.4f}.",
            f"- Sensitivity winner changed in {sensitivity_changes}/{len(sensitivity)} one-factor ±10-point perturbations.",
            f"- Primary firm/unresolved paths: {int(primary['firm'].sum())}/{int((~primary['firm']).sum())}.",
            f"- D2 stress firm/unresolved paths: {int(stress['firm'].sum())}/{int((~stress['firm']).sum())}.",
            f"- Firm-order checker failures: {int((checker['firm'] & ~checker['checker_feasible']).sum())}; unresolved checker rows: {int((~checker['checker_feasible']).sum())}, reasons: `{json.dumps(checker_reasons, sort_keys=True)}`; ledger failures: {int((~checker['ledger_feasible']).sum())}.",
            f"- Excluded candidate paths: {len(exclusions)}; reasons: `{json.dumps(exclusion_reasons, sort_keys=True)}`.",
            f"- Provisional future rows: {len(provisional)} ({int(provisional['status'].eq('firm').sum())} firm, {int(provisional['status'].eq('unresolved').sum())} unresolved); all emitted rows carry `PROVISIONAL` status.",
            f"- Aggregate independent checker for provisional firm orders: feasible={provisional_checker['feasible']}; checked orders={provisional_checker['checked_orders']}.",
            (
                "- Forecast/service/cost check (primary): lowest weekly WAPE was "
                f"`{lowest_error['candidate_id']}` + `{lowest_error['scheduler_id']}` "
                f"({lowest_error['weekly_wape']:.2%}); best fill rate was "
                f"`{best_service['candidate_id']}` + `{best_service['scheduler_id']}` "
                f"({best_service['fill_rate']:.2%}), and lowest emergency-cost proxy was "
                f"`{lowest_emergency_cost['candidate_id']}` + "
                f"`{lowest_emergency_cost['scheduler_id']}` "
                f"({lowest_emergency_cost['emergency_cost']:,.0f}). Lowest carrying-cost "
                f"proxy was `{lowest_carrying_cost['candidate_id']}` + "
                f"`{lowest_carrying_cost['scheduler_id']}` "
                f"({lowest_carrying_cost['carrying_cost']:,.0f}); the weighted winner "
                "is reported separately above."
            ),
            "",
            "## Limitations",
            "",
            "- Baseline is a separate historical-average forecast plus D2 reorder-point rule. Its trigger uses projected lead-time inventory position; order quantity replenishes to the D2 reorder point, subject to supplier constraints.",
            "- D2 is one snapshot postdating validation origins. Its use is a stress scenario except in the required historical baseline policy; primary MRP initialization is origin-visible with zero opening POs/backlog.",
            "- Track B schedule stability between replans is not measurable from the saved one-run histories. The 15% adherence/stability scoring component therefore uses simulated on-time received quantity only; neither value is an operational measurement.",
            "- Inventory turnover is a 14-week simulated COGS / average inventory value ratio, not annualized.",
            "- Supplier risk flag is null because the repository does not define an approved threshold for that field. Capacity allocation is represented by allocated order quantity against the reported supplier capacity.",
            "- The unsupported short-series analogue fallback remains deferred; paths without complete saved forecasts are excluded with reasons.",
            "- Phase-out stock value and weeks of cover are reported before and after the validation simulation. Obsolete parts P0046/P0087 have no D2 scheduling snapshot; their values and weeks of cover are explicitly `not_measurable`, not inferred.",
            "",
            "## Outputs",
            "",
            *[f"- `results/{Path(path).name}`" for path in OUTPUTS[:9]],
            "",
        ]
    )


if __name__ == "__main__":
    print(json.dumps(run_phase6(), indent=2, default=str))
