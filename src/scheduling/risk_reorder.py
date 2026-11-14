"""v4.2 A3/A4 MRP addendum: supplier risk and reorder points."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.data.loader import ROOT, load_scheduling
from src.scheduling.engine import (
    build_mrp_schedule,
    part_supplier_risk_score,
)
from src.scheduling.phase5 import load_validation_paths


RISK_LEVELS = ("low", "medium", "high")


def _assumptions(root: Path) -> dict[str, Any]:
    config = yaml.safe_load(
        (root / "configs" / "project.yaml").read_text(encoding="utf-8")
    )
    scheduling = config["scheduling"]
    simulator = scheduling["simulator_assumptions"]
    return {
        "seed": int(simulator["seed"]),
        "initial_stock_multiplier": float(
            simulator["initial_inventory"]["initial_stock_multiplier"]
        ),
        "due_later_week": int(simulator["inbound_receipts"]["due_later_week"]),
        "high_share_warning_pct": float(
            scheduling["supplier_risk"]["high_share_warning_pct"]
        ),
    }


def _risk_levels(parts: dict[str, dict[str, Any]]) -> dict[str, str]:
    scores = pd.Series(
        {part_id: part_supplier_risk_score(part) for part_id, part in parts.items()},
        dtype=float,
    ).dropna()
    if scores.empty:
        return {}
    ranked = scores.rank(method="first")
    levels = pd.qcut(ranked, 3, labels=list(RISK_LEVELS))
    return {str(part_id): str(level) for part_id, level in levels.items()}


def _scaled_snapshot(snapshot: dict[str, Any], multiplier: float) -> dict[str, Any]:
    scaled = dict(snapshot)
    usable = int(np.floor(float(snapshot["available_qty"]) * multiplier))
    scaled["available_qty"] = usable
    scaled["on_hand_qty"] = (
        usable + int(snapshot["reserved_qty"]) + int(snapshot["blocked_qty"])
    )
    return scaled


def _run_variant(
    paths: pd.DataFrame,
    parts: dict[str, dict[str, Any]],
    levels: dict[str, str],
    *,
    risk_adjusted: bool,
    initial_stock_multiplier: float,
    due_later_week: int,
    high_share_warning_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    weekly_frames: list[pd.DataFrame] = []
    order_frames: list[pd.DataFrame] = []
    plan_rows: list[dict[str, Any]] = []
    label = "risk_adjusted" if risk_adjusted else "traditional_baseline"
    for origin, origin_paths in paths.groupby(
        "origin_date", sort=True, observed=True
    ):
        capacity: dict[Any, float] = {}
        for part_id, forecast_group in origin_paths.groupby(
            "part_id", sort=True, observed=True
        ):
            snapshot = parts.get(str(part_id))
            if snapshot is None:
                continue
            snapshot = _scaled_snapshot(snapshot, initial_stock_multiplier)
            run_id = f"v42:{label}:{pd.Timestamp(origin).date()}:{part_id}"
            plan = build_mrp_schedule(
                snapshot,
                forecast_group,
                run_id=run_id,
                horizon_weeks=14,
                capacity_used=capacity,
                due_later_week=due_later_week,
                risk_adjusted=risk_adjusted,
            )
            capacity = plan.get("capacity_used", capacity)
            weekly = plan["weekly"].copy()
            if not weekly.empty:
                weekly["origin_date"] = origin
                weekly["variant"] = label
                weekly["supplier_risk_level"] = levels.get(str(part_id), "unclassified")
                weekly["current_reorder_point_qty"] = snapshot[
                    "current_reorder_point_qty"
                ]
                weekly["reorder_point_delta"] = (
                    weekly["reorder_point"] - weekly["current_reorder_point_qty"]
                )
                weekly["reorder_point_delta_pct"] = np.where(
                    weekly["current_reorder_point_qty"].gt(0),
                    weekly["reorder_point_delta"]
                    / weekly["current_reorder_point_qty"]
                    * 100.0,
                    np.nan,
                )
                weekly_frames.append(weekly)
            orders = plan["orders"].copy()
            if not orders.empty:
                orders["origin_date"] = origin
                orders["variant"] = label
                orders["supplier_risk_level"] = orders["part_id"].astype(str).map(
                    levels
                )
                orders["high_risk_supplier_large_share_flag"] = (
                    orders["supplier_risk_level"].eq("high")
                    & orders["sob_target_pct"].ge(high_share_warning_pct)
                )
                orders["risk_warning"] = np.where(
                    orders["high_risk_supplier_large_share_flag"],
                    "high_risk_supplier_large_share",
                    "",
                )
                orders["planned_receipt_qty_after_rejection"] = (
                    orders["order_qty"] * (1.0 - orders["supplier_reject_pct"] / 100.0)
                )
                order_frames.append(orders)
            plan_rows.append(
                {
                    "variant": label,
                    "origin_date": pd.Timestamp(origin),
                    "part_id": str(part_id),
                    "status": plan["status"],
                    "failed_checks": "|".join(plan["failed_checks"]),
                    "safety_stock": plan["safety_stock"],
                    "reorder_point": plan["reorder_point"],
                    "reorder_trigger_week": plan["reorder_trigger_week"],
                    "supplier_risk_score": plan["supplier_risk_score"],
                    "supplier_risk_level": levels.get(str(part_id), "unclassified"),
                    "risk_adjusted": plan["risk_adjusted"],
                }
            )
    weekly_frame = (
        pd.concat(weekly_frames, ignore_index=True) if weekly_frames else pd.DataFrame()
    )
    order_frame = (
        pd.concat(order_frames, ignore_index=True) if order_frames else pd.DataFrame()
    )
    return weekly_frame, order_frame, plan_rows


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    return value


def run_risk_reorder(root: Path = ROOT) -> dict[str, Any]:
    assumptions = _assumptions(root)
    scheduling = load_scheduling(root / "data" / "02_scheduling_inputs.csv")
    parts = {
        str(row["part_id"]): row for row in scheduling.to_dict(orient="records")
    }
    levels = _risk_levels(parts)
    paths = load_validation_paths(root)
    outputs: dict[str, tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]] = {}
    for risk_adjusted in (False, True):
        outputs[risk_adjusted] = _run_variant(
            paths,
            parts,
            levels,
            risk_adjusted=risk_adjusted,
            initial_stock_multiplier=assumptions["initial_stock_multiplier"],
            due_later_week=assumptions["due_later_week"],
            high_share_warning_pct=assumptions["high_share_warning_pct"],
        )
    weekly = pd.concat(
        [item[0] for item in outputs.values()], ignore_index=True
    )
    orders = pd.concat(
        [item[1] for item in outputs.values()], ignore_index=True
    )
    plans = pd.DataFrame(
        [row for item in outputs.values() for row in item[2]]
    )
    weekly["projected_unmet_units"] = (
        weekly["lost_sales_qty"].fillna(0) + weekly["backlog_qty"].fillna(0)
    )
    results = root / "results"
    weekly.to_csv(results / "phase6_supplier_risk_weekly.csv", index=False)
    orders.to_csv(results / "phase6_supplier_risk_orders.csv", index=False)
    plans.to_csv(results / "phase6_supplier_risk_plans.csv", index=False)

    comparison_rows = []
    for variant, group in plans.groupby("variant", sort=False):
        variant_orders = orders.loc[orders["variant"].eq(variant)]
        variant_weekly = weekly.loc[weekly["variant"].eq(variant)]
        comparison_rows.append(
            {
                "variant": variant,
                "part_origins": int(len(group)),
                "firm_part_origins": int(group["status"].eq("firm").sum()),
                "unresolved_part_origins": int(group["status"].ne("firm").sum()),
                "order_lines": int(len(variant_orders)),
                "order_qty": int(variant_orders["order_qty"].sum()),
                "expected_rejected_qty": int(
                    (
                        variant_orders["order_qty"]
                        - variant_orders["planned_receipt_qty_after_rejection"]
                    ).sum()
                ),
                "mean_safety_stock": float(group["safety_stock"].mean()),
                "mean_reorder_point": float(group["reorder_point"].mean()),
                "parts_with_reorder_trigger": int(
                    group["reorder_trigger_week"].notna().sum()
                ),
                "weekly_below_reorder_point": int(
                    variant_weekly["below_reorder_point_flag"].sum()
                ),
                "projected_unmet_units": float(
                    variant_weekly["projected_unmet_units"].sum()
                ),
                "high_risk_large_share_warnings": int(
                    variant_orders["high_risk_supplier_large_share_flag"].sum()
                ),
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(results / "phase6_supplier_risk_comparison.csv", index=False)

    risk_on = comparison.loc[comparison["variant"].eq("risk_adjusted")].iloc[0]
    risk_off = comparison.loc[
        comparison["variant"].eq("traditional_baseline")
    ].iloc[0]
    baseline_weekly = weekly.loc[weekly["variant"].eq("traditional_baseline")]
    reorder_summary = {}
    for criticality, group in baseline_weekly.groupby("criticality", observed=True):
        first = group.drop_duplicates(["origin_date", "part_id"])
        reorder_summary[str(criticality)] = {
            "part_origins": int(len(first)),
            "median_reorder_point": float(first["reorder_point"].median()),
            "median_current_reorder_point": float(
                first["current_reorder_point_qty"].median()
            ),
            "median_delta": float(first["reorder_point_delta"].median()),
            "median_delta_pct": float(
                first["reorder_point_delta_pct"].median()
            ),
            "above_current": int(first["reorder_point_delta"].gt(0).sum()),
            "below_current": int(first["reorder_point_delta"].lt(0).sum()),
            "equal_current": int(first["reorder_point_delta"].eq(0).sum()),
        }
    level_counts = pd.Series(levels).value_counts().to_dict()
    summary = {
        "status": "success",
        "amendment": "SPEC v4.2 A3/A4",
        "basis": "synthetic held-validation demand paths and one D2 snapshot",
        "assumptions": assumptions,
        "risk_level_parts": {str(key): int(value) for key, value in level_counts.items()},
        "comparison": comparison.to_dict(orient="records"),
        "quantity_change_pct_risk_on_vs_off": float(
            (risk_on["order_qty"] - risk_off["order_qty"])
            / risk_off["order_qty"]
            * 100.0
        ),
        "unresolved_change": int(
            risk_on["unresolved_part_origins"] - risk_off["unresolved_part_origins"]
        ),
        "projected_unmet_change_units": float(
            risk_on["projected_unmet_units"] - risk_off["projected_unmet_units"]
        ),
        "high_risk_large_share_warnings": int(
            risk_on["high_risk_large_share_warnings"]
        ),
        "reorder_by_criticality": reorder_summary,
        "notes": [
            "Traditional baseline ignores supplier risk by design.",
            "Risk-adjusted planning is deterministic expected-value planning; the simulator is not rerun here.",
            "SOB remains a target, not a binding contract.",
            "D2 has one snapshot, so no historical schedule accuracy is claimed.",
        ],
    }
    (results / "phase6_supplier_risk_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2), encoding="utf-8"
    )
    reorder_rows = []
    for key, value in reorder_summary.items():
        reorder_rows.append(
            f"| {key} | {value['part_origins']} | "
            f"{value['median_reorder_point']:.1f} | "
            f"{value['median_current_reorder_point']:.1f} | "
            f"{value['median_delta']:.1f} | {value['median_delta_pct']:.1f}% | "
            f"{value['above_current']} | {value['below_current']} |"
        )
    reorder_table = "\n".join(reorder_rows)
    report = f"""# Supplier risk and reorder points in MRP (v4.2 A3/A4)

**Basis:** synthetic held-validation demand paths; one D2 snapshot. No historical
schedule accuracy is claimed.

## What changed

- Planning lead time per supplier slot: `lead + (1 - OTD) x average delay`.
- Order quantity is grossed up by `1 / (1 - reject rate)` before MOQ and order
  multiple rounding.
- The traditional baseline ignores both adjustments, by design.
- `supplier_risk_score = 1 - reliability_pct / 100` is display only.
  `supplier_risk_level` is low/medium/high by population tertiles of the
  SOB-weighted part score.
- `reorder_point = lead-time point demand + safety stock`;
  `reorder_trigger_week` is the first week projected inventory is at or below
  it.

## Ablation on identical demand paths

| Variant | Part-origins | Unresolved | Order lines | Order qty | Expected rejection loss | Mean safety stock | Projected unmet units | High-risk large-share warnings |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Traditional baseline | {int(risk_off['part_origins'])} | {int(risk_off['unresolved_part_origins'])} | {int(risk_off['order_lines'])} | {int(risk_off['order_qty'])} | {int(risk_off['expected_rejected_qty'])} | {risk_off['mean_safety_stock']:.1f} | {risk_off['projected_unmet_units']:.0f} | {int(risk_off['high_risk_large_share_warnings'])} |
| Risk-adjusted | {int(risk_on['part_origins'])} | {int(risk_on['unresolved_part_origins'])} | {int(risk_on['order_lines'])} | {int(risk_on['order_qty'])} | {int(risk_on['expected_rejected_qty'])} | {risk_on['mean_safety_stock']:.1f} | {risk_on['projected_unmet_units']:.0f} | {int(risk_on['high_risk_large_share_warnings'])} |

Risk-adjusted order quantity changes by
**{summary['quantity_change_pct_risk_on_vs_off']:.2f}%** versus the baseline.
Projected unmet units change by **{summary['projected_unmet_change_units']:.0f}**
because longer expected lead times delay receipts; higher order quantity does
not guarantee earlier coverage. This is a measured MRP-projection trade, not a
service improvement. No realized-savings claim is made.

## Reorder point comparison with D2

| Criticality | Part-origins | Median reorder point | Median current | Median delta | Median delta % | Above current | Below current |
|---|---:|---:|---:|---:|---:|---:|---:|
{reorder_table}

D2 `current_reorder_point_qty` is a snapshot value with no documented derivation.
The delta is reported, not treated as an error in either direction.

## Risk levels

{", ".join(f"{level}: {count}" for level, count in sorted(level_counts.items()))}
parts.

## Limits

- The `high_share_warning_pct` threshold of
  {assumptions['high_share_warning_pct']:.0f}% is a provisional display
  parameter (`provisional_editable`), not a business rule.
- Reliability is used only for the display score; planning uses OTD and
  rejection.
- The simulator is not rerun, so no service-level claim is made for the
  risk-adjusted plan.
- The D2 snapshot postdates the validation origins; initial inventory is the
  same synthetic initialization as Phase 5.

## Outputs

`results/phase6_supplier_risk_weekly.csv`,
`results/phase6_supplier_risk_orders.csv`,
`results/phase6_supplier_risk_plans.csv`,
`results/phase6_supplier_risk_comparison.csv`,
`results/phase6_supplier_risk_summary.json`.
"""
    (results / "phase6_supplier_risk_report.md").write_text(
        report, encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(_json_safe(run_risk_reorder()), indent=2))
