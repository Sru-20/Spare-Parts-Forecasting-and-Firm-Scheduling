from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.scheduling.phase6 import (
    D2_STRESS_INITIALIZATION,
    POINT_CANDIDATES,
    _config,
    _kpi_rows,
    _paired_bootstrap,
    _run_plan,
    _sensitivity_scores,
    _traditional_baseline,
    _write_new_csv,
    load_comparison_paths,
)


def _part(**overrides):
    part = {
        "part_id": "P1",
        "lifecycle_status": "Active",
        "supply_mode": "Procurement",
        "criticality": "High",
        "service_level_target": 0.99,
        "on_hand_qty": 10,
        "reserved_qty": 0,
        "blocked_qty": 0,
        "available_qty": 10,
        "backorder_qty": 0,
        "in_transit_qty": 0,
        "open_po_qty": 0,
        "open_po_due_this_month_qty": 0,
        "open_po_due_later_qty": 0,
        "current_reorder_point_qty": 10,
        "unit_cost": 10.0,
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
                f"sup{slot}_otd_pct": 100.0 if present else None,
                f"sup{slot}_avg_delay_days": 0.0 if present else None,
                f"sup{slot}_reject_pct": 0.0 if present else None,
                f"sup{slot}_reliability_pct": 100.0 if present else None,
            }
        )
    part.update(overrides)
    return part


def _path():
    return pd.DataFrame(
        {
            "origin_date": pd.Timestamp("2025-01-13"),
            "part_id": "P1",
            "week": range(1, 15),
            "week_start_date": pd.date_range(
                "2025-01-20", periods=14, freq="W-MON"
            ),
            "historical_average_qty": [2.0] * 14,
            "forecast_qty": [9.0] * 14,
            "forecast_p90": [11.0] * 14,
            "forecast_p95": [12.0] * 14,
            "forecast_p99": [13.0] * 14,
            "actual_qty": [3.0] * 14,
        }
    )


def test_traditional_baseline_uses_history_forecast_and_d2_reorder_point():
    plan = _traditional_baseline(
        _part(on_hand_qty=7, available_qty=7, current_reorder_point_qty=10),
        _path(),
        run_id="baseline",
        horizon=14,
        due_later_week=4,
    )

    assert plan["orders"]["order_qty"].sum() > 0
    assert set(plan["orders"]["reason"]) == {
        "historical_average_forecast_d2_reorder_point"
    }
    assert plan["weekly"]["reorder_point"].eq(10).all()


def test_baseline_policy_is_not_the_mrp_risk_off_label():
    plan = _traditional_baseline(
        _part(), _path(), run_id="baseline", horizon=14, due_later_week=4
    )

    assert plan["weekly"]["forecast_qty"].eq(2.0).all()
    assert plan["weekly"]["forecast_qty"].ne(9.0).all()


def test_validation_loader_uses_common_complete_candidate_paths(tmp_path):
    artifact = tmp_path / "artifacts"
    results = tmp_path / "results"
    artifact.mkdir()
    results.mkdir()
    rows = []
    for model, candidate in (
        ("lightgbm_tweedie", POINT_CANDIDATES[0]),
        ("lightgbm_quantile", POINT_CANDIDATES[1]),
        ("historical_average", "historical_average::origin_mean"),
    ):
        for region in ("East", "West"):
            for step, date in enumerate(
                pd.date_range("2025-01-20", periods=14, freq="W-MON"), 1
            ):
                point = 4.0
                rows.append(
                    {
                        "model": model,
                        "candidate_id": candidate,
                        "origin_date": "2025-01-13",
                        "part_id": "P1",
                        "region": region,
                        "week_start_date": date,
                        "horizon_step": step,
                        "evaluation_split": "evaluation",
                        "forecast_qty": point,
                        "forecast_p90": 5.0 if model == "lightgbm_quantile" else None,
                        "forecast_p95": 6.0 if model == "lightgbm_quantile" else None,
                        "forecast_p99": 7.0 if model == "lightgbm_quantile" else None,
                        "actual_qty": 3.0,
                        "criticality": "High",
                    }
                )
    pd.DataFrame(rows).to_parquet(
        artifact / "track_a_backtest_forecasts_v41.parquet", index=False
    )
    pd.DataFrame(
        {
            "candidate_id": POINT_CANDIDATES,
            "eligible_for_weighted_selection": [True, True],
        }
    ).to_csv(results / "track_a_v41_selection_matrix.csv", index=False)

    paths, exclusions = load_comparison_paths(tmp_path)

    assert all(len(frame) == 14 for frame in paths.values())
    assert paths[POINT_CANDIDATES[0]]["actual_qty"].eq(6.0).all()
    assert exclusions.empty


def test_incomplete_candidate_prediction_is_excluded_not_imputed(tmp_path):
    artifact = tmp_path / "artifacts"
    results = tmp_path / "results"
    artifact.mkdir()
    results.mkdir()
    rows = []
    for model, candidate in (
        ("lightgbm_tweedie", POINT_CANDIDATES[0]),
        ("lightgbm_quantile", POINT_CANDIDATES[1]),
        ("historical_average", "historical_average::origin_mean"),
    ):
        for step, date in enumerate(
            pd.date_range("2025-01-20", periods=14, freq="W-MON"), 1
        ):
            for region in ("East", "West"):
                prediction = 4.0
                if model == "lightgbm_tweedie" and step == 1:
                    prediction = float("nan")
                rows.append(
                    {
                        "model": model,
                        "candidate_id": candidate,
                        "origin_date": "2025-01-13",
                        "part_id": "P1",
                        "region": region,
                        "week_start_date": date,
                        "horizon_step": step,
                        "evaluation_split": "evaluation",
                        "forecast_qty": prediction,
                        "forecast_p90": 5.0 if model == "lightgbm_quantile" else None,
                        "forecast_p95": 6.0 if model == "lightgbm_quantile" else None,
                        "forecast_p99": 7.0 if model == "lightgbm_quantile" else None,
                        "actual_qty": 3.0,
                        "criticality": "High",
                    }
                )
    pd.DataFrame(rows).to_parquet(
        artifact / "track_a_backtest_forecasts_v41.parquet", index=False
    )
    pd.DataFrame(
        {
            "candidate_id": POINT_CANDIDATES,
            "eligible_for_weighted_selection": [True, True],
        }
    ).to_csv(results / "track_a_v41_selection_matrix.csv", index=False)

    paths, exclusions = load_comparison_paths(tmp_path)

    assert all(frame.empty for frame in paths.values())
    assert "missing_candidate_prediction" in "|".join(
        exclusions["exclusion_reason"].astype(str)
    )


def test_weights_are_validated_and_sensitive_scores_repeat():
    config = _config(Path(__file__).resolve().parents[1])
    weights = config["scoring"]["tracks"]["scheduling"]["weights"]
    assert sum(weights.values()) == pytest.approx(1.0)
    summary = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "scheduler_id": scheduler,
                "initialization": "origin_visible",
                "critical_available_weeks": 10,
                "critical_total_weeks": 14,
                "fulfilled_units": 10,
                "actual_demand_units": 14,
                "emergency_cost_proxy": cost,
                "excess_inventory_units": 3,
                "carrying_cost_proxy": 5,
                "on_time_received_qty": 5,
                "scheduled_order_qty": 6,
                "turnover_14_week": 2.0,
                "firm": True,
                "hard_constraint_violation": False,
            }
            for scheduler, cost in (
                ("traditional_baseline", 10.0),
                ("mrp", 8.0),
                ("risk_adjusted_mrp", 9.0),
            )
        ]
    )
    first = _sensitivity_scores(summary, weights)
    second = _sensitivity_scores(summary, weights)
    pd.testing.assert_frame_equal(first, second)


def test_checker_failures_are_disqualified_from_scores():
    summary = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "scheduler_id": scheduler,
                "initialization": "origin_visible",
                "critical_available_weeks": 8,
                "critical_total_weeks": 10,
                "fulfilled_units": 8,
                "actual_demand_units": 10,
                "emergency_cost_proxy": 1.0,
                "excess_inventory_units": 1.0,
                "carrying_cost_proxy": 1.0,
                "on_time_received_qty": 5,
                "scheduled_order_qty": 6,
                "turnover_14_week": 2.0,
                "firm": True,
                "hard_constraint_violation": scheduler == "mrp",
            }
            for scheduler in (
                "traditional_baseline",
                "mrp",
                "risk_adjusted_mrp",
            )
        ]
    )
    scored = _sensitivity_scores(summary, {"critical_part_availability": 1.0})

    assert scored.loc[scored["scheduler_id"].eq("mrp"), "score"].eq(float("-inf")).all()


def test_kpi_zero_baseline_is_not_reported_as_percentage():
    rows = []
    for scheduler, emergency_cost in (
        ("traditional_baseline", 0.0),
        ("mrp", 5.0),
    ):
        rows.append(
            {
                "candidate_id": "c1",
                "scheduler_id": scheduler,
                "initialization": "origin_visible",
                "actual_demand_units": 10.0,
                "forecast_variance_absolute_units": 2.0,
                "forecast_variance_signed_units": 1.0,
                "forecast_bias": 0.1,
                "forecast_bias_abs": 0.1,
                "critical_available_weeks": 8,
                "critical_total_weeks": 10,
                "stockout_part_weeks": 2,
                "part_weeks": 10,
                "fulfilled_units": 8,
                "on_time_received_qty": 5,
                "scheduled_order_qty": 6,
                "emergency_order_count": 0,
                "emergency_cost_proxy": emergency_cost,
                "excess_inventory_units": 4.0,
                "carrying_cost_proxy": 2.0,
                "turnover_14_week": 1.0,
                "phase_out_stock_value_before": 0.0,
                "phase_out_stock_value_after": 0.0,
                "phase_out_stock_qty_before": 0.0,
                "phase_out_stock_qty_after": 0.0,
                "phase_out_average_weekly_demand": 0.0,
            }
        )

    kpis = _kpi_rows(pd.DataFrame(rows))
    emergency = kpis.loc[
        kpis["scheduler_id"].eq("mrp")
        & kpis["kpi"].eq("emergency_cost_proxy")
    ].iloc[0]
    reduction = kpis.loc[
        kpis["scheduler_id"].eq("mrp")
        & kpis["kpi"].eq("emergency_procurement_reduction")
    ].iloc[0]

    assert emergency["percentage_difference_status"] == "not_defined_zero_baseline"
    assert pd.isna(emergency["percentage_difference"])
    assert reduction["value"] == -5.0
    assert reduction["worse_than_baseline"]
    obsolete = kpis.loc[kpis["kpi"].eq("obsolete_stock_value_before")]
    assert obsolete["basis"].eq("not_measurable").all()
    assert obsolete["value"].isna().all()


def test_paired_bootstrap_is_matched_and_deterministic():
    rows = []
    for scheduler, fill in (
        ("traditional_baseline", 0.7),
        ("mrp", 0.8),
        ("risk_adjusted_mrp", 0.75),
    ):
        for origin in ("2025-01-13", "2025-04-21"):
            rows.append(
                {
                    "candidate_id": POINT_CANDIDATES[0],
                    "scheduler_id": scheduler,
                    "initialization": "origin_visible",
                    "origin_date": origin,
                    "part_id": "P1",
                    "fill_rate": fill,
                    "critical_availability": fill,
                    "stock_out_pct": 1.0 - fill,
                    "forecast_wape": 1.0 - fill,
                    "forecast_bias": 0.1,
                    "schedule_adherence": fill,
                    "emergency_cost_proxy": 3.0,
                    "excess_inventory_units": 2.0,
                    "carrying_cost_proxy": 1.0,
                    "turnover_14_week": 1.2,
                }
            )
    frame = pd.DataFrame(rows)
    first = _paired_bootstrap(frame, seed=42, replicates=100)
    second = _paired_bootstrap(frame, seed=42, replicates=100)
    pd.testing.assert_frame_equal(first, second)
    assert first["paired_part_origins"].eq(2).all()


def test_scheduler_run_is_deterministic_with_fixed_seed():
    root = Path(__file__).resolve().parents[1]
    inputs = (
        POINT_CANDIDATES[0],
        "mrp",
        D2_STRESS_INITIALIZATION,
        _path(),
        _part(),
        {},
        _config(root),
    )
    first = _run_plan(*inputs)
    second = _run_plan(*inputs)

    pd.testing.assert_frame_equal(
        pd.DataFrame([first[0]]), pd.DataFrame([second[0]])
    )
    pd.testing.assert_frame_equal(first[1], second[1])
    pd.testing.assert_frame_equal(first[2], second[2])
    assert first[3] == second[3]


def test_writer_refuses_to_overwrite(tmp_path):
    target = tmp_path / "existing.csv"
    target.write_text("original", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _write_new_csv(pd.DataFrame({"value": [1]}), target)

    assert target.read_text(encoding="utf-8") == "original"
