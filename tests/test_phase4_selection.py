import numpy as np
import pandas as pd

from src.forecasting.backtest import _lead_time_rows
from src.forecasting.phase4 import (
    _future_calendar,
    _rank_score,
    _sensitivity,
    _weighted_selection,
)


def _lead_window(forecasts):
    rows = []
    for step, prediction in enumerate(forecasts, start=1):
        rows.append(
            {
                "model": "candidate",
                "config_id": "config",
                "evaluation_split": "evaluation",
                "origin_date": pd.Timestamp("2025-01-13"),
                "part_id": "P1",
                "region": "North",
                "candidate_role": "forecast_candidate",
                "ablation_group": "none",
                "sbc_class": "Intermittent",
                "abc_class": "A",
                "xyz_class": "Z",
                "criticality": "High",
                "part_category": "Electrical",
                "lifecycle_status": "Active",
                "lead_time_days": 14.0,
                "lead_time_weeks": 2,
                "actual_qty": float(step * 10),
                "forecast_qty": prediction,
                "forecast_p90": np.nan,
                "forecast_p95": np.nan,
                "forecast_p99": np.nan,
                "mase_scale": 1.0,
                "runtime_seconds": 1.0,
                "horizon_step": step,
            }
        )
    return pd.DataFrame(rows)


def test_lead_time_aggregation_does_not_turn_missing_forecasts_into_zero():
    aggregated = _lead_time_rows(_lead_window([np.nan, 20.0]), horizon=14)

    assert aggregated.empty


def test_complete_lead_time_window_is_aggregated():
    aggregated = _lead_time_rows(_lead_window([10.0, 20.0]), horizon=14)

    assert len(aggregated) == 1
    assert aggregated.iloc[0]["actual_qty"] == 30.0
    assert aggregated.iloc[0]["forecast_qty"] == 30.0


def test_rank_score_assigns_worst_score_to_missing_metrics():
    scores = _rank_score(pd.Series([10.0, 20.0, np.nan]), lower_is_better=True)

    assert scores.iloc[0] == 100.0
    assert scores.iloc[1] == 0.0
    assert scores.iloc[2] == 0.0


def test_calibration_component_prioritizes_lead_time_and_sensitivity_has_16_cases():
    metrics = pd.DataFrame(
        [
            {
                "model": "lightgbm_quantile",
                "config_id": "a",
                "candidate_id": "a",
                "wape_mixed_pct": 1.0,
                "absolute_bias_mixed_pct": 1.0,
                "a_class_critical_parts_wape_pct": 1.0,
                "intermittent_lumpy_wape_pct": 1.0,
                "mean_coverage_error": 0.01,
                "pinball_loss": 1.0,
                "lead_time_mean_coverage_error": 0.10,
                "lead_time_pinball_loss": 10.0,
                "origin_lead_time_wape_std_pct": 1.0,
                "runtime_seconds": 1.0,
                "explainability_tier": 2,
                "n_weekly": 10,
                "n_lead_time": 10,
            },
            {
                "model": "random_forest",
                "config_id": "b",
                "candidate_id": "b",
                "wape_mixed_pct": 1.0,
                "absolute_bias_mixed_pct": 1.0,
                "a_class_critical_parts_wape_pct": 1.0,
                "intermittent_lumpy_wape_pct": 1.0,
                "mean_coverage_error": 0.10,
                "pinball_loss": 10.0,
                "lead_time_mean_coverage_error": 0.01,
                "lead_time_pinball_loss": 1.0,
                "origin_lead_time_wape_std_pct": 1.0,
                "runtime_seconds": 1.0,
                "explainability_tier": 2,
                "n_weekly": 10,
                "n_lead_time": 10,
            },
        ]
    )
    weights = {
        "wape": 0.0,
        "absolute_bias": 0.0,
        "a_class_critical_parts": 0.0,
        "intermittent_lumpy": 0.0,
        "calibration": 1.0,
        "stability_across_origins": 0.0,
        "runtime": 0.0,
        "explainability": 0.0,
    }
    tiers = {
        "formula_baseline": 3,
        "intermittent_baseline": 3,
        "global_tree_model": 2,
        "ensemble_or_router": 1,
    }
    ranked = _weighted_selection(metrics, weights, tiers, 0.70, 0.30)

    assert ranked.iloc[0]["candidate_id"] == "b"
    assert ranked.loc[ranked["candidate_id"].eq("a"), "calibration_score"].iloc[0] == 30.0
    assert ranked.loc[ranked["candidate_id"].eq("b"), "calibration_score"].iloc[0] == 70.0
    assert len(_sensitivity(ranked, weights, tiers, 0.10)) == 16


def test_future_bridge_calendar_drops_test_labels_and_outcomes():
    dates = pd.to_datetime(["2026-03-30", "2026-09-28"])
    demand = pd.DataFrame(
        {
            "part_id": ["P1", "P1"],
            "region": ["North", "North"],
            "week_start_date": dates,
            "split_role": ["final_test", "future"],
            "demand_qty": [999, pd.NA],
            "criticality": ["High", "High"],
            "month": [3, 9],
            "out_stockout_flag": [1, 0],
        }
    )

    calendar = _future_calendar(demand, pd.Timestamp("2026-03-23"))

    assert calendar["week_start_date"].tolist() == list(dates)
    assert "demand_qty" not in calendar
    assert "split_role" not in calendar
    assert "out_stockout_flag" not in calendar
