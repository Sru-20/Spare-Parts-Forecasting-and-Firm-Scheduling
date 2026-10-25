import numpy as np
import pandas as pd
import pytest

from src.forecasting.phase4_v41 import (
    _add_historical_average,
    _cluster_bootstrap_ci,
    _lead_time_rows,
    _label_provisional_bridge,
    _metrics_table,
    _metric_values,
    _rank_score,
    _sensitivity,
)


def _forecast_rows(lead_time_weeks=17):
    rows = []
    for step in range(1, 15):
        rows.append(
            {
                "model": "lightgbm_quantile",
                "config_id": "q",
                "origin_date": pd.Timestamp("2025-01-13"),
                "part_id": "P1",
                "region": "North",
                "horizon_step": step,
                "lead_time_weeks": lead_time_weeks,
                "actual_qty": float(step),
                "forecast_qty": float(step + 1),
                "forecast_p90": float(step + 3),
                "forecast_p95": float(step + 4),
                "forecast_p99": float(step + 5),
                "mase_scale": 1.0,
                "criticality": "High",
                "abc_class": "A",
                "sbc_class": "Intermittent",
                "part_category": "Electrical",
            }
        )
    return pd.DataFrame(rows)


def test_lead_time_extension_uses_configured_final_period_average():
    result = _lead_time_rows(_forecast_rows(), extension_period_weeks=4)

    assert len(result) == 1
    assert result.iloc[0]["lead_time_extension_weeks"] == 3
    assert result.iloc[0]["actual_qty"] == pytest.approx(142.5)
    assert result.iloc[0]["forecast_qty"] == pytest.approx(159.5)
    assert result.iloc[0]["forecast_p90"] == pytest.approx(193.5)


def test_metrics_table_contains_ratio_metrics_at_series_and_lead_levels():
    weekly = _forecast_rows(lead_time_weeks=1)
    lead = _lead_time_rows(weekly, extension_period_weeks=4)

    metrics = _metrics_table(weekly, lead)
    overall = metrics.loc[
        metrics["evaluation_level"].eq("weekly")
        & metrics["segment_dimension"].eq("overall")
    ].iloc[0]
    per_series = metrics.loc[
        metrics["evaluation_level"].eq("lead_time")
        & metrics["segment_dimension"].eq("part_region")
    ]

    assert overall["wape_ratio"] == pytest.approx(14.0 / 105.0)
    assert len(per_series) == 1
    assert per_series.iloc[0]["segment_value"] == "P1|North"


def test_historical_average_uses_only_observed_train_validation_rows_by_origin():
    origin = pd.Timestamp("2025-01-13")
    forecasts = pd.DataFrame(
        {
            "part_id": ["P1"],
            "region": ["North"],
            "week_start_date": [origin + pd.Timedelta(days=7)],
            "origin_date": [origin],
            "candidate_role": ["forecast_candidate"],
            "actual_qty": [4.0],
            "model": ["naive"],
            "config_id": ["default"],
            "forecast_qty": [0.0],
        }
    )
    demand = pd.DataFrame(
        {
            "part_id": ["P1", "P1", "P1", "P1"],
            "region": ["North"] * 4,
            "week_start_date": [
                origin - pd.Timedelta(days=14),
                origin - pd.Timedelta(days=7),
                origin,
                origin + pd.Timedelta(days=7),
            ],
            "split_role": ["train_val"] * 4,
            "demand_qty": [2.0, 4.0, pd.NA, 100.0],
        }
    )

    output, _ = _add_historical_average(forecasts, demand)
    baseline = output.loc[output["model"].eq("historical_average")]

    assert baseline["forecast_qty"].tolist() == [3.0]


def test_v41_metrics_store_error_ratios_not_percentages():
    rows = _forecast_rows(lead_time_weeks=1).iloc[:1].copy()
    rows["actual_qty"] = 10.0
    rows["forecast_qty"] = 12.0
    values = _metric_values(rows)

    assert values["wape_ratio"] == pytest.approx(0.2)
    assert values["bias_ratio"] == pytest.approx(0.2)
    assert values["forecast_accuracy_pct"] == pytest.approx(80.0)


def test_future_projection_is_labeled_as_provisional_bridge():
    output = _label_provisional_bridge(pd.DataFrame({"forecast_qty": [3.0]}))
    assert output["forecast_status"].tolist() == ["provisional_bridge"]


def test_two_way_bootstrap_resamples_series_and_origins():
    units = pd.DataFrame(
        [
            {
                "origin_date": origin,
                "part_id": part,
                "region": "North",
                "numerator": 2.0 * denominator,
                "denominator": denominator,
            }
            for origin in ("O1", "O2")
            for part, denominator in (("P1", 1.0), ("P2", 3.0))
        ]
    )

    low, high = _cluster_bootstrap_ci(
        units, "numerator", "denominator", replicates=100, seed=42, confidence=0.95
    )

    assert low == pytest.approx(2.0)
    assert high == pytest.approx(2.0)


def test_missing_rank_metrics_get_zero_and_sensitivity_has_fourteen_cases():
    assert _rank_score(pd.Series([1.0, 2.0, np.nan])).tolist() == [100.0, 0.0, 0.0]
    rows = pd.DataFrame(
        [
            {
                "candidate_id": "simple",
                "lead_time_wape_ratio": 0.2,
                "weekly_wape_ratio": 0.4,
                "critical_a_lead_wape_ratio": 0.2,
                "critical_a_asymmetric_bias_ratio": 0.1,
                "intermittent_lumpy_lead_wape_ratio": 0.3,
                "intermittent_lumpy_mase": 1.0,
                "lead_asymmetric_bias_ratio": 0.1,
                "weekly_asymmetric_bias_ratio": 0.2,
                "calibration_coverage_error": np.nan,
                "calibration_pinball_loss": np.nan,
                "has_probabilistic_output": False,
                "origin_lead_time_wape_std": 0.05,
                "runtime_seconds": 1.0,
                "simplicity_tier": 3,
                "explainability_tier": 3,
            },
            {
                "candidate_id": "quantile",
                "lead_time_wape_ratio": 0.25,
                "weekly_wape_ratio": 0.35,
                "critical_a_lead_wape_ratio": 0.3,
                "critical_a_asymmetric_bias_ratio": 0.2,
                "intermittent_lumpy_lead_wape_ratio": 0.25,
                "intermittent_lumpy_mase": 0.8,
                "lead_asymmetric_bias_ratio": 0.15,
                "weekly_asymmetric_bias_ratio": 0.15,
                "calibration_coverage_error": 0.01,
                "calibration_pinball_loss": 1.0,
                "has_probabilistic_output": True,
                "origin_lead_time_wape_std": 0.03,
                "runtime_seconds": 2.0,
                "simplicity_tier": 2,
                "explainability_tier": 2,
            },
        ]
    )
    weights = {
        "overall_error": 0.20,
        "critical_a_class_error": 0.20,
        "intermittent_lumpy": 0.15,
        "asymmetric_absolute_bias": 0.15,
        "calibration": 0.15,
        "stability_across_origins": 0.10,
        "runtime_and_explainability": 0.05,
    }

    cases = _sensitivity(rows, weights, points=0.10)

    assert len(cases) == 14
    assert cases["case"].nunique() == 14
