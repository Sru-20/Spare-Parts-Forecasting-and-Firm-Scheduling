import json

import numpy as np
import pandas as pd
import pytest

from src.data.loader import ROOT
from src.forecasting.bias_calibration import (
    _volume_assignment,
    apply_factors,
    evaluate_candidate,
    fit_group_factors,
    fit_multiplicative_factor,
)
from src.forecasting.phase4_v41 import _lead_time_rows


def _rows(pairs, **extra):
    return pd.DataFrame(
        [
            {"actual_qty": actual, "forecast_qty": forecast, **extra}
            for actual, forecast in pairs
        ]
    )


def test_factor_is_actual_over_forecast_and_is_clamped():
    frame = _rows([(120.0, 100.0)] * 40)

    assert fit_multiplicative_factor(frame) == pytest.approx(1.2)
    assert fit_multiplicative_factor(_rows([(200.0, 100.0)] * 40)) == 1.5
    assert fit_multiplicative_factor(_rows([(40.0, 100.0)] * 40)) == 0.5


def test_thin_or_degenerate_evidence_returns_identity():
    assert fit_multiplicative_factor(_rows([(120.0, 100.0)] * 5)) == 1.0
    assert fit_multiplicative_factor(_rows([(0.0, 0.0)] * 40)) == 1.0
    assert fit_multiplicative_factor(_rows([(50.0, 0.0)] * 40)) == 1.0


def test_non_finite_rows_are_excluded_before_fitting():
    frame = _rows([(np.nan, 100.0), (100.0, 50.0)] * 30)

    assert fit_multiplicative_factor(frame) == pytest.approx(1.5)


def test_group_factors_clamp_and_return_identity_for_missing_groups():
    frame = _rows(
        [(110.0, 100.0)] * 30
        + [(300.0, 100.0)] * 30
        + [(10.0, 100.0)] * 5,
        sbc_class="Smooth",
    )
    frame.loc[30:, "sbc_class"] = "Lumpy"
    frame.loc[60:, "sbc_class"] = "Thin"

    factors = fit_group_factors(frame, "sbc_class")

    assert factors["Smooth"] == pytest.approx(1.1)
    assert factors["Lumpy"] == 1.5
    assert factors["Thin"] == 1.0


def test_apply_factors_scales_only_the_point_forecast_without_mutating_input():
    frame = pd.DataFrame(
        {
            "forecast_qty": [10.0, 20.0],
            "forecast_p95": [15.0, 30.0],
            "sbc_class": ["Smooth", "Lumpy"],
        }
    )

    result = apply_factors(
        frame,
        factors={"Smooth": 1.1, "Lumpy": 1.5},
        group_column="sbc_class",
    )

    assert result["forecast_qty"].tolist() == pytest.approx([11.0, 30.0])
    assert result["forecast_p95"].tolist() == [15.0, 30.0]
    assert frame["forecast_qty"].tolist() == [10.0, 20.0]


def test_apply_factors_falls_back_to_identity_for_unmapped_groups():
    frame = pd.DataFrame({"forecast_qty": [10.0], "region": ["South"]})

    result = apply_factors(
        frame, factors={"North": 1.5}, group_column="region"
    )

    assert result["forecast_qty"].tolist() == [10.0]


def test_gate_requires_wape_gain_bias_limit_and_coverage_retention():
    baseline = {"lead_time_wape": 0.349, "lead_time_bias": -0.05, "p95_coverage": 0.974}

    passing = evaluate_candidate(
        baseline,
        {"lead_time_wape": 0.339, "lead_time_bias": 0.02, "p95_coverage": 0.970},
    )
    assert passing["passed"] is True

    no_margin = evaluate_candidate(
        baseline,
        {"lead_time_wape": 0.345, "lead_time_bias": 0.02, "p95_coverage": 0.970},
    )
    assert no_margin["passed"] is False
    assert "lead_time_wape_gain" in no_margin["reasons"][0]

    high_bias = evaluate_candidate(
        baseline,
        {"lead_time_wape": 0.330, "lead_time_bias": 0.12, "p95_coverage": 0.970},
    )
    assert high_bias["passed"] is False
    assert "abs_lead_time_bias" in high_bias["reasons"][0]

    degraded_coverage = evaluate_candidate(
        baseline,
        {"lead_time_wape": 0.330, "lead_time_bias": 0.02, "p95_coverage": 0.950},
    )
    assert degraded_coverage["passed"] is False
    assert "p95_coverage" in degraded_coverage["reasons"][0]


def test_gate_skips_coverage_when_champion_has_no_quantiles():
    result = evaluate_candidate(
        {"lead_time_wape": 0.349, "lead_time_bias": -0.05, "p95_coverage": np.nan},
        {"lead_time_wape": 0.339, "lead_time_bias": 0.02, "p95_coverage": np.nan},
    )

    assert result["passed"] is True


def test_volume_assignment_uses_tuning_totals_and_preserves_held_index():
    tuning = pd.DataFrame(
        {
            "part_id": ["P1", "P2", "P3", "P4"],
            "region": "North",
            "actual_qty": [10.0, 20.0, 30.0, 40.0],
            "forecast_qty": [10.0, 20.0, 30.0, 40.0],
        }
    )
    held = pd.DataFrame(
        {
            "part_id": ["P1", "P4", "P9"],
            "region": "North",
            "actual_qty": [1.0, 1.0, 1.0],
            "forecast_qty": [1.0, 1.0, 1.0],
        },
        index=[5, 7, 9],
    )

    labels = _volume_assignment(tuning, held)

    assert labels.index.tolist() == [5, 7, 9]
    assert labels.tolist() == ["Q1", "Q4", "unmapped"]


def test_published_factor_is_reproducible_from_tuning_only_evidence():
    summary_path = ROOT / "results" / "bias_calibration_summary.json"
    artifact_path = ROOT / "artifacts" / "bias_calibration_tuning_forecasts.csv.gz"
    if not summary_path.exists() or not artifact_path.exists():
        pytest.skip("Bias calibration outputs have not been generated")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    tuning = pd.read_csv(
        artifact_path, parse_dates=["week_start_date", "origin_date"]
    )
    lead_rows = _lead_time_rows(tuning, summary["extension_period_weeks"])

    factor = fit_multiplicative_factor(lead_rows)
    assert factor == pytest.approx(
        summary["variants"]["lead_pooled"]["factor"], rel=1e-9
    )

    observed = {
        value.date().isoformat()
        for value in pd.to_datetime(tuning["origin_date"]).unique()
    }
    assert observed.isdisjoint(summary["held_origins"])
    assert observed == set(summary["tuning_origins"])
