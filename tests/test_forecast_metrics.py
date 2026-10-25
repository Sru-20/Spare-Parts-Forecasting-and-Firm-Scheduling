import numpy as np
import pandas as pd
import pytest

from src.forecasting.baselines import forecast_baseline
from src.forecasting.metrics import forecast_metrics, naive_scale_by_series


def test_forecast_metrics_match_contract_and_zero_edges():
    result = forecast_metrics(
        [0, 10, 20],
        [0, 12, 16],
        mase_scale=[2, 2, 2],
        quantile_forecasts={
            0.90: [1, 15, 22],
            0.95: [2, 16, 24],
            0.99: [3, 18, 30],
        },
    )
    assert result["wape_pct"] == pytest.approx(20.0)
    assert result["forecast_accuracy_pct"] == pytest.approx(80.0)
    assert result["bias_pct"] == pytest.approx(-6.6666667)
    assert result["mae"] == pytest.approx(2.0)
    assert result["rmse"] == pytest.approx(np.sqrt(20 / 3))
    assert result["smape_pct"] == pytest.approx(200 * (2 / 22 + 4 / 36) / 3)
    assert result["mase"] == pytest.approx(1.0)
    assert result["p90_coverage"] == pytest.approx(1.0)

    all_zero = forecast_metrics([0, 0], [0, 0])
    assert np.isnan(all_zero["wape_pct"])
    assert np.isnan(all_zero["bias_pct"])
    assert all_zero["smape_pct"] == 0
    assert all_zero["tracking_signal"] == 0
    assert np.isnan(all_zero["mase"])


def test_empty_metrics_are_explicitly_undefined():
    result = forecast_metrics([], [])
    assert result["n_scored"] == 0
    assert np.isnan(result["wape_pct"])
    assert np.isnan(result["p99_coverage"])


def test_naive_and_intermittent_baselines_use_only_available_training_state():
    dates = pd.date_range("2024-01-01", periods=60, freq="W-MON")
    history = pd.DataFrame(
        {
            "part_id": "P1",
            "region": "North",
            "week_start_date": dates,
            "demand_qty": [0, 4, 0, 8, 0] * 12,
            "split_role": ["warmup"] * 20 + ["train_val"] * 40,
        }
    )
    future_dates = pd.date_range(dates[-1] + pd.Timedelta(days=7), periods=3, freq="W-MON")
    history.loc[19, "demand_qty"] = 999

    naive = forecast_baseline(history, future_dates, "naive")
    croston = forecast_baseline(history, future_dates, "croston")
    sba = forecast_baseline(history, future_dates, "sba")
    tsb = forecast_baseline(history, future_dates, "tsb")
    assert naive.tolist() == [0.0, 0.0, 0.0]
    assert np.all(croston >= 0)
    assert np.all(sba >= 0)
    assert np.all(tsb >= 0)
    assert np.all(sba < croston)

    scales = naive_scale_by_series(history)
    assert scales[("P1", "North")] > 0


def test_croston_intervals_count_missing_calendar_weeks_without_zero_filling():
    dates = pd.date_range("2025-01-06", periods=6, freq="W-MON")
    history = pd.DataFrame(
        {
            "part_id": ["P1"] * 6,
            "region": ["North"] * 6,
            "week_start_date": dates,
            "demand_qty": pd.array([10, 0, pd.NA, 0, 0, 10], dtype="Int64"),
            "split_role": ["train_val"] * 6,
        }
    )
    forecast = forecast_baseline(
        history, [dates[-1] + pd.Timedelta(days=7)], "croston"
    )
    assert forecast[0] == pytest.approx(7.1428571)
