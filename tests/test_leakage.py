import numpy as np
import pandas as pd
import pytest

from src.features.time_features import (
    build_recursive_forecast_features,
    build_time_features,
    model_feature_columns,
)
from src.features.segmentation import classify_sbc_series


def _weekly_series(weeks=80):
    dates = pd.date_range("2024-01-01", periods=weeks, freq="W-MON")
    demand = [0, 0, 4, 0, 7, 2, 0, 0, 5, 1] * (weeks // 10)
    demand.extend([0] * (weeks - len(demand)))
    return pd.DataFrame(
        {
            "part_id": "P1",
            "region": "North",
            "week_start_date": dates,
            "demand_qty": demand,
            "month": dates.month,
            "promotion_flag": [0] * weeks,
            "market_condition_index": np.arange(weeks, dtype=float),
            "out_stockout_flag": [0] * weeks,
        }
    )


def test_features_are_invariant_to_history_after_origin():
    frame = _weekly_series()
    origin = 60
    full_features = build_time_features(frame)
    truncated_features = build_time_features(frame.iloc[: origin + 1])
    target_date = frame.loc[origin, "week_start_date"]
    columns = [
        column for column in full_features.columns
        if column not in {"demand_qty"}
    ]
    pd.testing.assert_series_equal(
        full_features.loc[origin, columns],
        truncated_features.loc[origin, columns],
        check_names=False,
    )

    changed = frame.copy()
    changed.loc[origin + 1 :, "demand_qty"] = 10_000
    changed_features = build_time_features(changed)
    pd.testing.assert_series_equal(
        full_features.loc[origin, columns],
        changed_features.loc[origin, columns],
        check_names=False,
    )
    assert full_features.loc[origin, "week_start_date"] == target_date


def test_lags_and_rolling_statistics_use_only_prior_observations():
    frame = _weekly_series()
    features = build_time_features(frame)
    row = 40
    previous = frame.loc[row - 1, "demand_qty"]
    prior_4 = frame.loc[row - 4 : row - 1, "demand_qty"]
    assert features.loc[row, "demand_lag_1"] == previous
    assert features.loc[row, "demand_rolling_4_mean"] == pytest.approx(
        prior_4.mean()
    )
    assert features.loc[row, "demand_rolling_4_median"] == pytest.approx(
        prior_4.median()
    )
    assert features.loc[row, "demand_rolling_4_std"] == pytest.approx(
        prior_4.std(ddof=1)
    )


def test_default_features_exclude_targets_outputs_and_unapproved_covariates():
    features = build_time_features(_weekly_series())
    columns = model_feature_columns(features)
    assert "demand_lag_1" in columns
    assert "month" in columns
    assert "market_condition_index" not in columns
    assert "out_stockout_flag" not in columns
    assert "demand_qty" not in columns


def test_conditional_covariates_require_explicit_origin_approval():
    frame = _weekly_series()
    without_approval = build_time_features(frame)
    with_approval = build_time_features(
        frame, origin_known_columns=("market_condition_index",)
    )
    assert "market_condition_index" not in model_feature_columns(without_approval)
    assert "market_condition_index" in model_feature_columns(with_approval)
    with pytest.raises(ValueError, match="allowed name"):
        build_time_features(frame, origin_known_columns=("out_stockout_flag",))


def test_recursive_forecast_uses_prior_predictions_for_subsequent_lags():
    history = _weekly_series(60).drop(columns=["market_condition_index", "out_stockout_flag"])
    future_dates = pd.date_range(
        history["week_start_date"].max() + pd.Timedelta(days=7),
        periods=3,
        freq="W-MON",
    )
    calendar = pd.DataFrame(
        {
            "part_id": ["P1"] * 3,
            "region": ["North"] * 3,
            "week_start_date": future_dates,
            "month": future_dates.month,
            "promotion_flag": [0, 1, 0],
        }
    )
    seen_lags = []

    def predictor(features):
        lag = float(features.iloc[0]["demand_lag_1"])
        seen_lags.append(lag)
        return lag + 1

    result = build_recursive_forecast_features(history, calendar, predictor)
    assert result["forecast_demand_qty"].tolist() == [
        float(history.iloc[-1]["demand_qty"]) + 1,
        float(history.iloc[-1]["demand_qty"]) + 2,
        float(history.iloc[-1]["demand_qty"]) + 3,
    ]
    assert seen_lags[1] == result.iloc[0]["forecast_demand_qty"]
    assert seen_lags[2] == result.iloc[1]["forecast_demand_qty"]


def test_future_labels_and_outcomes_are_rejected():
    history = _weekly_series(60)
    date = history["week_start_date"].max() + pd.Timedelta(days=7)
    future = pd.DataFrame(
        {
            "part_id": ["P1"],
            "region": ["North"],
            "week_start_date": [date],
            "demand_qty": [42],
        }
    )
    with pytest.raises(ValueError, match="must not include demand labels"):
        build_recursive_forecast_features(history, future, lambda _: 0)
    future = future.drop(columns=["demand_qty"])
    future["out_spike_flag"] = 0
    with pytest.raises(ValueError, match="forbidden columns"):
        build_recursive_forecast_features(history, future, lambda _: 0)


def test_sbc_thresholds_and_sparse_series_match_generator_rules():
    smooth = classify_sbc_series(pd.Series([10.0] * 30))
    assert smooth["sbc_class"] == "Smooth"
    intermittent = classify_sbc_series(
        pd.Series([10.0, 0.0] * 20)
    )
    assert intermittent["sbc_class"] == "Intermittent"
    short = classify_sbc_series(pd.Series([1.0] * 25))
    assert pd.isna(short["sbc_class"])
    sparse = classify_sbc_series(pd.Series([1.0] + [0.0] * 29))
    assert sparse["sbc_class"] == "Lumpy"
    assert sparse["classification_status"] == "classified_low_positive_count"
