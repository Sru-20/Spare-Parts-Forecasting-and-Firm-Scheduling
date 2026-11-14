import numpy as np
import pandas as pd
import pytest

from src.features.time_features import build_time_features
from src.forecasting.global_models import (
    _add_series_trend_features,
    _ratio_where_positive,
    _recursive_feature_frame,
)
from src.forecasting.model_experiments import (
    _apply_blend,
    _blend_weights,
    _pair_blend,
)


def _history(values, part="P1", region="North"):
    dates = pd.date_range("2024-01-01", periods=len(values), freq="W-MON")
    return pd.DataFrame(
        {
            "part_id": part,
            "region": region,
            "week_start_date": dates,
            "demand_qty": values,
        }
    )


def _augmented(history):
    return _add_series_trend_features(build_time_features(history))


def test_ratio_features_are_capped_and_ignore_non_positive_denominators():
    ratio = _ratio_where_positive(
        pd.Series([100.0, 5.0, 1.0]), pd.Series([0.01, 2.0, 0.0])
    )

    assert ratio.iloc[0] == 10.0
    assert ratio.iloc[1] == 2.5
    assert np.isnan(ratio.iloc[2])


def test_trend_features_use_only_rows_before_each_target_date():
    values = [1.0, 0.0, 2.0, 0.0, 3.0, 1.0, 0.0, 4.0] * 8
    history = _history(values)
    baseline = _augmented(history)

    changed = history.copy()
    changed.loc[changed.index >= 50, "demand_qty"] = 999.0
    modified = _augmented(changed)

    columns = [
        "demand_trend_4_13",
        "demand_trend_13_52",
        "demand_growth_4",
        "demand_cv_13",
        "demand_zero_rate_13",
        "demand_history_weeks",
    ]
    pd.testing.assert_frame_equal(
        baseline.loc[:49, columns], modified.loc[:49, columns]
    )


def test_recursive_features_match_training_features_at_the_origin():
    values = [2.0, 0.0, 1.0, 0.0, 3.0, 0.0, 2.0, 1.0, 0.0, 4.0] * 6
    history = _history(values)
    features = _augmented(history)
    target_row = features.iloc[[-1]]
    feature_columns = [
        column
        for column in features.columns
        if column not in {"week_start_date", "demand_qty"}
    ]
    current_calendar = target_row[["part_id", "region", "week_start_date"]].copy()

    recursive = _recursive_feature_frame(
        current_calendar,
        {("P1", "North"): values[:-1]},
        {("P1", "North"): None},
        feature_columns,
        extra_features=True,
    )

    for column in (
        "demand_trend_4_13",
        "demand_trend_13_52",
        "demand_growth_4",
        "demand_cv_13",
        "demand_zero_rate_13",
        "demand_history_weeks",
    ):
        assert recursive.iloc[0][column] == pytest.approx(
            target_row.iloc[0][column], rel=1e-9, abs=1e-9
        ), column


def _weekly_frame(model, forecast):
    return pd.DataFrame(
        {
            "model": model,
            "config_id": "test",
            "origin_date": pd.Timestamp("2025-01-13"),
            "part_id": "P1",
            "region": "North",
            "week_start_date": pd.Timestamp("2025-01-20"),
            "horizon_step": 1,
            "lead_time_weeks": 1,
            "actual_qty": [12.0],
            "forecast_qty": [float(forecast)],
            "mase_scale": [1.0],
        }
    )


def test_blend_weight_selects_the_lower_error_forecast():
    actual_near_tweedie = _weekly_frame("lightgbm_tweedie", 10.0)
    quantile = _weekly_frame("lightgbm_quantile", 20.0)
    weight, metrics = _blend_weights(
        actual_near_tweedie, quantile, extension_weeks=4
    )
    assert weight == 0.8
    assert metrics["lead_time_wape"] == pytest.approx(0.0)

    actual_near_quantile = _weekly_frame("lightgbm_tweedie", 10.0)
    actual_near_quantile["actual_qty"] = 18.0
    weight, metrics = _blend_weights(
        actual_near_quantile, quantile, extension_weeks=4
    )
    assert weight == 0.2
    assert metrics["lead_time_wape"] == pytest.approx(0.0)


def test_blend_application_is_weighted_and_does_not_mutate_inputs():
    tweedie = _weekly_frame("lightgbm_tweedie", 10.0)
    quantile = _weekly_frame("lightgbm_quantile", 20.0)
    original_tweedie = tweedie.copy()
    original_quantile = quantile.copy()

    paired = _pair_blend(tweedie, quantile)
    blended = _apply_blend(tweedie, quantile, 0.75)

    assert paired["forecast_qty_quantile"].tolist() == [20.0]
    assert blended["forecast_qty"].tolist() == pytest.approx([12.5])
    pd.testing.assert_frame_equal(tweedie, original_tweedie)
    pd.testing.assert_frame_equal(quantile, original_quantile)
