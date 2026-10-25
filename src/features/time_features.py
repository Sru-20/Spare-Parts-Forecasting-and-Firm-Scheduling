"""Time-aware demand features built strictly from origin-visible observations."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from math import isfinite
from math import pi

import numpy as np
import pandas as pd


CALENDAR_FEATURES = (
    "year", "month", "quarter_fy", "fiscal_year", "week_of_year",
    "week_of_month", "festival_flag", "festival_name",
    "festival_intensity", "weeks_to_next_festival",
    "weeks_since_last_festival", "public_holiday_flag", "monsoon_flag",
    "summer_peak_flag", "fiscal_year_end_flag", "dealer_order_cycle_flag",
    "promotion_flag", "promotion_intensity", "campaign_flag",
    "campaign_intensity",
)
OPTIONAL_ORIGIN_KNOWN_COVARIATES = frozenset({
    "plant_shutdown_flag", "plant_working_days", "weeks_since_last_shutdown",
    "market_event_flag", "market_condition_index", "vehicle_parc",
    "avg_vehicle_age_years", "lifecycle_status", "criticality",
    "abc_class", "xyz_class", "substitution_group",
})
STATIC_FEATURES = (
    "part_id", "region", "component", "vehicle_system", "part_category",
    "vehicle_model", "vehicle_type", "criticality", "make_or_buy",
    "is_imported", "unit_cost", "fulfilment_mode",
)
FORBIDDEN_FEATURES = frozenset({
    "demand_qty", "demand_qty_capped", "is_missing_week", "split_role",
    "intended_sbc_class", "demand_driver_tags", "lifecycle_pattern",
    "demand_class_trainval", "adi_trainval", "cv2_trainval", "part_type",
    "out_consumption_qty", "out_backorder_qty", "out_lost_sales_qty",
    "out_stockout_flag", "out_spike_flag", "out_unusual_demand_flag",
})


def _validate_inputs(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    date_column: str,
    target_column: str,
    origin_known_columns: Sequence[str],
) -> None:
    required = {*group_columns, date_column, target_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Feature input is missing required columns: {sorted(missing)}")
    if frame[list(group_columns) + [date_column]].isna().any().any():
        raise ValueError("Series keys and dates must not contain missing values")
    if frame.duplicated([*group_columns, date_column]).any():
        raise ValueError("Feature input contains duplicate series-date keys")
    if target_column in origin_known_columns:
        raise ValueError("The target is not an origin-known covariate")
    invalid = set(origin_known_columns) - OPTIONAL_ORIGIN_KNOWN_COVARIATES
    if invalid:
        raise ValueError(
            "Conditional covariates require explicit review and an allowed name; "
            f"invalid={sorted(invalid)}"
        )
    leaked = set(origin_known_columns) & FORBIDDEN_FEATURES
    if leaked or any(name.startswith("out_") for name in origin_known_columns):
        raise ValueError(f"Forbidden feature requested: {sorted(leaked)}")


def build_time_features(
    frame: pd.DataFrame,
    group_columns: Sequence[str] = ("part_id", "region"),
    date_column: str = "week_start_date",
    target_column: str = "demand_qty",
    origin_known_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Create one-step features; every target-derived value uses rows before t.

    The returned target is retained as a label, not as a feature. Calendar
    features are passed through because their values are known for the target
    date. Conditional covariates require explicit approval by the caller.
    """
    _validate_inputs(
        frame, group_columns, date_column, target_column, origin_known_columns
    )
    ordered = frame.sort_values(
        [*group_columns, date_column], kind="stable"
    ).copy()
    count = len(ordered)
    target_values = ordered[target_column].to_numpy(dtype=float, na_value=np.nan)
    dates = pd.DatetimeIndex(ordered[date_column]).asi8 // (86_400 * 1_000_000_000)
    generated: dict[str, np.ndarray] = {}
    for lag in (1, 4, 13, 52):
        generated[f"demand_lag_{lag}"] = np.full(count, np.nan)
    for window in (4, 13, 52):
        for statistic in ("mean", "median", "std"):
            generated[f"demand_rolling_{window}_{statistic}"] = np.full(
                count, np.nan
            )
    generated["demand_same_period_last_year"] = np.full(count, np.nan)
    generated["demand_zero_run_length"] = np.zeros(count, dtype=float)
    generated["weeks_since_last_demand"] = np.full(count, np.nan)

    groups = ordered.groupby(
        list(group_columns), sort=False, observed=True
    ).indices.values()
    for positions in groups:
        values = target_values[positions]
        series = pd.Series(values, dtype="float64")
        for lag in (1, 4, 13, 52):
            generated[f"demand_lag_{lag}"][positions] = (
                series.shift(lag).to_numpy(dtype=float, na_value=np.nan)
            )
        generated["demand_same_period_last_year"][positions] = (
            series.shift(52).to_numpy(dtype=float, na_value=np.nan)
        )
        prior = series.shift(1)
        for window in (4, 13, 52):
            rolling = prior.rolling(window=window, min_periods=1)
            generated[f"demand_rolling_{window}_mean"][positions] = (
                rolling.mean().to_numpy(dtype=float, na_value=np.nan)
            )
            generated[f"demand_rolling_{window}_median"][positions] = (
                rolling.median().to_numpy(dtype=float, na_value=np.nan)
            )
            generated[f"demand_rolling_{window}_std"][positions] = (
                rolling.std(ddof=1).to_numpy(dtype=float, na_value=np.nan)
            )

        zero_run = 0
        last_positive_day: int | None = None
        for local_position, value in enumerate(values):
            position = positions[local_position]
            current_day = int(dates[position])
            generated["demand_zero_run_length"][position] = float(zero_run)
            if last_positive_day is not None:
                generated["weeks_since_last_demand"][position] = float(
                    (current_day - last_positive_day) // 7
                )
            if np.isnan(value):
                zero_run = 0
            elif value > 0:
                zero_run = 0
                last_positive_day = current_day
            else:
                zero_run += 1

    passthrough = set(CALENDAR_FEATURES) | set(origin_known_columns)
    copied_columns = list(dict.fromkeys(
        column
        for column in (
            *group_columns,
            date_column,
            target_column,
            *STATIC_FEATURES,
            *sorted(passthrough),
        )
        if column in ordered.columns
    ))
    features = pd.concat(
        [
            ordered.loc[:, copied_columns],
            pd.DataFrame(generated, index=ordered.index),
        ],
        axis=1,
    )
    angle = 2.0 * pi * ordered[date_column].dt.month / 12.0
    features["demand_month_sin"] = np.sin(angle)
    features["demand_month_cos"] = np.cos(angle)
    return features.sort_index()


def model_feature_columns(feature_frame: pd.DataFrame) -> list[str]:
    """Return feature names only, excluding labels, keys and all forbidden data."""
    excluded = FORBIDDEN_FEATURES | {
        "week_start_date", "is_missing_week", "split_role",
    }
    return [
        column for column in feature_frame.columns
        if column not in excluded
        and not column.startswith("out_")
    ]


def build_recursive_forecast_features(
    history: pd.DataFrame,
    future_calendar: pd.DataFrame,
    predict_one_step: Callable[[pd.DataFrame], float],
    group_columns: Sequence[str] = ("part_id", "region"),
    date_column: str = "week_start_date",
    target_column: str = "demand_qty",
    origin_known_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Recursively forecast future rows, feeding only prior predictions to lags.

    `future_calendar` must contain series keys, dates, and known covariates only;
    it must not contain a target, outcomes, or test/future labels.
    """
    if target_column in future_calendar.columns:
        raise ValueError("future_calendar must not include demand labels")
    forbidden = [
        column for column in future_calendar.columns
        if column.startswith("out_") or column in FORBIDDEN_FEATURES
    ]
    if forbidden:
        raise ValueError(f"future_calendar contains forbidden columns: {forbidden}")
    _validate_inputs(
        history, group_columns, date_column, target_column, origin_known_columns
    )
    missing_keys = {*group_columns, date_column} - set(future_calendar.columns)
    if missing_keys:
        raise ValueError(
            f"future_calendar is missing keys: {sorted(missing_keys)}"
        )
    unexpected = set(future_calendar.columns) - (
        {*group_columns, date_column}
        | set(STATIC_FEATURES)
        | set(CALENDAR_FEATURES)
        | set(origin_known_columns)
    )
    if unexpected:
        raise ValueError(
            f"future_calendar contains unapproved columns: {sorted(unexpected)}"
        )

    history_columns = list(dict.fromkeys(
        column for column in (
            *group_columns,
            date_column,
            target_column,
            *STATIC_FEATURES,
            *CALENDAR_FEATURES,
            *origin_known_columns,
        )
        if column in history.columns
    ))
    history = history.loc[:, history_columns].copy()
    history[target_column] = pd.to_numeric(
        history[target_column], errors="raise"
    ).astype("Float64")
    histories = {
        key if isinstance(key, tuple) else (key,): group.copy()
        for key, group in history.groupby(
            list(group_columns), sort=False, observed=True
        )
    }
    forecast_rows: list[dict[str, object]] = []
    ordered_future = future_calendar.sort_values(
        [*group_columns, date_column], kind="stable"
    )
    for _, calendar_row in ordered_future.iterrows():
        key = tuple(calendar_row[column] for column in group_columns)
        if key not in histories:
            raise ValueError(f"No observed history for forecast series {key}")
        past = histories[key]
        row = calendar_row.to_dict()
        future_row = pd.DataFrame([row])
        future_row[target_column] = pd.Series(
            pd.array([pd.NA], dtype="Float64"),
            index=future_row.index,
        )
        augmented = pd.concat([past, future_row], ignore_index=True)
        features = build_time_features(
            augmented,
            group_columns=group_columns,
            date_column=date_column,
            target_column=target_column,
            origin_known_columns=origin_known_columns,
        ).tail(1)
        forecast = float(predict_one_step(features))
        if not isfinite(forecast) or forecast < 0:
            raise ValueError(
                f"One-step predictor returned invalid demand forecast {forecast}"
            )
        forecast_rows.append(
            {
                **{
                    column: calendar_row[column]
                    for column in future_calendar.columns
                },
                "forecast_demand_qty": forecast,
            }
        )
        appended = calendar_row.to_dict()
        appended[target_column] = forecast
        forecast_row = pd.DataFrame([appended])
        forecast_row[target_column] = pd.Series(
            pd.array([forecast], dtype="Float64"),
            index=forecast_row.index,
        )
        histories[key] = pd.concat([past, forecast_row], ignore_index=True)
    return pd.DataFrame(forecast_rows)
