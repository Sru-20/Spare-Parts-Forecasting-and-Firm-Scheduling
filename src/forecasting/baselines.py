"""Simple and intermittent-demand benchmark forecasts."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


BASELINE_MODELS = (
    "naive",
    "seasonal_naive_52",
    "moving_average_13",
    "same_period_last_year",
    "croston",
    "sba",
    "tsb",
    "exponential_smoothing",
)


def _observed_history(history: pd.DataFrame) -> pd.DataFrame:
    """Select observed train-validation labels for fitted baseline states."""
    return history.loc[
        history["split_role"].eq("train_val")
        & history["demand_qty"].notna()
    ].sort_values("week_start_date", kind="stable")


def _croston(values: np.ndarray, alpha: float, bias_corrected: bool) -> float:
    positive_indices = np.flatnonzero(values > 0)
    if not len(positive_indices):
        return 0.0
    first = int(positive_indices[0])
    demand_estimate = float(values[first])
    interval_estimate = float(first + 1)
    previous = first
    for index in positive_indices[1:]:
        interval = float(index - previous)
        demand_estimate += alpha * (float(values[index]) - demand_estimate)
        interval_estimate += alpha * (interval - interval_estimate)
        previous = int(index)
    forecast = demand_estimate / interval_estimate
    return forecast * (1.0 - alpha / 2.0) if bias_corrected else forecast


def _ses(values: np.ndarray) -> float:
    """ETS(A,N,N): simple exponential smoothing with fit-time alpha."""
    observed = values[np.isfinite(values)]
    if not len(observed):
        return np.nan
    if len(observed) < 2:
        return float(observed[-1])

    def one_step_sse(alpha: float) -> float:
        level = float(observed[0])
        total = 0.0
        for value in observed[1:]:
            total += (float(value) - level) ** 2
            level = alpha * float(value) + (1.0 - alpha) * level
        return total

    result = minimize_scalar(
        one_step_sse, bounds=(0.01, 0.99), method="bounded"
    )
    alpha = float(result.x)
    level = float(observed[0])
    for value in observed[1:]:
        level = alpha * float(value) + (1.0 - alpha) * level
    return max(0.0, level)


def _tsb(values: np.ndarray, alpha: float, beta: float) -> float:
    positive_indices = np.flatnonzero(values > 0)
    if not len(positive_indices):
        return 0.0
    first = int(positive_indices[0])
    demand_estimate = float(values[first])
    probability = 1.0 / float(first + 1)
    for value in values[first + 1 :]:
        if not np.isfinite(value):
            continue
        occurrence = 1.0 if value > 0 else 0.0
        probability += beta * (occurrence - probability)
        if occurrence:
            demand_estimate += alpha * (float(value) - demand_estimate)
    return demand_estimate * probability


def forecast_baseline(
    history: pd.DataFrame,
    target_dates: Sequence[pd.Timestamp],
    model: str,
    moving_average_window: int = 13,
    croston_alpha: float = 0.10,
    tsb_alpha: float = 0.10,
    tsb_beta: float = 0.10,
) -> np.ndarray:
    """Forecast one series without using observations after its origin."""
    if model not in BASELINE_MODELS:
        raise ValueError(f"Unknown baseline model: {model}")
    dates = pd.DatetimeIndex(pd.to_datetime(target_dates))
    if model in {"seasonal_naive_52", "same_period_last_year"}:
        indexed = history.set_index("week_start_date")["demand_qty"]
        all_history = history.loc[history["demand_qty"].notna()]
        fallback = float(all_history["demand_qty"].iloc[-1]) if len(all_history) else np.nan
        forecasts = []
        for target_date in dates:
            reference = target_date - pd.Timedelta(weeks=52)
            value = indexed.get(reference, pd.NA)
            forecasts.append(
                float(value) if pd.notna(value) else fallback
            )
        return np.asarray(forecasts, dtype=float)

    trainval = history.loc[history["split_role"].eq("train_val")].sort_values(
        "week_start_date", kind="stable"
    )
    calendar_values = trainval["demand_qty"].to_numpy(dtype=float, na_value=np.nan)
    observed_values = calendar_values[np.isfinite(calendar_values)]
    if not len(observed_values):
        return np.full(len(dates), np.nan, dtype=float)
    if model == "naive":
        estimate = float(observed_values[-1])
    elif model == "moving_average_13":
        recent_calendar_weeks = trainval.tail(moving_average_window)
        recent_observed = recent_calendar_weeks["demand_qty"].dropna().astype(float)
        estimate = (
            float(recent_observed.mean())
            if not recent_observed.empty
            else float(observed_values[-1])
        )
    elif model == "croston":
        estimate = _croston(calendar_values, croston_alpha, bias_corrected=False)
    elif model == "sba":
        estimate = _croston(calendar_values, croston_alpha, bias_corrected=True)
    elif model == "exponential_smoothing":
        estimate = _ses(calendar_values)
    else:
        estimate = _tsb(calendar_values, tsb_alpha, tsb_beta)
    return np.full(len(dates), max(0.0, estimate), dtype=float)
