"""Shared forecast metrics for weekly and lead-time evaluation."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


QUANTILES = (0.90, 0.95, 0.99)


def forecast_metrics(
    actual: Sequence[float],
    forecast: Sequence[float],
    mase_scale: Sequence[float] | None = None,
    quantile_forecasts: Mapping[float, Sequence[float]] | None = None,
) -> dict[str, float | int]:
    """Calculate frozen-contract metrics, leaving undefined results as NaN."""
    actual_values = np.asarray(actual, dtype=float)
    forecast_values = np.asarray(forecast, dtype=float)
    if actual_values.shape != forecast_values.shape:
        raise ValueError("actual and forecast arrays must have identical shapes")
    valid = np.isfinite(actual_values) & np.isfinite(forecast_values)
    actual_values = actual_values[valid]
    forecast_values = forecast_values[valid]
    count = len(actual_values)
    result: dict[str, float | int] = {
        "n_scored": count,
        "actual_total": float(np.sum(actual_values)) if count else np.nan,
        "forecast_total": float(np.sum(forecast_values)) if count else np.nan,
    }
    if not count:
        for name in (
            "wape_pct", "forecast_accuracy_pct", "bias_pct", "mae", "rmse",
            "smape_pct", "mase", "tracking_signal",
            "forecast_vs_actual_variance",
            "forecast_mean", "actual_mean", "pinball_loss",
            "mape_pct", "mape_nonzero_n", "mape_zero_excluded",
        ):
            result[name] = np.nan
        for quantile in QUANTILES:
            result[f"p{int(quantile * 100)}_coverage"] = np.nan
            result[f"p{int(quantile * 100)}_coverage_error"] = np.nan
        return result

    errors = forecast_values - actual_values
    absolute_error = np.abs(errors)
    actual_total = float(np.sum(actual_values))
    mae = float(np.mean(absolute_error))
    if actual_total == 0:
        wape = bias = np.nan
    else:
        wape = float(100.0 * np.sum(absolute_error) / actual_total)
        bias = float(100.0 * np.sum(errors) / actual_total)
    denominator = np.abs(actual_values) + np.abs(forecast_values)
    smape_terms = np.divide(
        200.0 * absolute_error,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    nonzero = actual_values != 0
    mape_terms = np.abs(errors[nonzero]) / np.abs(actual_values[nonzero])
    result.update(
        {
            "wape_pct": wape,
            "forecast_accuracy_pct": 100.0 - wape if np.isfinite(wape) else np.nan,
            "bias_pct": bias,
            "mae": mae,
            "rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "mape_pct": float(100.0 * np.mean(mape_terms))
            if len(mape_terms)
            else np.nan,
            "mape_nonzero_n": int(nonzero.sum()),
            "mape_zero_excluded": int(count - nonzero.sum()),
            "smape_pct": float(np.mean(smape_terms)),
            "tracking_signal": (
                float(np.sum(errors) / mae)
                if mae != 0
                else (0.0 if np.sum(errors) == 0 else np.nan)
            ),
            "forecast_vs_actual_variance": float(np.var(errors, ddof=0)),
            "forecast_mean": float(np.mean(forecast_values)),
            "actual_mean": float(np.mean(actual_values)),
        }
    )

    if mase_scale is None:
        result["mase"] = np.nan
    else:
        scales = np.asarray(mase_scale, dtype=float)
        if scales.shape != valid.shape:
            raise ValueError("mase_scale must align with actual and forecast arrays")
        scales = scales[valid]
        valid_scale = np.isfinite(scales) & (scales > 0)
        result["mase"] = (
            float(np.mean(absolute_error[valid_scale]) / np.mean(scales[valid_scale]))
            if valid_scale.any()
            else np.nan
        )

    pinball_values: list[float] = []
    quantile_forecasts = quantile_forecasts or {}
    for quantile in QUANTILES:
        name = f"p{int(quantile * 100)}"
        values = quantile_forecasts.get(quantile)
        if values is None:
            result[f"{name}_coverage"] = np.nan
            result[f"{name}_coverage_error"] = np.nan
            continue
        q_forecast = np.asarray(values, dtype=float)
        if q_forecast.shape != valid.shape:
            raise ValueError(f"{name} forecast must align with actual and forecast arrays")
        q_forecast = q_forecast[valid]
        q_valid = np.isfinite(q_forecast)
        if not q_valid.any():
            result[f"{name}_coverage"] = np.nan
            result[f"{name}_coverage_error"] = np.nan
            continue
        residual = actual_values[q_valid] - q_forecast[q_valid]
        pinball_values.extend(
            np.maximum(quantile * residual, (quantile - 1.0) * residual).tolist()
        )
        coverage = float(np.mean(actual_values[q_valid] <= q_forecast[q_valid]))
        result[f"{name}_coverage"] = coverage
        result[f"{name}_coverage_error"] = abs(coverage - quantile)
    result["pinball_loss"] = (
        float(np.mean(pinball_values)) if pinball_values else np.nan
    )
    return result


def naive_scale_by_series(history: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Return adjacent observed train_val one-step MAE by part-region series."""
    frame = history
    scales: dict[tuple[str, str], float] = {}
    for key, group in frame.groupby(
        ["part_id", "region"], sort=False, observed=True
    ):
        group = group.sort_values("week_start_date", kind="stable")
        observed = group.loc[
            group["split_role"].eq("train_val") & group["demand_qty"].notna(),
            ["week_start_date", "demand_qty"],
        ]
        adjacent = observed["week_start_date"].diff().dt.days.eq(7)
        differences = observed["demand_qty"].diff().abs().loc[adjacent]
        scale = float(differences.mean()) if not differences.empty else np.nan
        scales[(str(key[0]), str(key[1]))] = scale
    return scales
