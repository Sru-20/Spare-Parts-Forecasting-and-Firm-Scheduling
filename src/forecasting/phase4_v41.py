"""SPEC v4.1 Track A selection using held-validation forecasts only."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from src.data.loader import ROOT, load_demand, load_scheduling
from src.forecasting.backtest import (
    _fold_data,
    _lead_time_map,
)
from src.forecasting.baselines import forecast_baseline
from src.forecasting.metrics import QUANTILES, forecast_metrics


FORECAST_ARTIFACT = "artifacts/track_a_backtest_forecasts.parquet"
OUTPUT_FORECASTS = "artifacts/track_a_backtest_forecasts_v41.parquet"
SERIES_KEYS = ("part_id", "region")
QUANTILE_COLUMNS = {
    0.90: "forecast_p90",
    0.95: "forecast_p95",
    0.99: "forecast_p99",
}
METRIC_DIMENSIONS = (
    "sbc_class",
    "abc_class",
    "criticality",
    "region",
    "part_category",
    "horizon_step",
    "lead_time_weeks",
)
COMPOSITE_MODELS = {"simple_average_ensemble", "sbc_segment_router"}
SIMPLEX_COMPLEXITY = {
    "naive": (3, 3),
    "seasonal_naive_52": (3, 3),
    "historical_average": (3, 3),
    "sba": (3, 3),
    "tsb": (3, 3),
    "croston": (3, 3),
    "moving_average_13": (3, 3),
    "same_period_last_year": (3, 3),
    "lightgbm_point": (2, 2),
    "lightgbm_tweedie": (2, 2),
    "lightgbm_quantile": (2, 2),
    "simple_average_ensemble": (1, 1),
    "sbc_segment_router": (1, 1),
}


def _read_config(root: Path) -> dict[str, Any]:
    with (root / "configs" / "project.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Project configuration must be a mapping")
    return config


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0 or not np.isfinite(denominator):
        return np.nan
    return float(numerator / denominator)


def _pooled_wape(frame: pd.DataFrame) -> float:
    actual = pd.to_numeric(frame["actual_qty"], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    forecast = pd.to_numeric(frame["forecast_qty"], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    valid = np.isfinite(actual) & np.isfinite(forecast)
    if not valid.any():
        return np.nan
    return _safe_ratio(
        float(np.abs(actual[valid] - forecast[valid]).sum()),
        float(actual[valid].sum()),
    )


def _add_historical_average(
    forecasts: pd.DataFrame,
    demand: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build an origin-visible, flat mean forecast without using held labels."""
    base_rows = forecasts.loc[
        forecasts["candidate_role"].eq("forecast_candidate")
    ].drop_duplicates(
        ["part_id", "region", "week_start_date", "origin_date"],
        keep="first",
    )
    outputs: list[pd.DataFrame] = []
    runtimes: list[dict[str, object]] = []
    for origin_value, group in base_rows.groupby("origin_date", sort=True):
        origin = pd.Timestamp(origin_value)
        started = time.perf_counter()
        history = demand.loc[
            demand["split_role"].eq("train_val")
            & demand["week_start_date"].le(origin)
            & demand["demand_qty"].notna(),
            [*SERIES_KEYS, "demand_qty"],
        ]
        means = history.groupby(
            list(SERIES_KEYS), observed=True, sort=False
        )["demand_qty"].mean()
        result = group.copy()
        result["forecast_qty"] = [
            means.get((part_id, region), np.nan)
            for part_id, region in zip(result["part_id"], result["region"])
        ]
        elapsed = time.perf_counter() - started
        for column in ("forecast_p90", "forecast_p95", "forecast_p99"):
            result[column] = np.nan
        result["model"] = "historical_average"
        result["config_id"] = "origin_mean"
        result["candidate_role"] = "forecast_candidate"
        result["runtime_seconds"] = elapsed
        outputs.append(result)
        runtimes.append(
            {"model": "historical_average", "origin_date": origin, "runtime_seconds": elapsed}
        )
    if not outputs:
        raise ValueError("No held-validation forecast rows available for the baseline")
    historical = pd.concat(outputs, ignore_index=True)
    return pd.concat([forecasts, historical], ignore_index=True), pd.DataFrame(runtimes)


def _lead_time_rows(
    forecasts: pd.DataFrame,
    extension_period_weeks: int,
) -> pd.DataFrame:
    """Aggregate observed lead-time windows and label the >14-week extension."""
    required = {
        "model", "config_id", "origin_date", "part_id", "region",
        "horizon_step", "lead_time_weeks", "actual_qty", "forecast_qty",
    }
    absent = required - set(forecasts.columns)
    if absent:
        raise ValueError(f"Forecast rows lack lead-time fields: {sorted(absent)}")
    keys = ["model", "config_id", "origin_date", *SERIES_KEYS]
    quantile_columns = list(QUANTILE_COLUMNS.values())
    work = forecasts.copy()
    work["horizon_step"] = pd.to_numeric(work["horizon_step"], errors="coerce")
    work["lead_time_weeks"] = pd.to_numeric(
        work["lead_time_weeks"], errors="coerce"
    )
    work["actual_qty"] = pd.to_numeric(work["actual_qty"], errors="coerce")
    work["forecast_qty"] = pd.to_numeric(work["forecast_qty"], errors="coerce")
    valid_actual = np.isfinite(
        work["actual_qty"].to_numpy(dtype=float, na_value=np.nan)
    )
    in_window = work["horizon_step"].le(work["lead_time_weeks"].clip(upper=14))
    selected = work.loc[valid_actual & in_window].copy()
    if selected.empty:
        return pd.DataFrame()
    for column in quantile_columns:
        if column in selected:
            selected[column] = pd.to_numeric(
                selected[column], errors="coerce"
            ).replace([np.inf, -np.inf], np.nan)
    selected["forecast_qty"] = selected["forecast_qty"].replace(
        [np.inf, -np.inf], np.nan
    )
    aggregations: dict[str, tuple[str, str]] = {
        "lead_time_weeks": ("lead_time_weeks", "first"),
        "expected_rows": ("actual_qty", "size"),
        "forecast_rows": ("forecast_qty", "count"),
        "actual_qty": ("actual_qty", "sum"),
        "forecast_qty": ("forecast_qty", "sum"),
    }
    if "mase_scale" in selected:
        selected["mase_scale"] = pd.to_numeric(
            selected["mase_scale"], errors="coerce"
        )
        aggregations["mase_scale"] = ("mase_scale", "first")
    for dimension in METRIC_DIMENSIONS:
        if dimension in selected and dimension not in keys:
            aggregations[dimension] = (dimension, "first")
    for column in quantile_columns:
        if column in selected:
            aggregations[f"{column}_sum"] = (column, "sum")
            aggregations[f"{column}_rows"] = (column, "count")
    grouped = (
        selected.groupby(keys, sort=False, dropna=False, observed=True)
        .agg(**aggregations)
        .reset_index()
    )
    for column in ("actual_qty", "forecast_qty", *quantile_columns):
        if column in grouped:
            grouped[column] = pd.to_numeric(
                grouped[column], errors="coerce"
            ).astype(float)
    grouped = grouped.loc[grouped["forecast_rows"].eq(grouped["expected_rows"])].copy()
    if grouped.empty:
        return pd.DataFrame()
    grouped["lead_time_extension_weeks"] = (
        grouped["lead_time_weeks"].astype(int).sub(14).clip(lower=0)
    )
    extension_keys = grouped.loc[
        grouped["lead_time_extension_weeks"].gt(0), keys
    ]
    if not extension_keys.empty:
        last_period = selected.merge(
            extension_keys, on=keys, how="inner", validate="many_to_one"
        )
        last_period = last_period.loc[
            last_period["horizon_step"].ge(15 - extension_period_weeks)
        ]
        extension_aggregations: dict[str, tuple[str, str]] = {
            "extension_rows": ("actual_qty", "size"),
            "extension_forecast_rows": ("forecast_qty", "count"),
            "actual_extension_mean": ("actual_qty", "mean"),
            "forecast_extension_mean": ("forecast_qty", "mean"),
        }
        for column in quantile_columns:
            if column in last_period:
                extension_aggregations[f"{column}_extension_mean"] = (
                    column,
                    "mean",
                )
                extension_aggregations[f"{column}_extension_rows"] = (
                    column,
                    "count",
                )
        extension = (
            last_period.groupby(keys, sort=False, dropna=False, observed=True)
            .agg(**extension_aggregations)
            .reset_index()
        )
        grouped = grouped.merge(
            extension, on=keys, how="left", validate="one_to_one"
        )
        grouped = grouped.loc[
            grouped["lead_time_extension_weeks"].eq(0)
            | (
                grouped["extension_rows"].gt(0)
                & grouped["extension_rows"].eq(
                    grouped["extension_forecast_rows"]
                )
            )
        ].copy()
        multiplier = grouped["lead_time_extension_weeks"]
        extended = multiplier.gt(0)
        grouped.loc[extended, "actual_qty"] += (
            grouped.loc[extended, "actual_extension_mean"]
            * multiplier.loc[extended]
        )
        grouped.loc[extended, "forecast_qty"] += (
            grouped.loc[extended, "forecast_extension_mean"]
            * multiplier.loc[extended]
        )
        for column in quantile_columns:
            sum_column = f"{column}_sum"
            row_column = f"{column}_rows"
            extension_mean = f"{column}_extension_mean"
            extension_count = f"{column}_extension_rows"
            if sum_column not in grouped:
                grouped[column] = np.nan
                continue
            quantile_complete = grouped[row_column].eq(grouped["expected_rows"])
            grouped[column] = grouped[sum_column].where(quantile_complete)
            if extension_mean in grouped:
                can_extend = (
                    grouped["lead_time_extension_weeks"].eq(0)
                    | grouped[extension_count].eq(grouped["extension_rows"])
                )
                grouped.loc[extended & can_extend, column] += (
                    grouped.loc[extended & can_extend, extension_mean]
                    * multiplier.loc[extended & can_extend]
                )
                grouped.loc[extended & ~can_extend, column] = np.nan
    else:
        grouped["lead_time_extension_weeks"] = 0
        for column in quantile_columns:
            sum_column = f"{column}_sum"
            row_column = f"{column}_rows"
            if sum_column in grouped:
                grouped[column] = grouped[sum_column].where(
                    grouped[row_column].eq(grouped["expected_rows"])
                )
            else:
                grouped[column] = np.nan
    return grouped.drop(
        columns=[
            column
            for column in grouped.columns
            if column.endswith(("_sum", "_rows"))
            or column.startswith("actual_extension_")
            or column.startswith("forecast_extension_")
            or column.endswith("_extension_mean")
            or column.endswith("_extension_rows")
            or column in {"expected_rows", "forecast_rows", "extension_rows",
                          "extension_forecast_rows"}
        ],
        errors="ignore",
    )


def _metric_values(frame: pd.DataFrame) -> dict[str, float | int]:
    actual = pd.to_numeric(frame["actual_qty"], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    forecast = pd.to_numeric(frame["forecast_qty"], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    scale = (
        pd.to_numeric(frame["mase_scale"], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        if "mase_scale" in frame
        else None
    )
    quantiles = {
        quantile: pd.to_numeric(frame[column], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        for quantile, column in QUANTILE_COLUMNS.items()
        if column in frame and frame[column].notna().any()
    }
    result = forecast_metrics(actual, forecast, scale, quantiles)
    for old_name, new_name in (
        ("wape_pct", "wape_ratio"),
        ("bias_pct", "bias_ratio"),
        ("smape_pct", "smape_ratio"),
    ):
        value = result[old_name]
        result[new_name] = float(value) / 100.0 if np.isfinite(value) else np.nan
    result["pinball_loss_qty_units"] = result.pop("pinball_loss")
    return result


def _aggregate_metric_groups(
    source: pd.DataFrame,
    level: str,
    segment_dimension: str,
    group_columns: list[str],
) -> pd.DataFrame:
    actual = pd.to_numeric(source["actual_qty"], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    forecast = pd.to_numeric(source["forecast_qty"], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    valid = np.isfinite(actual) & np.isfinite(forecast)
    if not valid.any():
        return pd.DataFrame()
    work = source.loc[valid, group_columns].copy()
    work["actual_qty"] = actual[valid]
    work["forecast_qty"] = forecast[valid]
    work["error"] = work["forecast_qty"] - work["actual_qty"]
    work["absolute_error"] = work["error"].abs()
    work["squared_error"] = work["error"] ** 2
    denominator = work["actual_qty"].abs() + work["forecast_qty"].abs()
    work["smape_term"] = np.divide(
        2.0 * work["absolute_error"].to_numpy(dtype=float),
        denominator.to_numpy(dtype=float),
        out=np.zeros(len(work), dtype=float),
        where=denominator.to_numpy(dtype=float) != 0,
    )
    scale = (
        pd.to_numeric(source.loc[valid, "mase_scale"], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        if "mase_scale" in source
        else np.full(len(work), np.nan, dtype=float)
    )
    valid_scale = np.isfinite(scale) & (scale > 0)
    work["mase_abs_error"] = np.where(
        valid_scale, work["absolute_error"], 0.0
    )
    work["mase_scale_value"] = np.where(valid_scale, scale, 0.0)
    work["mase_count"] = valid_scale.astype(float)
    for quantile, column in QUANTILE_COLUMNS.items():
        quantile_forecast = (
            pd.to_numeric(source.loc[valid, column], errors="coerce").to_numpy(
                dtype=float, na_value=np.nan
            )
            if column in source
            else np.full(len(work), np.nan, dtype=float)
        )
        q_valid = np.isfinite(quantile_forecast)
        work[f"p{int(quantile * 100)}_count"] = q_valid.astype(float)
        work[f"p{int(quantile * 100)}_hits"] = (
            (work["actual_qty"].to_numpy(dtype=float) <= quantile_forecast)
            & q_valid
        ).astype(float)
        residual = work["actual_qty"].to_numpy(dtype=float) - quantile_forecast
        work[f"p{int(quantile * 100)}_pinball"] = np.where(
            q_valid,
            np.maximum(quantile * residual, (quantile - 1.0) * residual),
            0.0,
        )
    aggregations: dict[str, tuple[str, str]] = {
        "n_scored": ("actual_qty", "size"),
        "actual_total": ("actual_qty", "sum"),
        "forecast_total": ("forecast_qty", "sum"),
        "sum_error": ("error", "sum"),
        "sum_abs_error": ("absolute_error", "sum"),
        "mean_abs_error": ("absolute_error", "mean"),
        "mean_squared_error": ("squared_error", "mean"),
        "mean_error": ("error", "mean"),
        "mean_squared_error_term": ("squared_error", "mean"),
        "smape_ratio": ("smape_term", "mean"),
        "mase_abs_error": ("mase_abs_error", "sum"),
        "mase_scale_value": ("mase_scale_value", "sum"),
        "mase_count": ("mase_count", "sum"),
        "actual_mean": ("actual_qty", "mean"),
        "forecast_mean": ("forecast_qty", "mean"),
    }
    for quantile in QUANTILES:
        suffix = int(quantile * 100)
        aggregations[f"p{suffix}_count"] = (f"p{suffix}_count", "sum")
        aggregations[f"p{suffix}_hits"] = (f"p{suffix}_hits", "sum")
        aggregations[f"p{suffix}_pinball"] = (f"p{suffix}_pinball", "sum")
    grouped = (
        work.groupby(group_columns, sort=True, dropna=False, observed=True)
        .agg(**aggregations)
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for record in grouped.to_dict(orient="records"):
        actual_total = float(record["actual_total"])
        absolute_error = float(record["sum_abs_error"])
        mae = float(record["mean_abs_error"])
        bias = _safe_ratio(float(record["sum_error"]), actual_total)
        wape = _safe_ratio(absolute_error, actual_total)
        row: dict[str, object] = {
            "model": record["model"],
            "evaluation_level": level,
            "segment_dimension": segment_dimension,
            "segment_value": (
                f"{record['part_id']}|{record['region']}"
                if segment_dimension == "part_region"
                else str(record.get(segment_dimension, "all"))
            ),
            "n_scored": int(record["n_scored"]),
            "actual_total": actual_total,
            "forecast_total": float(record["forecast_total"]),
            "wape_ratio": wape,
            "forecast_accuracy_pct": 100.0 * (1.0 - wape)
            if np.isfinite(wape)
            else np.nan,
            "bias_ratio": bias,
            "mae": mae,
            "rmse": float(np.sqrt(record["mean_squared_error"])),
            "smape_ratio": float(record["smape_ratio"]),
            "mase": _safe_ratio(
                float(record["mase_abs_error"]),
                float(record["mase_scale_value"]),
            ),
            "tracking_signal": (
                float(record["sum_error"]) / mae
                if mae != 0
                else (0.0 if record["sum_error"] == 0 else np.nan)
            ),
            "forecast_vs_actual_variance": max(
                0.0,
                float(record["mean_squared_error_term"])
                - float(record["mean_error"]) ** 2,
            ),
            "pinball_loss_qty_units": np.nan,
        }
        pinball_sum = 0.0
        pinball_count = 0.0
        for quantile in QUANTILES:
            suffix = int(quantile * 100)
            count = float(record[f"p{suffix}_count"])
            coverage = (
                float(record[f"p{suffix}_hits"]) / count if count else np.nan
            )
            row[f"p{suffix}_coverage"] = coverage
            row[f"p{suffix}_coverage_error"] = (
                abs(coverage - quantile) if np.isfinite(coverage) else np.nan
            )
            pinball_sum += float(record[f"p{suffix}_pinball"])
            pinball_count += count
        if pinball_count:
            row["pinball_loss_qty_units"] = pinball_sum / pinball_count
        rows.append(row)
    return pd.DataFrame(rows)


def _metrics_table(forecasts: pd.DataFrame, lead_rows: pd.DataFrame) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for level, source in (("weekly", forecasts), ("lead_time", lead_rows)):
        if source.empty:
            continue
        for model in source["model"].dropna().astype(str).unique():
            model_rows = source.loc[source["model"].eq(model)]
            base_groups = ["model"]
            records.append(
                _aggregate_metric_groups(model_rows, level, "overall", base_groups)
            )
            if set(SERIES_KEYS).issubset(model_rows.columns):
                model_rows = model_rows.assign(
                    part_region=model_rows["part_id"].astype(str)
                    + "|"
                    + model_rows["region"].astype(str)
                )
                records.append(
                    _aggregate_metric_groups(
                        model_rows,
                        level,
                        "part_region",
                        ["model", "part_region", *SERIES_KEYS],
                    )
                )
            for dimension in METRIC_DIMENSIONS:
                if dimension in model_rows:
                    records.append(
                        _aggregate_metric_groups(
                            model_rows,
                            level,
                            dimension,
                            ["model", dimension],
                        )
                    )
    nonempty = [frame for frame in records if not frame.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()


def _cluster_bootstrap_ci(
    units: pd.DataFrame,
    numerator: str,
    denominator: str,
    replicates: int,
    seed: int,
    confidence: float,
) -> tuple[float, float]:
    needed = {"origin_date", "part_id", "region", numerator, denominator}
    if units.empty or needed - set(units.columns):
        return np.nan, np.nan
    work = units.copy()
    work["series_id"] = work["part_id"].astype(str) + "\x1f" + work["region"].astype(str)
    grouped = work.groupby(
        ["origin_date", "series_id"], sort=True, dropna=False, observed=True
    )[[numerator, denominator]].sum()
    if grouped.empty:
        return np.nan, np.nan
    origins = grouped.index.get_level_values(0).unique().tolist()
    series = grouped.index.get_level_values(1).unique().tolist()
    num = grouped[numerator].unstack("series_id").reindex(
        index=origins, columns=series, fill_value=0.0
    ).fillna(0.0).to_numpy(dtype=float)
    den = grouped[denominator].unstack("series_id").reindex(
        index=origins, columns=series, fill_value=0.0
    ).fillna(0.0).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    origin_draws = rng.integers(
        0, len(origins), size=(replicates, len(origins))
    )
    series_draws = rng.integers(
        0, len(series), size=(replicates, len(series))
    )
    origin_counts = np.zeros((replicates, len(origins)), dtype=np.int16)
    series_counts = np.zeros((replicates, len(series)), dtype=np.int16)
    np.add.at(
        origin_counts,
        (np.repeat(np.arange(replicates), len(origins)), origin_draws.ravel()),
        1,
    )
    np.add.at(
        series_counts,
        (np.repeat(np.arange(replicates), len(series)), series_draws.ravel()),
        1,
    )
    sampled_num = np.einsum(
        "ro,os,rs->r", origin_counts, num, series_counts, optimize=True
    )
    sampled_den = np.einsum(
        "ro,os,rs->r", origin_counts, den, series_counts, optimize=True
    )
    values = np.divide(
        sampled_num,
        sampled_den,
        out=np.full(replicates, np.nan, dtype=float),
        where=sampled_den != 0,
    )
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))


def _bootstrap_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame.loc[
        frame["actual_qty"].notna() & frame["forecast_qty"].notna()
    ].copy()
    valid["actual_qty"] = pd.to_numeric(valid["actual_qty"], errors="coerce")
    valid["forecast_qty"] = pd.to_numeric(valid["forecast_qty"], errors="coerce")
    valid["error"] = valid["forecast_qty"] - valid["actual_qty"]
    valid["absolute_error"] = valid["error"].abs()
    return valid


def _paired_wape_difference_ci(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    replicates: int,
    seed: int,
    confidence: float,
) -> tuple[float, float, float]:
    keys = ["origin_date", *SERIES_KEYS]
    left = _bootstrap_inputs(candidate)
    right = _bootstrap_inputs(baseline)
    paired = left.merge(
        right,
        on=keys,
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    if paired.empty:
        return np.nan, np.nan, np.nan
    paired["wape_delta"] = (
        paired["absolute_error_candidate"] - paired["absolute_error_baseline"]
    )
    paired["actual_denominator"] = paired["actual_qty_candidate"]
    point = _safe_ratio(
        float(paired["wape_delta"].sum()),
        float(paired["actual_denominator"].sum()),
    )
    low, high = _cluster_bootstrap_ci(
        paired,
        "wape_delta",
        "actual_denominator",
        replicates,
        seed,
        confidence,
    )
    return point, low, high


def _threshold_status(
    low: float,
    high: float,
    lower_bound: float,
    upper_bound: float,
) -> str:
    if not np.isfinite(low) or not np.isfinite(high):
        return "insufficient_data"
    if low >= lower_bound and high <= upper_bound:
        return "pass"
    if high < lower_bound or low > upper_bound:
        return "fail"
    return "inconclusive"


def _asymmetric_bias(bias: float, under_weight: float) -> float:
    if not np.isfinite(bias):
        return np.nan
    return abs(bias) * (under_weight if bias < 0 else 1.0)


def _rank_score(values: pd.Series, lower_is_better: bool = True) -> pd.Series:
    valid = values.notna()
    scores = pd.Series(0.0, index=values.index, dtype=float)
    count = int(valid.sum())
    if count == 0:
        return scores
    ranks = values.loc[valid].rank(method="average", ascending=lower_is_better)
    if count == 1:
        scores.loc[valid] = 100.0
    else:
        scores.loc[valid] = (1.0 - (ranks - 1.0) / (count - 1.0)) * 100.0
    return scores


def _ordered_ranking(
    metrics: pd.DataFrame,
    weights: dict[str, float],
    overlap_removed: bool = False,
) -> pd.DataFrame:
    ranked = metrics.copy()
    ranked["score_overall_error"] = (
        0.70 * _rank_score(ranked["lead_time_wape_ratio"])
        + 0.30 * _rank_score(ranked["weekly_wape_ratio"])
    )
    critical_wape = _rank_score(ranked["critical_a_lead_wape_ratio"])
    critical_bias = _rank_score(ranked["critical_a_asymmetric_bias_ratio"])
    ranked["score_critical_a_class_error"] = (
        critical_wape if overlap_removed else 0.50 * critical_wape + 0.50 * critical_bias
    )
    ranked["score_intermittent_lumpy"] = (
        0.50 * _rank_score(ranked["intermittent_lumpy_lead_wape_ratio"])
        + 0.50 * _rank_score(ranked["intermittent_lumpy_mase"])
    )
    ranked["score_asymmetric_absolute_bias"] = (
        0.70 * _rank_score(ranked["lead_asymmetric_bias_ratio"])
        + 0.30 * _rank_score(ranked["weekly_asymmetric_bias_ratio"])
    )
    coverage_score = _rank_score(ranked["calibration_coverage_error"])
    pinball_score = _rank_score(ranked["calibration_pinball_loss"])
    ranked["score_calibration"] = 0.50 * coverage_score + 0.50 * pinball_score
    ranked.loc[~ranked["has_probabilistic_output"], "score_calibration"] = 0.0
    ranked["score_stability_across_origins"] = _rank_score(
        ranked["origin_lead_time_wape_std"]
    )
    runtime_score = _rank_score(ranked["runtime_seconds"])
    explainability_score = _rank_score(
        ranked["explainability_tier"], lower_is_better=False
    )
    ranked["score_runtime_and_explainability"] = (
        0.50 * runtime_score + 0.50 * explainability_score
    )
    component_columns = {
        "overall_error": "score_overall_error",
        "critical_a_class_error": "score_critical_a_class_error",
        "intermittent_lumpy": "score_intermittent_lumpy",
        "asymmetric_absolute_bias": "score_asymmetric_absolute_bias",
        "calibration": "score_calibration",
        "stability_across_origins": "score_stability_across_origins",
        "runtime_and_explainability": "score_runtime_and_explainability",
    }
    ranked["weighted_score"] = sum(
        ranked[column] * weights[component]
        for component, column in component_columns.items()
    )
    ranked["weighted_score"] = ranked["weighted_score"].fillna(0.0)
    return ranked.sort_values(
        [
            "weighted_score",
            "simplicity_tier",
            "explainability_tier",
            "runtime_seconds",
            "candidate_id",
        ],
        ascending=[False, False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def _sensitivity(
    metrics: pd.DataFrame,
    weights: dict[str, float],
    points: float,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for component in weights:
        for delta in (-points, points):
            changed = weights.copy()
            changed[component] = max(0.0, changed[component] + delta)
            total = sum(changed.values())
            if total <= 0:
                raise ValueError("Sensitivity weights cannot sum to zero")
            changed = {name: value / total for name, value in changed.items()}
            ranked = _ordered_ranking(metrics, changed)
            if ranked.empty:
                continue
            records.append(
                {
                    "case": f"{component}_{int(delta * 100):+d}pp",
                    "perturbed_component": component,
                    "perturbation_points": int(delta * 100),
                    "winner": ranked.iloc[0]["candidate_id"],
                    "winner_score": ranked.iloc[0]["weighted_score"],
                }
            )
    return pd.DataFrame(records)


def _choose_sba_or_tsb(
    root: Path,
    demand: pd.DataFrame,
    scheduling: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    forecast_config = config["forecast"]
    backtest_config = forecast_config["backtest"]
    tuning_count = int(backtest_config["tuning_origin_count"])
    origins = [
        pd.Timestamp(value)
        for value in backtest_config["origins"][:tuning_count]
    ]
    lead_times = _lead_time_map(scheduling)
    summaries: list[dict[str, object]] = []
    for model in ("sba", "tsb"):
        pieces: list[pd.DataFrame] = []
        for origin in origins:
            history, targets, calendar = _fold_data(
                demand, origin, int(backtest_config["horizon_weeks"])
            )
            history_by_series = {
                key: group
                for key, group in history.groupby(
                    list(SERIES_KEYS), sort=False, observed=True
                )
            }
            predictions_list: list[dict[str, object]] = []
            for key, series_calendar in calendar.groupby(
                list(SERIES_KEYS), sort=False, observed=True
            ):
                series_history = history_by_series.get(key)
                if series_history is None:
                    series_history = history.iloc[:0]
                predictions = forecast_baseline(
                    series_history,
                    series_calendar["week_start_date"].tolist(),
                    model,
                )
                for (_, calendar_row), prediction in zip(
                    series_calendar.iterrows(), predictions
                ):
                    predictions_list.append(
                        {
                            "part_id": key[0],
                            "region": key[1],
                            "week_start_date": calendar_row["week_start_date"],
                            "forecast_qty": float(prediction),
                            "forecast_p90": np.nan,
                            "forecast_p95": np.nan,
                            "forecast_p99": np.nan,
                        }
                    )
            predictions_frame = pd.DataFrame(predictions_list)
            joined = predictions_frame.merge(
                targets[[
                    "part_id", "region", "week_start_date", "demand_qty",
                    "criticality", "part_category",
                ]],
                on=["part_id", "region", "week_start_date"],
                how="left",
                validate="one_to_one",
            ).merge(
                lead_times[["part_id", "lead_time_weeks"]],
                on="part_id",
                how="left",
                validate="many_to_one",
            )
            joined = joined.rename(columns={"demand_qty": "actual_qty"})
            joined["origin_date"] = origin
            joined["model"] = model
            joined["config_id"] = "baseline_default"
            joined["horizon_step"] = (
                (joined["week_start_date"] - origin).dt.days // 7
            )
            pieces.append(joined)
        lead_rows = _lead_time_rows(
            pd.concat(pieces, ignore_index=True),
            int(
                backtest_config["over_horizon_extension_period_weeks"]
            ),
        )
        summaries.append(
            {
                "model": model,
                "tuning_origins": len(origins),
                "lead_time_wape_ratio": _pooled_wape(lead_rows),
                "status": "success" if not lead_rows.empty else "no_scoreable_rows",
            }
        )
    result = pd.DataFrame(summaries).sort_values(
        ["lead_time_wape_ratio", "model"], kind="stable"
    ).reset_index(drop=True)
    result["selected"] = False
    if result["lead_time_wape_ratio"].notna().any():
        result.loc[result["lead_time_wape_ratio"].idxmin(), "selected"] = True
    result["selection_basis"] = "pooled lead-time WAPE on two tuning origins only"
    return result


def _candidate_metrics(
    forecasts: pd.DataFrame,
    lead_rows: pd.DataFrame,
    under_weight: float,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for model, rows in forecasts.groupby("model", sort=True, observed=True):
        lead = lead_rows.loc[lead_rows["model"].eq(model)]
        valid_weekly = _bootstrap_inputs(rows)
        valid_lead = _bootstrap_inputs(lead)
        weekly_values = _metric_values(valid_weekly)
        lead_values = _metric_values(valid_lead)
        critical_weekly = valid_weekly.loc[valid_weekly["criticality"].eq("High")]
        critical_lead = valid_lead.loc[valid_lead["criticality"].eq("High")]
        critical_a_lead = valid_lead.loc[
            valid_lead["criticality"].eq("High") | valid_lead["abc_class"].eq("A")
        ]
        intermittent = valid_lead.loc[
            valid_lead["sbc_class"].isin(["Intermittent", "Lumpy"])
        ]
        record: dict[str, object] = {
            "model": model,
            "config_id": str(rows["config_id"].iloc[0]),
            "candidate_id": f"{model}::{rows['config_id'].iloc[0]}",
            "weekly_wape_ratio": weekly_values["wape_ratio"],
            "weekly_bias_ratio": weekly_values["bias_ratio"],
            "lead_time_wape_ratio": lead_values["wape_ratio"],
            "lead_time_bias_ratio": lead_values["bias_ratio"],
            "critical_underforecast_bias_ratio": _metric_values(
                critical_lead
            )["bias_ratio"],
            "critical_a_lead_wape_ratio": _metric_values(
                critical_a_lead
            )["wape_ratio"],
            "critical_a_asymmetric_bias_ratio": _asymmetric_bias(
                float(_metric_values(critical_a_lead)["bias_ratio"]),
                under_weight,
            ),
            "intermittent_lumpy_lead_wape_ratio": _metric_values(
                intermittent
            )["wape_ratio"],
            "intermittent_lumpy_mase": _intermittent_mase(
                forecasts.loc[forecasts["model"].eq(model)]
            ),
            "lead_asymmetric_bias_ratio": _asymmetric_bias(
                float(lead_values["bias_ratio"]), under_weight
            ),
            "weekly_asymmetric_bias_ratio": _asymmetric_bias(
                float(weekly_values["bias_ratio"]), under_weight
            ),
            "calibration_coverage_error": _calibration_coverage_error(
                valid_weekly, valid_lead
            ),
            "calibration_pinball_loss": _calibration_pinball_loss(
                valid_weekly, valid_lead
            ),
            "has_probabilistic_output": bool(
                rows[["forecast_p90", "forecast_p95", "forecast_p99"]]
                .notna()
                .any()
                .any()
            ),
            "runtime_seconds": _runtime_seconds(rows),
            "n_weekly": int(len(valid_weekly)),
            "n_lead_time": int(len(valid_lead)),
            "origin_lead_time_wape_std": _origin_wape_std(lead),
        }
        simplicity, explainability = SIMPLEX_COMPLEXITY.get(str(model), (1, 1))
        record["simplicity_tier"] = simplicity
        record["explainability_tier"] = explainability
        records.append(record)
    return pd.DataFrame(records)


def _intermittent_mase(forecasts: pd.DataFrame) -> float:
    subset = forecasts.loc[
        forecasts["sbc_class"].isin(["Intermittent", "Lumpy"])
    ].copy()
    subset = _bootstrap_inputs(subset)
    if subset.empty:
        return np.nan
    subset["mase_scale"] = pd.to_numeric(subset["mase_scale"], errors="coerce")
    subset = subset.loc[subset["mase_scale"].gt(0)]
    if subset.empty:
        return np.nan
    subset["scaled_error"] = subset["absolute_error"] / subset["mase_scale"]
    subset["series_origin"] = (
        subset["part_id"].astype(str)
        + "\x1f"
        + subset["region"].astype(str)
        + "\x1f"
        + subset["origin_date"].astype(str)
    )
    values = subset.groupby("series_origin", observed=True)["scaled_error"].mean()
    return float(values.mean()) if len(values) else np.nan


def _calibration_coverage_error(weekly: pd.DataFrame, lead: pd.DataFrame) -> float:
    errors: list[tuple[float, float]] = []
    for frame, weight in ((lead, 0.70), (weekly, 0.30)):
        if frame.empty:
            continue
        values = []
        for quantile, column in QUANTILE_COLUMNS.items():
            if column not in frame or frame[column].isna().all():
                continue
            valid = frame[["actual_qty", column]].dropna()
            if valid.empty:
                continue
            values.append(
                abs(float(valid["actual_qty"].le(valid[column]).mean()) - quantile)
            )
        if values:
            errors.append((weight, float(np.mean(values))))
    total_weight = sum(weight for weight, _ in errors)
    return (
        float(sum(weight * value for weight, value in errors) / total_weight)
        if total_weight
        else np.nan
    )


def _calibration_pinball_loss(weekly: pd.DataFrame, lead: pd.DataFrame) -> float:
    losses: list[tuple[float, float]] = []
    for frame, weight in ((lead, 0.70), (weekly, 0.30)):
        values: list[float] = []
        for quantile, column in QUANTILE_COLUMNS.items():
            if column not in frame:
                continue
            valid = frame[["actual_qty", column]].dropna()
            if valid.empty:
                continue
            residual = valid["actual_qty"].to_numpy(dtype=float) - valid[column].to_numpy(dtype=float)
            values.extend(
                np.maximum(quantile * residual, (quantile - 1.0) * residual).tolist()
            )
        if values:
            losses.append((weight, float(np.mean(values))))
    total_weight = sum(weight for weight, _ in losses)
    return (
        float(sum(weight * value for weight, value in losses) / total_weight)
        if total_weight
        else np.nan
    )


def _runtime_seconds(frame: pd.DataFrame) -> float:
    if "runtime_seconds" not in frame:
        return np.nan
    return float(
        pd.to_numeric(frame[["origin_date", "runtime_seconds"]].drop_duplicates(
            "origin_date"
        )["runtime_seconds"], errors="coerce").sum()
    )


def _origin_wape_std(lead: pd.DataFrame) -> float:
    if lead.empty:
        return np.nan
    values = pd.Series(
        {
            origin: _pooled_wape(group)
            for origin, group in lead.groupby("origin_date", observed=True)
        }
    ).dropna()
    return float(values.std(ddof=0)) if len(values) else np.nan


def _selection_scope(
    metrics: pd.DataFrame,
    lead_rows: pd.DataFrame,
    selected_sba_tsb: str,
    minimum_composite_wins: int,
) -> dict[str, str]:
    scope = {
        model: "required_candidate_family"
        for model in (
            "naive",
            "seasonal_naive_52",
            "lightgbm_tweedie",
            "lightgbm_quantile",
            selected_sba_tsb,
            "historical_average",
        )
    }
    simple = lead_rows.loc[lead_rows["model"].isin(
        {"naive", "seasonal_naive_52", "lightgbm_tweedie", "lightgbm_quantile", selected_sba_tsb}
    )].copy()
    simple_origin_wape = pd.DataFrame(
        [
            {"model": model, "origin_date": origin, "wape_ratio": _pooled_wape(group)}
            for (model, origin), group in simple.groupby(
                ["model", "origin_date"], observed=True
            )
        ]
    )
    best_by_origin = simple_origin_wape.groupby("origin_date")["wape_ratio"].min()
    for composite in sorted(COMPOSITE_MODELS):
        candidate = lead_rows.loc[lead_rows["model"].eq(composite)]
        by_origin = pd.Series(
            {
                origin: _pooled_wape(group)
                for origin, group in candidate.groupby("origin_date", observed=True)
            }
        ).dropna()
        comparable = by_origin.index.intersection(best_by_origin.dropna().index)
        wins = int((by_origin.loc[comparable] < best_by_origin.loc[comparable]).sum())
        pooled = _pooled_wape(candidate)
        pooled_simple = [
            _pooled_wape(lead_rows.loc[lead_rows["model"].eq(model)])
            for model in (
                "naive",
                "seasonal_naive_52",
                "lightgbm_tweedie",
                "lightgbm_quantile",
                selected_sba_tsb,
            )
        ]
        pooled_simple = [value for value in pooled_simple if np.isfinite(value)]
        best_pooled = min(pooled_simple) if pooled_simple else np.nan
        if (
            wins >= minimum_composite_wins
            and np.isfinite(pooled)
            and np.isfinite(best_pooled)
            and pooled < best_pooled
        ):
            scope[composite] = "repeatable_validation_gain"
        else:
            scope[composite] = (
                f"excluded_no_repeatable_gain_{wins}_of_{len(comparable)}_origins"
            )
    scope["same_period_last_year"] = "excluded_duplicate_of_seasonal_naive"
    scope["sba"] = (
        "required_candidate_family"
        if selected_sba_tsb == "sba"
        else "excluded_not_best_sba_tsb_on_tuning_origins"
    )
    scope["tsb"] = (
        "required_candidate_family"
        if selected_sba_tsb == "tsb"
        else "excluded_not_best_sba_tsb_on_tuning_origins"
    )
    for model in metrics["model"].astype(str).unique():
        scope.setdefault(model, "excluded_additional_phase3_candidate")
    return scope


def _attach_screening(
    metrics: pd.DataFrame,
    forecasts: pd.DataFrame,
    lead_rows: pd.DataFrame,
    scope: dict[str, str],
    gate_config: dict[str, Any],
    bootstrap_config: dict[str, Any],
) -> pd.DataFrame:
    result = metrics.copy()
    result["selection_scope"] = result["model"].map(scope).fillna(
        "excluded_additional_phase3_candidate"
    )
    result["is_reference_baseline"] = result["model"].isin(
        {"seasonal_naive_52", "historical_average"}
    )
    by_model = {model: rows for model, rows in forecasts.groupby("model", observed=True)}
    lead_by_model = {model: rows for model, rows in lead_rows.groupby("model", observed=True)}
    seasonal = lead_by_model.get("seasonal_naive_52", pd.DataFrame())
    average = lead_by_model.get("historical_average", pd.DataFrame())
    reps = int(bootstrap_config["bootstrap_replicates"])
    confidence = float(bootstrap_config["bootstrap_confidence_level"])
    seed = int(bootstrap_config["random_seed"])
    overall_bias_limit = float(gate_config["overall_absolute_bias_max"])
    critical_bias_limit = float(gate_config["critical_underforecast_bias_min"])
    p95_target = float(gate_config["critical_p95_coverage_target"])
    p95_tolerance = float(gate_config["critical_p95_coverage_tolerance"])
    rows: list[dict[str, object]] = []
    for index, row in result.iterrows():
        record = row.to_dict()
        model = str(row["model"])
        weekly = by_model.get(model, pd.DataFrame())
        lead = lead_by_model.get(model, pd.DataFrame())
        weekly_ci = _cluster_bootstrap_ci(
            _bootstrap_inputs(weekly).assign(
                numerator=lambda data: data["absolute_error"],
                denominator=lambda data: data["actual_qty"],
            ),
            "numerator",
            "denominator",
            reps,
            seed + index,
            confidence,
        )
        lead_ci = _cluster_bootstrap_ci(
            _bootstrap_inputs(lead).assign(
                numerator=lambda data: data["absolute_error"],
                denominator=lambda data: data["actual_qty"],
            ),
            "numerator",
            "denominator",
            reps,
            seed + 100 + index,
            confidence,
        )
        bias_ci = _cluster_bootstrap_ci(
            _bootstrap_inputs(lead).assign(
                numerator=lambda data: data["error"],
                denominator=lambda data: data["actual_qty"],
            ),
            "numerator",
            "denominator",
            reps,
            seed + 200 + index,
            confidence,
        )
        critical = _bootstrap_inputs(lead).loc[
            _bootstrap_inputs(lead)["criticality"].eq("High")
        ]
        critical_ci = _cluster_bootstrap_ci(
            critical.assign(
                numerator=lambda data: data["error"],
                denominator=lambda data: data["actual_qty"],
            ),
            "numerator",
            "denominator",
            reps,
            seed + 300 + index,
            confidence,
        )
        for prefix, interval in (("weekly_wape", weekly_ci), ("lead_wape", lead_ci)):
            record[f"{prefix}_ci_low"] = interval[0]
            record[f"{prefix}_ci_high"] = interval[1]
        record["lead_bias_ci_low"], record["lead_bias_ci_high"] = bias_ci
        record["critical_underforecast_bias_ci_low"] = critical_ci[0]
        record["critical_underforecast_bias_ci_high"] = critical_ci[1]
        seasonal_delta = _paired_wape_difference_ci(
            lead, seasonal, reps, seed + 400 + index, confidence
        ) if not seasonal.empty else (np.nan, np.nan, np.nan)
        average_delta = _paired_wape_difference_ci(
            lead, average, reps, seed + 500 + index, confidence
        ) if not average.empty else (np.nan, np.nan, np.nan)
        record.update(
            {
                "lead_wape_delta_vs_seasonal_naive": seasonal_delta[0],
                "lead_wape_delta_vs_seasonal_naive_ci_low": seasonal_delta[1],
                "lead_wape_delta_vs_seasonal_naive_ci_high": seasonal_delta[2],
                "lead_wape_delta_vs_historical_average": average_delta[0],
                "lead_wape_delta_vs_historical_average_ci_low": average_delta[1],
                "lead_wape_delta_vs_historical_average_ci_high": average_delta[2],
                "beats_both_baselines_point": bool(
                    row["model"] not in {"seasonal_naive_52", "historical_average"}
                    and np.isfinite(row["lead_time_wape_ratio"])
                    and np.isfinite(_pooled_wape(seasonal))
                    and np.isfinite(_pooled_wape(average))
                    and row["lead_time_wape_ratio"] < _pooled_wape(seasonal)
                    and row["lead_time_wape_ratio"] < _pooled_wape(average)
                ),
                "overall_bias_screen": _threshold_status(
                    bias_ci[0], bias_ci[1], -overall_bias_limit, overall_bias_limit
                ),
                "critical_underforecast_screen": (
                    "not_applicable"
                    if critical.empty
                    else _threshold_status(
                        critical_ci[0], critical_ci[1], critical_bias_limit, np.inf
                    )
                ),
            }
        )
        if model == "lightgbm_quantile":
            critical_p95 = lead.loc[
                lead["criticality"].eq("High")
                & lead["forecast_p95"].notna()
            ].copy()
            critical_p95["covered"] = (
                critical_p95["actual_qty"] <= critical_p95["forecast_p95"]
            ).astype(float)
            coverage = (
                float(critical_p95["covered"].mean())
                if not critical_p95.empty
                else np.nan
            )
            coverage_ci = _cluster_bootstrap_ci(
                critical_p95.assign(
                    covered_n=lambda data: data["covered"],
                    covered_d=lambda data: 1.0,
                ),
                "covered_n",
                "covered_d",
                reps,
                seed + 600 + index,
                confidence,
            )
            record["critical_p95_coverage"] = coverage
            record["critical_p95_coverage_ci_low"] = coverage_ci[0]
            record["critical_p95_coverage_ci_high"] = coverage_ci[1]
            record["critical_p95_coverage_screen"] = _threshold_status(
                coverage_ci[0],
                coverage_ci[1],
                p95_target - p95_tolerance,
                p95_target + p95_tolerance,
            )
        else:
            record["critical_p95_coverage"] = np.nan
            record["critical_p95_coverage_ci_low"] = np.nan
            record["critical_p95_coverage_ci_high"] = np.nan
            record["critical_p95_coverage_screen"] = "not_applicable"
        baseline_enforced = bool(
            gate_config.get("enforce", {}).get("baseline_beating", False)
        )
        record["eligible_for_weighted_selection"] = bool(
            record["selection_scope"]
            in {"required_candidate_family", "repeatable_validation_gain"}
            and not record["is_reference_baseline"]
            and (
                record["beats_both_baselines_point"]
                if baseline_enforced
                else True
            )
        )
        rows.append(record)
    return pd.DataFrame(rows)


def _forecast_future_candidates(
    root: Path,
    config: dict[str, Any],
    demand: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Forecast the future split using only train-validation labels.

    The dataset places the 14 future weeks after a sealed 26-week final-test
    window. Until Phase 7, the only honest bridge is recursive prediction
    through that gap, so every row records its effective step from train end.
    """
    from src.forecasting.phase4 import _forecast_final_model

    if candidates.empty:
        return pd.DataFrame(), {"status": "blocked_no_candidates"}
    training_end = pd.Timestamp(
        config["forecast"]["split_dates"]["train_validation_end"]
    )
    future = demand.loc[
        demand["split_role"].eq("future"),
        ["part_id", "region", "week_start_date", "criticality", "part_category"],
    ].copy()
    future["week_start_date"] = pd.to_datetime(future["week_start_date"])
    future = future.drop_duplicates(
        ["part_id", "region", "week_start_date"], keep="first"
    )
    target_dates = sorted(future["week_start_date"].dropna().unique())
    if len(target_dates) != int(config["forecast"]["horizon_weeks"]):
        raise ValueError(
            f"Expected {config['forecast']['horizon_weeks']} future weeks, "
            f"found {len(target_dates)}"
        )
    output_rows: list[pd.DataFrame] = []
    models_dir = root / "artifacts" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["forecast"]["backtest"]["seed"])
    for candidate in candidates.itertuples(index=False):
        selected = pd.Series(
            {"model": candidate.model, "config_id": candidate.config_id}
        )
        predictions, model_bundle = _forecast_final_model(
            root, config, demand, selected, None, seed
        )
        predictions["week_start_date"] = pd.to_datetime(
            predictions["week_start_date"]
        )
        predictions = predictions.merge(
            future,
            on=["part_id", "region", "week_start_date"],
            how="inner",
            validate="one_to_one",
        )
        if len(predictions) != len(future):
            raise RuntimeError(
                f"{candidate.candidate_id} returned {len(predictions)} of "
                f"{len(future)} expected future rows"
            )
        predictions["forecast_origin"] = training_end.date().isoformat()
        predictions["forecast_target_step"] = (
            (predictions["week_start_date"] - pd.Timestamp(target_dates[0])).dt.days
            // 7
            + 1
        ).astype(int)
        predictions["recursive_step_from_train_end"] = (
            (predictions["week_start_date"] - training_end).dt.days // 7
        ).astype(int)
        predictions["unobserved_bridge_weeks"] = (
            predictions["recursive_step_from_train_end"]
            - predictions["forecast_target_step"]
        )
        predictions["model"] = candidate.model
        predictions["config_id"] = candidate.config_id
        predictions["candidate_id"] = candidate.candidate_id
        predictions["forecast_role"] = (
            "point_champion"
            if candidate.model == "lightgbm_tweedie"
            else "quantile_candidate"
            if candidate.model == "lightgbm_quantile"
            else "downstream_candidate"
        )
        predictions["quantile_output"] = candidate.model == "lightgbm_quantile"
        predictions["forecast_p50"] = np.where(
            predictions["quantile_output"],
            pd.to_numeric(predictions["forecast_qty"], errors="coerce"),
            np.nan,
        )
        model_bundle["forecast_role"] = predictions["forecast_role"].iloc[0]
        model_bundle["effective_future_steps"] = int(
            predictions["recursive_step_from_train_end"].max()
        )
        safe_model = str(candidate.model).replace("/", "_")
        safe_config = str(candidate.config_id).replace("/", "_")
        joblib.dump(
            model_bundle,
            models_dir / f"track_a_v41_{safe_model}_{safe_config}.joblib",
        )
        output_rows.append(predictions)
    result = _label_provisional_bridge(pd.concat(output_rows, ignore_index=True))
    quantile_rows = result.loc[result["model"].eq("lightgbm_quantile")]
    crossing_counts = {
        f"{left}_gt_{right}": int(
            (quantile_rows[left] > quantile_rows[right]).sum()
        )
        for left, right in (
            ("forecast_p50", "forecast_p90"),
            ("forecast_p90", "forecast_p95"),
            ("forecast_p95", "forecast_p99"),
        )
    }
    if any(crossing_counts.values()):
        raise ValueError(f"Future quantile forecasts cross: {crossing_counts}")
    path = root / "results" / "forecast_future_14w_v41.csv"
    result.to_csv(path, index=False)
    summary = {
        "status": "success",
        "future_rows_per_candidate": int(len(future)),
        "future_weeks": len(target_dates),
        "candidate_ids": result["candidate_id"].drop_duplicates().tolist(),
        "maximum_recursive_step": int(
            result["recursive_step_from_train_end"].max()
        ),
        "unobserved_bridge_weeks": int(result["unobserved_bridge_weeks"].max()),
        "quantile_crossings": crossing_counts,
        "path": str(path),
    }
    return result, summary


def _label_provisional_bridge(forecasts: pd.DataFrame) -> pd.DataFrame:
    labeled = forecasts.copy()
    labeled["forecast_status"] = "provisional_bridge"
    return labeled


def _write_report(
    root: Path,
    selection: pd.DataFrame,
    sensitivity: pd.DataFrame,
    sba_tsb: pd.DataFrame,
    forecasts: pd.DataFrame,
    lead_rows: pd.DataFrame,
    champion_minimum: float,
    extension_period: int,
    future_summary: dict[str, object],
) -> str:
    if selection.empty:
        champion = "No eligible weighted candidate."
        contenders: list[str] = []
        stable = False
    else:
        max_share = float(selection["sensitivity_win_share"].max())
        stable = max_share >= champion_minimum
        top = selection.loc[selection["sensitivity_win_share"].eq(max_share)].sort_values(
            ["weighted_score", "simplicity_tier", "explainability_tier", "runtime_seconds"],
            ascending=[False, False, False, True],
            kind="stable",
        )
        champion = str(top.iloc[0]["candidate_id"]) if stable else "No stable champion."
        contenders = (
            selection.head(2)["candidate_id"].astype(str).tolist()
            if stable
            else selection.loc[
                selection["sensitivity_win_count"].gt(0), "candidate_id"
            ].astype(str).tolist()
        )
        if not contenders:
            contenders = selection.head(3)["candidate_id"].astype(str).tolist()
    quantile = forecasts.loc[forecasts["model"].eq("lightgbm_quantile")]
    quantile_crossings = {
        f"{left}_gt_{right}": int(
            (quantile.loc[:, [left, right]].dropna()[left]
             > quantile.loc[:, [left, right]].dropna()[right]).sum()
        )
        for left, right in (
            ("forecast_qty", "forecast_p90"),
            ("forecast_p90", "forecast_p95"),
            ("forecast_p95", "forecast_p99"),
        )
    }
    extension_count = int(lead_rows["lead_time_extension_weeks"].gt(0).sum())
    report = f"""# Track A selection — SPEC v4.1

**Evidence window:** four held-validation origins only. Final-test and future labels were not scored.  
**Claim scope:** method validated on simulated data; results are not proven factory savings.

## Decision

- Status: {"champion supported" if stable else "contender set; downstream validation needed"}.
- Champion: {champion}.
- Candidates carried into the Track B forecast-by-scheduler check: {", ".join(contenders) if contenders else "none"}.
- Sensitivity winners: {len(sensitivity)} one-at-a-time ±10-point cases; required champion share: {champion_minimum:.0%}.
- The additional critical-bias-overlap-removed scenario is recorded separately in the selection CSV.
- Future output: {future_summary.get("path", "not produced")}. It is a recursive train-validation-only projection through the sealed test-date gap, not a directly validated 14-step forecast; the effective recursive step is {future_summary.get("maximum_recursive_step", "not measured")}, with {future_summary.get("unobserved_bridge_weeks", "not measured")} unobserved bridge weeks. Refit and evaluate against the final-test window only in Phase 7.

## Required candidate family and scope

SBA versus TSB was selected using pooled lead-time WAPE on the two configured tuning origins, before the four held origins were scored. The comparison is in `track_a_v41_sba_tsb_selection.csv`. Historical average uses the observed train-validation mean available strictly through each origin. Other Phase 3 candidates remain in the forecast artifact and selection matrix; ensembles/routers enter the weighted pool only if they beat the best simple candidate on pooled lead-time WAPE and on at least three held origins.

## Known-at-origin feature decision

The held-validation backtests use date-derived and static fields included in the fold's approved calendar whitelist. Promotions/campaigns, market events/indices, plant shutdown/working-day fields, vehicle parc/age, and substitution groups are excluded because the source does not record publication/availability dates proving they were known at each origin. No `out_*` or label-derived field is used.

## Measured results

- Held-validation forecast rows in the v4.1 artifact: {len(forecasts):,}; lead-time aggregates: {len(lead_rows):,}.
- Quantile crossing counts (`P50<=P90`, `P90<=P95`, `P95<=P99`): {quantile_crossings}.
- Lead-time rows using the labelled extension: {extension_count}. For lead times over 14 weeks, the final {extension_period} weeks are the “last period”; actual and forecast sums are extended by their respective mean times the extra lead weeks. This is a project assumption.
- Forecast metrics are stored in ratio units in `track_a_v41_metrics.csv`; percentages are display-only. Bootstrap intervals resample part-region series and origins together. Four origins and synthetic data limit inference.
- Screening enforces only beating both seasonal-naive and origin-visible historical-average baselines by point estimate. Bias and calibration screens are reported with intervals and do not reject a candidate when uncertainty crosses the threshold. Bias screens use lead-time aggregation; “critical” means High criticality, and the critical/A error segment is High criticality or ABC A.
- Weighted selection uses rank normalization within the baseline-eligible set. Missing metric scores are zero. Critical/A-class error averages lead-time WAPE and asymmetric-bias ranks; intermittent/lumpy averages lead-time WAPE and weekly MASE ranks; runtime/explainability is split equally. These are explicit scoring conventions.

## Outputs

- `artifacts/track_a_backtest_forecasts_v41.parquet`
- `results/track_a_v41_selection_matrix.csv`
- `results/track_a_v41_weight_sensitivity.csv`
- `results/track_a_v41_metrics.csv`
- `results/track_a_v41_sba_tsb_selection.csv`
- `results/forecast_future_14w_v41.csv`
- Frozen model artifacts under `artifacts/models/`

The Phase 3 artifacts and legacy Phase 4 files were not overwritten. The future output is a projection, not measured performance. Phase 5 must compare the top two forecast candidates on identical demand paths with frozen simulator assumptions; the final test remains sealed.
"""
    path = root / "results" / "phase4_track_a_v41_report.md"
    path.write_text(report, encoding="utf-8")
    return str(path)


def run_phase4_v41(root: Path = ROOT) -> dict[str, object]:
    config = _read_config(root)
    with (root / "configs" / "business.yaml").open(encoding="utf-8") as stream:
        business_config = yaml.safe_load(stream)
    forecast_config = config["forecast"]
    scoring = config["scoring"]["tracks"]["forecasting"]
    gate_config = scoring["gates"]
    bootstrap_config = gate_config
    extension_period = int(
        forecast_config["backtest"]["over_horizon_extension_period_weeks"]
    )
    forecast_path = root / FORECAST_ARTIFACT
    if not forecast_path.is_file():
        raise FileNotFoundError(f"Missing Phase 3 validation artifact: {forecast_path}")
    forecasts = pd.read_parquet(forecast_path)
    if forecasts["evaluation_split"].ne("evaluation").any():
        raise ValueError("Phase 4 v4.1 accepts held-validation rows only")
    if forecasts["origin_date"].astype(str).nunique() != 4:
        raise ValueError("Expected exactly four held-validation origins")
    if pd.to_datetime(forecasts["week_start_date"]).max() > pd.Timestamp("2026-03-23"):
        raise ValueError("Phase 4 v4.1 rows extend beyond the train-validation cutoff")
    forecasts["origin_date"] = pd.to_datetime(forecasts["origin_date"])
    forecasts["week_start_date"] = pd.to_datetime(forecasts["week_start_date"])
    quantile_mask = forecasts["model"].eq("lightgbm_quantile")
    forecasts["forecast_p50"] = np.where(
        quantile_mask,
        pd.to_numeric(forecasts["forecast_qty"], errors="coerce"),
        np.nan,
    )
    quantile_rows = forecasts.loc[quantile_mask]
    if (
        quantile_rows["forecast_p50"].gt(quantile_rows["forecast_p90"]).any()
        or quantile_rows["forecast_p90"].gt(quantile_rows["forecast_p95"]).any()
        or quantile_rows["forecast_p95"].gt(quantile_rows["forecast_p99"]).any()
    ):
        raise ValueError("Quantile forecasts cross in the Phase 3 validation artifact")

    demand = load_demand(root / "data" / "01_demand_forecasting.csv")
    scheduling = load_scheduling(root / "data" / "02_scheduling_inputs.csv")
    sba_tsb = _choose_sba_or_tsb(root, demand, scheduling, config)
    selected_sba_tsb_rows = sba_tsb.loc[sba_tsb["selected"]]
    if selected_sba_tsb_rows.empty:
        raise RuntimeError("Unable to select an SBA/TSB candidate from tuning origins")
    selected_sba_tsb = str(selected_sba_tsb_rows.iloc[0]["model"])
    forecasts, _ = _add_historical_average(forecasts, demand)
    forecasts["forecast_p50"] = np.where(
        forecasts["model"].eq("lightgbm_quantile"),
        pd.to_numeric(forecasts["forecast_qty"], errors="coerce"),
        np.nan,
    )
    quantile_rows = forecasts.loc[forecasts["model"].eq("lightgbm_quantile")]
    if (quantile_rows["forecast_p50"] > quantile_rows["forecast_p90"]).any():
        raise ValueError("Quantile P50 exceeds P90")
    lead_rows = _lead_time_rows(forecasts, extension_period)
    forecasts.to_parquet(root / OUTPUT_FORECASTS, index=False)
    metrics_table = _metrics_table(forecasts, lead_rows)
    metrics_table.to_csv(root / "results" / "track_a_v41_metrics.csv", index=False)

    under_weight = float(
        business_config["forecast"]["asymmetric_bias"]["under_weight"]
    )
    candidate_metrics = _candidate_metrics(forecasts, lead_rows, under_weight)
    sensitivity_config = scoring["sensitivity"]
    scope = _selection_scope(
        candidate_metrics,
        lead_rows,
        selected_sba_tsb,
        int(sensitivity_config["composite_minimum_origin_wins"]),
    )
    screened = _attach_screening(
        candidate_metrics,
        forecasts,
        lead_rows,
        scope,
        gate_config,
        bootstrap_config,
    )
    eligible = screened.loc[screened["eligible_for_weighted_selection"]].copy()
    weights = {
        key: float(value) for key, value in scoring["weights"].items()
    }
    points = float(sensitivity_config["perturbation_points"]) / 100.0
    minimum_share = float(sensitivity_config["champion_minimum_win_share"])
    if eligible.empty:
        ranked = eligible.copy()
        sensitivity = pd.DataFrame()
        no_overlap_winner = ""
    else:
        ranked = _ordered_ranking(eligible, weights)
        sensitivity = _sensitivity(eligible, weights, points)
        winner_counts = sensitivity["winner"].value_counts()
        ranked["sensitivity_win_count"] = ranked["candidate_id"].map(
            winner_counts
        ).fillna(0).astype(int)
        ranked["sensitivity_win_share"] = (
            ranked["sensitivity_win_count"] / max(len(sensitivity), 1)
        )
        no_overlap = _ordered_ranking(eligible, weights, overlap_removed=True)
        no_overlap_winner = (
            str(no_overlap.iloc[0]["candidate_id"]) if not no_overlap.empty else ""
        )
        ranked["overlap_removed_rank"] = ranked["candidate_id"].map(
            no_overlap.reset_index().set_index("candidate_id")["index"]
        )
        ranked["overlap_removed_winner"] = ranked["candidate_id"].eq(
            no_overlap_winner
        )
    if ranked.empty:
        future_candidates = ranked
        future_forecast, future_summary = pd.DataFrame(), {
            "status": "blocked_no_eligible_candidates"
        }
    else:
        stable = ranked["sensitivity_win_share"].max() >= minimum_share
        future_candidates = (
            ranked.head(2)
            if stable
            else ranked.loc[ranked["sensitivity_win_count"].gt(0)]
        )
        if future_candidates.empty:
            future_candidates = ranked.head(3)
        future_forecast, future_summary = _forecast_future_candidates(
            root, config, demand, future_candidates
        )
    all_results = screened.merge(
        ranked[[
            "candidate_id", "weighted_score", "sensitivity_win_count",
            "sensitivity_win_share", "overlap_removed_rank",
            "overlap_removed_winner",
        ]] if not ranked.empty else pd.DataFrame(
            columns=[
                "candidate_id", "weighted_score", "sensitivity_win_count",
                "sensitivity_win_share", "overlap_removed_rank",
                "overlap_removed_winner",
            ]
        ),
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    all_results["candidate_status"] = np.where(
        all_results["eligible_for_weighted_selection"],
        "eligible",
        all_results["selection_scope"],
    )
    all_results.to_csv(root / "results" / "track_a_v41_selection_matrix.csv", index=False)
    sensitivity.to_csv(root / "results" / "track_a_v41_weight_sensitivity.csv", index=False)
    sba_tsb.to_csv(root / "results" / "track_a_v41_sba_tsb_selection.csv", index=False)
    report = _write_report(
        root,
        ranked,
        sensitivity,
        sba_tsb,
        forecasts,
        lead_rows,
        minimum_share,
        extension_period,
        future_summary,
    )
    return {
        "status": "champion_supported"
        if not ranked.empty and ranked["sensitivity_win_share"].max() >= minimum_share
        else "contender_set" if not ranked.empty else "blocked_no_eligible_candidates",
        "selected_sba_tsb": selected_sba_tsb,
        "weighted_winner": str(ranked.iloc[0]["candidate_id"]) if not ranked.empty else None,
        "overlap_removed_winner": no_overlap_winner,
        "forecast_rows": int(len(forecasts)),
        "lead_time_rows": int(len(lead_rows)),
        "eligible_candidates": int(len(ranked)),
        "sensitivity_cases": int(len(sensitivity)),
        "future_forecast_rows": int(len(future_forecast)),
        "future_forecast": future_summary,
        "report": report,
    }


if __name__ == "__main__":
    print(run_phase4_v41())
