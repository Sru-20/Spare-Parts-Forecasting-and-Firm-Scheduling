"""Reproducible rolling-origin Track A candidate evaluation."""
from __future__ import annotations

import itertools
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.data.loader import ROOT, load_demand, load_scheduling
from src.features.segmentation import build_segment_table
from src.forecasting.baselines import BASELINE_MODELS, forecast_baseline
from src.forecasting.global_models import (
    BASE_FEATURE_DROP_GROUPS,
    UNAVAILABLE_ABLATIONS,
    build_fold_training_data,
    fit_global_models,
)
from src.forecasting.metrics import forecast_metrics, naive_scale_by_series


GLOBAL_MODEL_FAMILIES = (
    "lightgbm_point",
    "lightgbm_tweedie",
    "lightgbm_quantile",
)
ANALOGUE_FALLBACK = "analogue_category_vehicle_fallback"
SIMPLE_AVERAGE_ENSEMBLE = "simple_average_ensemble"
SBC_ROUTER = "sbc_segment_router"
SIMPLE_AVERAGE_MEMBERS = (
    "naive",
    "seasonal_naive_52",
    "moving_average_13",
    "croston",
    "sba",
    "tsb",
    *GLOBAL_MODEL_FAMILIES,
)
SEGMENT_DIMENSIONS = (
    "sbc_class",
    "abc_class",
    "criticality",
    "region",
    "part_category",
    "lifecycle_status",
)
FORECAST_COLUMNS = (
    "part_id",
    "region",
    "week_start_date",
    "forecast_qty",
    "forecast_p90",
    "forecast_p95",
    "forecast_p99",
)


def _load_config(root: Path) -> dict[str, Any]:
    config_path = root / "configs" / "project.yaml"
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict) or "forecast" not in config:
        raise ValueError(f"Invalid project configuration: {config_path}")
    return config


def _search_configurations(search: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    budget = int(search["configurations_per_objective"])
    names = (
        "num_leaves",
        "learning_rate",
        "n_estimators",
        "min_child_samples",
        "feature_fraction",
    )
    combinations = list(itertools.product(*(search[name] for name in names)))
    rng = random.Random(seed)
    selected = rng.sample(combinations, k=min(budget, len(combinations)))
    return [
        dict(zip(names, values))
        for values in selected
    ]


def _lead_time_map(scheduling: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in scheduling.itertuples(index=False):
        values = row._asdict()
        weighted_days = 0.0
        weight_sum = 0.0
        for supplier in range(1, 4):
            share = values.get(f"sup{supplier}_sob_pct")
            lead_days = values.get(f"sup{supplier}_lead_time_days")
            if pd.notna(share) and pd.notna(lead_days) and float(share) > 0:
                weighted_days += float(share) * float(lead_days)
                weight_sum += float(share)
        lead_days_value = weighted_days / weight_sum if weight_sum else np.nan
        records.append(
            {
                "part_id": values["part_id"],
                "lead_time_days": lead_days_value,
                "lead_time_weeks": (
                    max(1, math.ceil(lead_days_value / 7.0))
                    if np.isfinite(lead_days_value)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(records)


def _fold_data(
    demand: pd.DataFrame,
    origin: pd.Timestamp,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pretest = demand.loc[
        demand["split_role"].isin(["warmup", "train_val"])
        & demand["week_start_date"].le(origin)
    ].copy()
    target_dates = pd.date_range(
        origin + pd.Timedelta(weeks=1),
        periods=horizon,
        freq="W-MON",
    )
    target_rows = demand.loc[
        demand["split_role"].eq("train_val")
        & demand["week_start_date"].isin(target_dates)
    ].copy()
    if target_rows.empty or target_rows["week_start_date"].max() > pd.Timestamp("2026-03-23"):
        raise ValueError(f"Origin {origin.date()} extends outside train_val")
    known_columns = list(dict.fromkeys(
        column
        for column in (
            "part_id", "region", "week_start_date",
            "component", "vehicle_system", "part_category", "vehicle_model",
            "vehicle_type", "criticality", "make_or_buy", "is_imported",
            "unit_cost", "fulfilment_mode",
            "year", "month", "quarter_fy", "fiscal_year", "week_of_year",
            "week_of_month", "festival_flag", "festival_name",
            "festival_intensity", "weeks_to_next_festival",
            "weeks_since_last_festival", "public_holiday_flag", "monsoon_flag",
            "summer_peak_flag", "fiscal_year_end_flag",
            "dealer_order_cycle_flag",
        )
        if column in target_rows.columns
    ))
    calendar = target_rows.loc[:, known_columns].copy()
    if "demand_qty" in calendar:
        raise RuntimeError("Forecast calendar must not contain actual demand labels")
    return pretest, target_rows, calendar


def _origin_report_metadata(
    history: pd.DataFrame,
    target_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, str], float]]:
    segments = build_segment_table(history)
    segment_columns = [
        "part_id", "region", "sbc_class", "abc_class",
        "xyz_class", "criticality", "component", "vehicle_type",
    ]
    segments = segments.loc[:, segment_columns].copy()
    origin_metadata = (
        history.sort_values("week_start_date", kind="stable")
        .drop_duplicates(["part_id", "region"], keep="last")
        [["part_id", "region", "part_category", "lifecycle_status"]]
    )
    target_metadata = target_rows[
        [
            "part_id", "region", "criticality", "component", "vehicle_type",
            "part_category",
        ]
    ].drop_duplicates(["part_id", "region"])
    metadata = target_metadata.merge(
        segments.drop(
            columns=["criticality", "component", "vehicle_type"]
        ),
        on=["part_id", "region"],
        how="left",
        validate="one_to_one",
    )
    metadata = metadata.merge(
        origin_metadata[["part_id", "region", "lifecycle_status"]],
        on=["part_id", "region"],
        how="left",
        validate="one_to_one",
    )
    for column in ("sbc_class", "abc_class", "criticality", "part_category", "lifecycle_status"):
        metadata[column] = metadata[column].astype("string").fillna("Unclassified")
    return metadata, naive_scale_by_series(history)


def _baseline_forecasts(
    history: pd.DataFrame,
    target_calendar: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    forecast_rows: list[dict[str, object]] = []
    for (part_id, region), calendar in target_calendar.groupby(
        ["part_id", "region"], sort=False, observed=True
    ):
        series_history = history.loc[
            history["part_id"].eq(part_id) & history["region"].eq(region)
        ]
        predictions = forecast_baseline(
            series_history,
            calendar["week_start_date"].tolist(),
            model_name,
        )
        for (_, row), prediction in zip(calendar.iterrows(), predictions):
            forecast_rows.append(
                {
                    "part_id": part_id,
                    "region": region,
                    "week_start_date": row["week_start_date"],
                    "forecast_qty": float(prediction),
                    "forecast_p90": np.nan,
                    "forecast_p95": np.nan,
                    "forecast_p99": np.nan,
                }
            )
    return pd.DataFrame(forecast_rows)


def _analogue_fallback_forecasts(
    history: pd.DataFrame,
    target_calendar: pd.DataFrame,
    metadata: pd.DataFrame,
    model_name: str = ANALOGUE_FALLBACK,
) -> pd.DataFrame:
    """Forecast unclassified series from matching part-category/type peers."""
    unclassified = metadata.loc[
        metadata["sbc_class"].eq("Unclassified"),
        [
            "part_id", "region", "part_category", "vehicle_type",
        ],
    ]
    if unclassified.empty:
        return pd.DataFrame(columns=FORECAST_COLUMNS)
    observed = history.loc[
        history["split_role"].eq("train_val") & history["demand_qty"].notna()
    ].copy()
    if observed.empty:
        return pd.DataFrame(columns=FORECAST_COLUMNS)
    latest = history["week_start_date"].max()
    observed = observed.loc[
        observed["week_start_date"].ge(latest - pd.Timedelta(weeks=12))
    ]
    donor_rates = (
        observed.groupby(
            ["part_id", "region", "part_category", "vehicle_type"],
            observed=True,
            dropna=False,
        )
        .agg(demand_rate=("demand_qty", "mean"))
        .reset_index()
    )
    output: list[dict[str, object]] = []
    for target in unclassified.itertuples(index=False):
        target_part = str(target.part_id)
        peers = donor_rates.loc[
            donor_rates["part_category"].eq(target.part_category)
            & donor_rates["vehicle_type"].eq(target.vehicle_type)
            & donor_rates["part_id"].astype(str).ne(target_part)
        ]
        same_region = peers.loc[peers["region"].eq(target.region)]
        if not same_region.empty:
            peers = same_region
        forecast_value = (
            float(peers["demand_rate"].median()) if not peers.empty else np.nan
        )
        series_calendar = target_calendar.loc[
            target_calendar["part_id"].eq(target_part)
            & target_calendar["region"].eq(target.region)
        ]
        for _, row in series_calendar.iterrows():
            output.append(
                {
                    "part_id": row["part_id"],
                    "region": row["region"],
                    "week_start_date": row["week_start_date"],
                    "forecast_qty": forecast_value,
                    "forecast_p90": np.nan,
                    "forecast_p95": np.nan,
                    "forecast_p99": np.nan,
                }
            )
    return pd.DataFrame(output, columns=FORECAST_COLUMNS)


def _decorate_forecasts(
    forecasts: pd.DataFrame,
    target_rows: pd.DataFrame,
    metadata: pd.DataFrame,
    scales: dict[tuple[str, str], float],
    lead_times: pd.DataFrame,
    model_name: str,
    config_id: str,
    evaluation_split: str,
    origin: pd.Timestamp,
    horizon: int,
    runtime_seconds: float,
    ablation_group: str = "none",
) -> pd.DataFrame:
    actuals = target_rows.loc[
        :,
        ["part_id", "region", "week_start_date", "demand_qty"],
    ].rename(columns={"demand_qty": "actual_qty"})
    output = forecasts.merge(
        actuals,
        on=["part_id", "region", "week_start_date"],
        how="left",
        validate="one_to_one",
    )
    output = output.merge(
        metadata,
        on=["part_id", "region"],
        how="left",
        validate="many_to_one",
    )
    output = output.merge(
        lead_times,
        on="part_id",
        how="left",
        validate="many_to_one",
    )
    output["mase_scale"] = [
        scales.get((str(part_id), str(region)), np.nan)
        for part_id, region in zip(output["part_id"], output["region"])
    ]
    output["model"] = model_name
    output["candidate_role"] = (
        "fallback_only"
        if model_name == ANALOGUE_FALLBACK
        else "forecast_candidate"
    )
    output["config_id"] = config_id
    output["evaluation_split"] = evaluation_split
    output["origin_date"] = origin
    output["horizon_step"] = (
        (output["week_start_date"] - origin).dt.days // 7
    ).astype(int)
    output["runtime_seconds"] = runtime_seconds
    output["ablation_group"] = ablation_group
    if (output["horizon_step"] < 1).any() or (output["horizon_step"] > horizon).any():
        raise ValueError("Forecast rows fall outside the configured horizon")
    return output


def _lead_time_rows(forecasts: pd.DataFrame, horizon: int) -> pd.DataFrame:
    keys = [
        "model", "config_id", "evaluation_split", "origin_date",
        "part_id", "region", "candidate_role", "ablation_group",
    ]
    if forecasts.empty:
        return pd.DataFrame()
    lead_weeks = pd.to_numeric(forecasts["lead_time_weeks"], errors="coerce")
    step = pd.to_numeric(forecasts["horizon_step"], errors="coerce")
    selected = forecasts.loc[
        lead_weeks.notna()
        & lead_weeks.le(horizon)
        & step.le(lead_weeks)
        & forecasts["actual_qty"].notna()
    ].copy()
    if selected.empty:
        return pd.DataFrame()
    selected["forecast_qty"] = pd.to_numeric(
        selected["forecast_qty"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    selected["lead_time_weeks"] = pd.to_numeric(
        selected["lead_time_weeks"], errors="raise"
    ).astype(int)
    aggregations: dict[str, tuple[str, str]] = {
        "actual_qty": ("actual_qty", "sum"),
        "forecast_qty": ("forecast_qty", "sum"),
        "forecast_rows": ("forecast_qty", "count"),
        "observed_weeks_in_lead_window": ("actual_qty", "size"),
        "lead_time_days": ("lead_time_days", "first"),
        "lead_time_weeks": ("lead_time_weeks", "first"),
        "mase_scale": ("mase_scale", "first"),
        "runtime_seconds": ("runtime_seconds", "first"),
    }
    quantile_columns = {
        "forecast_p90": "forecast_p90",
        "forecast_p95": "forecast_p95",
        "forecast_p99": "forecast_p99",
    }
    for output_column, source_column in quantile_columns.items():
        if source_column in selected:
            selected[source_column] = pd.to_numeric(
                selected[source_column], errors="coerce"
            ).replace([np.inf, -np.inf], np.nan)
            aggregations[output_column] = (source_column, "sum")
            aggregations[f"{output_column}_rows"] = (source_column, "count")
    for column in SEGMENT_DIMENSIONS:
        if column in selected and column not in keys:
            aggregations[column] = (column, "first")
    output = (
        selected.groupby(
            keys,
            sort=False,
            dropna=False,
            observed=True,
        )
        .agg(**aggregations)
        .reset_index()
    )
    output = output.loc[
        output["forecast_rows"].eq(output["observed_weeks_in_lead_window"])
    ].copy()
    for output_column in quantile_columns:
        count_column = f"{output_column}_rows"
        if count_column in output:
            output.loc[
                ~output[count_column].eq(output["observed_weeks_in_lead_window"]),
                output_column,
            ] = np.nan
            output = output.drop(columns=count_column)
    return output.drop(columns="forecast_rows")


def _metric_result(
    group: pd.DataFrame,
    level: str,
    segment_dimension: str,
    segment_value: str,
    horizon_step: str,
) -> dict[str, object]:
    quantile_forecasts = {
        quantile: group[f"forecast_p{int(quantile * 100)}"].to_numpy(dtype=float)
        for quantile in (0.90, 0.95, 0.99)
        if f"forecast_p{int(quantile * 100)}" in group
        and group[f"forecast_p{int(quantile * 100)}"].notna().any()
    }
    metrics = forecast_metrics(
        group["actual_qty"].to_numpy(dtype=float),
        group["forecast_qty"].to_numpy(dtype=float),
        mase_scale=group["mase_scale"].to_numpy(dtype=float),
        quantile_forecasts=quantile_forecasts,
    )
    first = group.iloc[0]
    result = {
        "model": first["model"],
        "config_id": first["config_id"],
        "evaluation_split": first["evaluation_split"],
        "candidate_role": first["candidate_role"],
        "ablation_group": first["ablation_group"],
        "origin_date": first["origin_date"],
        "evaluation_level": level,
        "segment_dimension": segment_dimension,
        "segment_value": segment_value,
        "horizon_step": horizon_step,
        "runtime_seconds": float(
            group[["origin_date", "runtime_seconds"]]
            .drop_duplicates("origin_date")["runtime_seconds"]
            .sum()
        ),
        "n_forecast": int(np.isfinite(group["forecast_qty"].to_numpy(dtype=float)).sum()),
        "status": "success",
    }
    result.update(metrics)
    return result


def _metric_rows(forecasts: pd.DataFrame, horizon: int) -> pd.DataFrame:
    results: list[dict[str, object]] = []
    slice_dimensions = [
        column for column in SEGMENT_DIMENSIONS if column in forecasts
    ]
    for _, candidate_fold in forecasts.groupby(
        ["model", "config_id", "evaluation_split", "origin_date"],
        sort=False,
        observed=True,
    ):
        candidate_fold = candidate_fold.loc[candidate_fold["actual_qty"].notna()]
        for level, data in (
            ("weekly", candidate_fold),
            ("lead_time", _lead_time_rows(candidate_fold, horizon)),
        ):
            if data.empty:
                continue
            results.append(
                _metric_result(data, level, "overall", "all", "all")
            )
            if level == "weekly":
                for step, step_data in data.groupby("horizon_step", sort=True):
                    results.append(
                        _metric_result(
                            step_data, level, "overall", "all", str(int(step))
                        )
                    )
            for dimension in slice_dimensions:
                for value, segment_data in data.groupby(
                    dimension, dropna=False, sort=True, observed=True
                ):
                    results.append(
                        _metric_result(
                            segment_data,
                            level,
                            dimension,
                            str(value),
                            "all",
                        )
                    )

    metric_frame = pd.DataFrame(results)
    if metric_frame.empty:
        return metric_frame
    pooled = []
    pooled_keys = [
        "model", "config_id", "evaluation_split", "evaluation_level",
        "segment_dimension", "segment_value", "horizon_step",
    ]
    for _, group in forecasts.groupby(
        ["model", "config_id", "evaluation_split"],
        sort=False,
        observed=True,
    ):
        for level, data in (
            ("weekly", group.loc[group["actual_qty"].notna()]),
            ("lead_time", _lead_time_rows(group, horizon)),
        ):
            if data.empty:
                continue
            pooled.append(_metric_result(data, level, "overall", "all", "all"))
            if level == "weekly":
                for dimension in slice_dimensions:
                    for value, segment_data in data.groupby(
                        dimension, dropna=False, sort=True, observed=True
                    ):
                        pooled.append(
                            _metric_result(
                                segment_data,
                                level,
                                dimension,
                                str(value),
                                "all",
                            )
                        )
                for step, step_data in data.groupby("horizon_step", sort=True):
                    pooled.append(
                        _metric_result(
                            step_data, level, "overall", "all", str(int(step))
                        )
                    )
    if pooled:
        pooled_frame = pd.DataFrame(pooled)
        pooled_frame["origin_date"] = pd.NaT
        pooled_frame["pooled_origins"] = True
        metric_frame["pooled_origins"] = False
        metric_frame = pd.concat([metric_frame, pooled_frame], ignore_index=True)
    return metric_frame


def _pooled_model_scores(forecasts: pd.DataFrame, horizon: int) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (model, config_id), group in forecasts.groupby(
        ["model", "config_id"], sort=False, observed=True
    ):
        weekly = group.loc[group["actual_qty"].notna()]
        weekly_metrics = forecast_metrics(
            weekly["actual_qty"], weekly["forecast_qty"], weekly["mase_scale"]
        )
        lead = _lead_time_rows(group, horizon)
        lead_metrics = forecast_metrics(
            lead["actual_qty"], lead["forecast_qty"], lead["mase_scale"]
        ) if not lead.empty else {"wape_pct": np.nan}
        records.append(
            {
                "model": model,
                "config_id": config_id,
                "weekly_wape_pct": weekly_metrics["wape_pct"],
                "lead_time_wape_pct": lead_metrics["wape_pct"],
                "n_weekly": weekly_metrics["n_scored"],
                "n_lead_time": lead_metrics.get("n_scored", 0),
            }
        )
    return pd.DataFrame(records)


def _failure_row(
    model: str,
    config_id: str,
    evaluation_split: str,
    origin: pd.Timestamp,
    error: Exception,
) -> dict[str, object]:
    return {
        "model": model,
        "config_id": config_id,
        "evaluation_split": evaluation_split,
        "candidate_role": (
            "fallback_only" if model == ANALOGUE_FALLBACK else "forecast_candidate"
        ),
        "origin_date": origin,
        "evaluation_level": "overall",
        "segment_dimension": "overall",
        "segment_value": "all",
        "horizon_step": "all",
        "status": "failed",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "n_scored": 0,
        "runtime_seconds": np.nan,
    }


def _run_global_fold(
    history: pd.DataFrame,
    calendar: pd.DataFrame,
    origin: pd.Timestamp,
    family: str,
    config: dict[str, Any],
    seed: int,
    dropped_groups: tuple[str, ...] = (),
    training_design: tuple[pd.DataFrame, pd.Series] | None = None,
) -> tuple[pd.DataFrame, dict[float, object], int]:
    return fit_global_models(
        history,
        origin,
        calendar,
        config,
        family,
        seed,
        dropped_groups=dropped_groups,
        training_design=training_design,
    )


def _build_candidate_report(
    root: Path,
    metrics: pd.DataFrame,
    tuning_log: pd.DataFrame,
    ablations: pd.DataFrame,
    forecasts: pd.DataFrame,
    origins: list[pd.Timestamp],
    tuning_origins: list[pd.Timestamp],
    evaluation_origins: list[pd.Timestamp],
    lead_times: pd.DataFrame,
    missing_lead_parts: list[str],
    over_horizon_parts: list[str],
) -> None:
    successful = metrics.loc[
        metrics["status"].eq("success")
        & metrics["candidate_role"].eq("forecast_candidate")
        & metrics["evaluation_split"].eq("evaluation")
        & metrics["pooled_origins"].fillna(False)
        & metrics["evaluation_level"].eq("lead_time")
        & metrics["segment_dimension"].eq("overall")
    ].sort_values("wape_pct", na_position="last")
    best = successful.head(5)
    tuning_succeeded = int(tuning_log["status"].eq("success").sum()) if not tuning_log.empty else 0
    tuning_failed = int(tuning_log["status"].eq("failed").sum()) if not tuning_log.empty else 0
    analogue_results = metrics.loc[
        metrics["candidate_role"].eq("fallback_only")
        & metrics["evaluation_split"].eq("evaluation")
        & metrics["pooled_origins"].fillna(False)
        & metrics["evaluation_level"].eq("weekly")
        & metrics["segment_dimension"].eq("sbc_class")
        & metrics["segment_value"].eq("Unclassified")
    ]
    forecast_coverage = (
        forecasts.groupby(["model", "candidate_role"], observed=True)
        .agg(
            forecast_rows=("forecast_qty", "size"),
            finite_forecasts=(
                "forecast_qty",
                lambda values: int(
                    np.isfinite(pd.to_numeric(values, errors="coerce")).sum()
                ),
            ),
        )
        .reset_index()
    )
    forecast_coverage["unavailable_forecasts"] = (
        forecast_coverage["forecast_rows"]
        - forecast_coverage["finite_forecasts"]
    )
    forecast_coverage["coverage_pct"] = (
        100.0
        * forecast_coverage["finite_forecasts"]
        / forecast_coverage["forecast_rows"]
    )
    report = f"""# Phase 3 Track A — rolling-origin baselines and Tier 1 models

## Evaluation boundary and design

- Dataset kind: synthetic. Findings mean **method validated on simulated data**, not proven factory savings.
- Backtest origins: {", ".join(str(origin.date()) for origin in origins)}; fixed horizon: 14 weeks.
- Tuning origins: {", ".join(str(origin.date()) for origin in tuning_origins)}. Held validation origins: {", ".join(str(origin.date()) for origin in evaluation_origins)}.
- Tuning uses three seed-42 random-search configurations per LightGBM objective on the same origins and search space. Only the selected configuration per objective is run on the four held validation origins. Baselines use those same held validation origins.
- Fit targets are observed `train_val` demand strictly before/at each origin. Warm-up initializes lag and seasonal-history values, not target-fitting rows. Every scored date is train-validation; final-test/future labels were not evaluated.
- Lead time is the SOB-weighted supplier mean in D2 divided by seven and rounded up (minimum one week). It is available for {int(lead_times["lead_time_days"].notna().sum()):,} of {len(lead_times) + len(missing_lead_parts):,} demand parts. Parts without a D2 lead time or with lead time longer than the 14-week horizon are omitted from lead-time metrics: {len(missing_lead_parts)} missing; {len(over_horizon_parts)} over horizon. They remain in weekly metrics.
- Lead-time actual and forecast totals use the same observed weeks in each part-region's first lead-time window. Missing actual weeks are excluded from both sides; denominators are reported.
- `same_period_last_year` and 52-week seasonal naïve are retained as separate requested candidate labels; they use the same 52-week mapping, so their forecasts are mathematically identical.

## Candidate results

Pooled held-validation candidates sorted by lead-time WAPE (not a champion decision; weighted selection and sensitivity are Phase 4):

{_markdown_table(best) if not best.empty else "No successful held-validation candidate results."}

Candidates/configurations with documented fitting or execution failures are retained in `results/track_a_candidates.csv`. Tuning configurations: {tuning_succeeded} successful and {tuning_failed} failed.

## Forecast coverage

Unavailable predictions remain blank and are excluded from scoring; they are never replaced by zero. Baselines leave 190 target rows blank where no observed train-validation demand is available at the origin; the separate category/type fallback supplied finite forecasts for those same rows. The fallback-only diagnostic is not a full-candidate comparison:

{_markdown_table(forecast_coverage)}

## Composite candidates

`{SIMPLE_AVERAGE_ENSEMBLE}` is the unweighted mean of nine distinct baseline and LightGBM forecasts; the duplicate same-period-last-year mapping is omitted to avoid double weighting. `{SBC_ROUTER}` uses the fixed project-configured rule: LightGBM Tweedie for Smooth/Erratic, SBA for Intermittent/Lumpy, and the category/type analogue fallback for Unclassified. Segment labels are recomputed from origin-visible history. These are measured candidates, not selected winners; the equal-weight ensemble has no probabilistic output.

## Unclassified-series analogue fallback

The separate `{ANALOGUE_FALLBACK}` fallback uses only same-part-category, same-vehicle-type donor parts visible at the origin; same-region donors are preferred when available. Each donor's rate is the mean observed train-validation demand over the trailing 13 calendar weeks, and the fallback is the median donor rate. The target series is excluded as its own donor. This broader group was selected because the supplied catalogue has unique component/vehicle-type combinations per part, leaving the original exact group with no alternate-part donors. This is a fallback-only diagnostic, not a full-coverage candidate or a Track A winner:

{_markdown_table(analogue_results) if not analogue_results.empty else "No unclassified-series analogue observations were scoreable."}

## Ablations

Seasonality and festival ablations were evaluated by removing those feature groups from the best held-validation ML family. Promotions/campaigns, market, and shutdown ablations are explicitly blocked because the source does not prove which future plans were published by each backtest origin:

{_markdown_table(ablations) if not ablations.empty else "No ablation outcomes."}

## Metric interpretation

Metrics follow `SPEC.md`. Weekly and lead-time levels, origin, segment and horizon-step records are in `results/track_a_candidates.csv`. Missing-target rows are excluded, genuine zero demand is retained, and undefined zero-total WAPE/bias remain blank. MASE uses adjacent observed train-validation one-step naïve MAE per series at that origin. The sMAPE zero/zero contribution is zero. Point-only models have no quantile calibration output.

## Limitations

These are synthetic-data results from six rolling origins; tuning uses two early train-validation origins and held validation uses four later origins. Dependence remains across parts and origins, and there is no real operational cost or external demand evidence. Phase 4 will apply the frozen weighted selection, sensitivity, paired comparisons, and champion freeze. The final test remains sealed for Phase 7.

## Reproduction

Run `python -m src.forecasting.backtest` from the repository root with dependencies in `requirements.txt` installed. Seed: 42.
"""
    (root / "results" / "phase3_track_a_report.md").write_text(
        report, encoding="utf-8"
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without an optional tabulate dependency."""
    columns = [str(column) for column in frame.columns]
    values = frame.fillna("").astype(str).to_numpy().tolist()
    widths = [
        max([len(column), *(len(row[index]) for row in values)])
        for index, column in enumerate(columns)
    ]
    header = "| " + " | ".join(
        column.ljust(widths[index]) for index, column in enumerate(columns)
    ) + " |"
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [
        "| " + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)
        ) + " |"
        for row in values
    ]
    return "\n".join([header, separator, *body])


def _forecast_artifact_frame(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Use ISO origin strings to keep the Parquet timestamp schema portable."""
    output = forecasts.copy()
    output["origin_date"] = pd.to_datetime(output["origin_date"]).dt.strftime(
        "%Y-%m-%d"
    )
    return output


def _append_composite_candidates(
    forecasts: pd.DataFrame,
    sbc_router: dict[str, str],
) -> pd.DataFrame:
    """Append the fixed equal-weight ensemble and origin-known SBC router."""
    if forecasts.empty:
        return forecasts.copy()
    frame = forecasts.copy()
    frame["origin_date"] = pd.to_datetime(frame["origin_date"])
    keys = [
        "part_id", "region", "week_start_date", "origin_date",
        "evaluation_split", "horizon_step",
    ]
    context_columns = [
        *keys,
        *[
            column
            for column in (
                "actual_qty", *SEGMENT_DIMENSIONS, "lead_time_days",
                "lead_time_weeks", "mase_scale",
            )
            if column in frame and column not in keys
        ],
    ]
    template = frame.drop_duplicates(keys, keep="first")[context_columns].copy()
    appended: list[pd.DataFrame] = []

    ensemble_sources = frame.loc[frame["model"].isin(SIMPLE_AVERAGE_MEMBERS)]
    if not ensemble_sources.empty:
        started = time.perf_counter()
        ensemble_values = (
            ensemble_sources.groupby(keys, sort=False, observed=True)["forecast_qty"]
            .mean()
            .rename("forecast_qty")
            .reset_index()
        )
        ensemble = template.merge(
            ensemble_values, on=keys, how="left", validate="one_to_one"
        )
        ensemble["forecast_p90"] = np.nan
        ensemble["forecast_p95"] = np.nan
        ensemble["forecast_p99"] = np.nan
        ensemble["model"] = SIMPLE_AVERAGE_ENSEMBLE
        ensemble["candidate_role"] = "forecast_candidate"
        ensemble["config_id"] = "equal_weight_nine_model_mean"
        ensemble["ablation_group"] = "none"
        ensemble["runtime_seconds"] = time.perf_counter() - started
        appended.append(ensemble)

    if sbc_router:
        started = time.perf_counter()
        router_template = template.copy()
        classes = router_template["sbc_class"].astype("string").fillna("Unclassified")
        router_template["source_model"] = classes.map(sbc_router)
        if router_template["source_model"].isna().any():
            missing_classes = sorted(
                classes.loc[router_template["source_model"].isna()].unique().tolist()
            )
            raise ValueError(f"SBC router has no source model for classes: {missing_classes}")
        source_columns = [
            *keys, "model", "forecast_qty",
            "forecast_p90", "forecast_p95", "forecast_p99",
        ]
        source_predictions = frame.loc[
            frame["model"].isin(set(sbc_router.values())), source_columns
        ].rename(columns={"model": "source_model"})
        router = router_template.merge(
            source_predictions,
            on=[*keys, "source_model"],
            how="left",
            validate="one_to_one",
        )
        router["model"] = SBC_ROUTER
        router["candidate_role"] = "forecast_candidate"
        router["config_id"] = "fixed_sbc_model_map"
        router["ablation_group"] = "none"
        router["runtime_seconds"] = time.perf_counter() - started
        appended.append(router.drop(columns="source_model"))

    return pd.concat([frame, *appended], ignore_index=True, sort=False)


def _write_charts(root: Path, forecasts: pd.DataFrame, metrics: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    evaluation = forecasts.loc[
        forecasts["evaluation_split"].eq("evaluation")
        & forecasts["actual_qty"].notna()
    ]
    if evaluation.empty:
        return
    pooled_scores = _pooled_model_scores(evaluation, horizon=14)
    top_ml = pooled_scores.loc[
        pooled_scores["model"].isin(GLOBAL_MODEL_FAMILIES)
    ].sort_values("lead_time_wape_pct", na_position="last")
    top_baseline = pooled_scores.loc[
        pooled_scores["model"].isin(BASELINE_MODELS)
    ].sort_values("lead_time_wape_pct", na_position="last")
    selected = []
    if not top_ml.empty:
        selected.append((top_ml.iloc[0]["model"], top_ml.iloc[0]["config_id"]))
    if not top_baseline.empty:
        selected.append((top_baseline.iloc[0]["model"], top_baseline.iloc[0]["config_id"]))
    for model in (SIMPLE_AVERAGE_ENSEMBLE, SBC_ROUTER):
        composite = pooled_scores.loc[pooled_scores["model"].eq(model)]
        if not composite.empty:
            selected.append((model, composite.iloc[0]["config_id"]))
    chosen = evaluation.loc[
        evaluation.apply(
            lambda row: (row["model"], row["config_id"]) in selected, axis=1
        )
    ]
    aggregate = (
        chosen.groupby(["model", "week_start_date"], observed=True)
        .agg(
            actual=("actual_qty", "sum"),
            forecast=("forecast_qty", "sum"),
        )
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    for model, group in aggregate.groupby("model", observed=True):
        ax.plot(
            group["week_start_date"],
            group["actual"],
            label=f"{model} actual",
            color="black",
            alpha=0.35,
        )
        ax.plot(
            group["week_start_date"],
            group["forecast"],
            label=f"{model} forecast total",
            alpha=0.75,
        )
    ax.set_title("Held-validation weekly demand and candidate forecasts")
    ax.set_ylabel("Demand units")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "results" / "track_a_forecast_vs_actual.png", dpi=150)
    plt.close(fig)

    segment_scores = metrics.loc[
        metrics["evaluation_split"].eq("evaluation")
        & metrics["pooled_origins"].fillna(False)
        & metrics["evaluation_level"].eq("weekly")
        & metrics["segment_dimension"].eq("sbc_class")
        & metrics["model"].isin([model for model, _ in selected])
    ]
    if not segment_scores.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        for model, group in segment_scores.groupby("model", observed=True):
            ax.plot(
                group["segment_value"],
                group["wape_pct"],
                marker="o",
                label=model,
            )
        ax.set_title("Held-validation weekly WAPE by origin-derived SBC class")
        ax.set_ylabel("WAPE (%)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(root / "results" / "track_a_segment_accuracy.png", dpi=150)
        plt.close(fig)

    horizon_scores = metrics.loc[
        metrics["evaluation_split"].eq("evaluation")
        & metrics["pooled_origins"].fillna(False)
        & metrics["evaluation_level"].eq("weekly")
        & metrics["segment_dimension"].eq("overall")
        & metrics["horizon_step"].ne("all")
        & metrics["model"].isin([model for model, _ in selected])
    ]
    if not horizon_scores.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        for model, group in horizon_scores.groupby("model", observed=True):
            group = group.assign(
                step=pd.to_numeric(group["horizon_step"], errors="raise")
            ).sort_values("step")
            ax.plot(group["step"], group["bias_pct"], marker="o", label=model)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Held-validation bias by horizon step")
        ax.set_xlabel("Forecast week")
        ax.set_ylabel("Bias (%)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(root / "results" / "track_a_bias_by_horizon.png", dpi=150)
        plt.close(fig)


def run_phase3_backtest(root: Path = ROOT) -> dict[str, object]:
    """Run the frozen Phase 3 train-validation rolling-origin backtest."""
    config = _load_config(root)
    forecast_config = config["forecast"]
    backtest_config = forecast_config["backtest"]
    horizon = int(backtest_config["horizon_weeks"])
    seed = int(backtest_config["seed"])
    origins = [pd.Timestamp(value) for value in backtest_config["origins"]]
    tuning_count = int(backtest_config["tuning_origin_count"])
    if len(origins) <= tuning_count:
        raise ValueError("At least one held validation origin is required")
    if not backtest_config["non_overlapping_horizons"]:
        raise ValueError("Phase 3 requires fixed non-overlapping horizons")

    demand = load_demand(root / "data" / "01_demand_forecasting.csv")
    demand = demand.loc[
        demand["split_role"].isin(["warmup", "train_val"])
        & demand["week_start_date"].le(pd.Timestamp("2026-03-23"))
    ].copy()
    if demand["split_role"].eq("final_test").any() or demand["split_role"].eq("future").any():
        raise RuntimeError("Phase 3 data boundary unexpectedly includes test/future rows")
    scheduling = load_scheduling(root / "data" / "02_scheduling_inputs.csv")
    lead_times = _lead_time_map(scheduling)
    over_horizon = lead_times.loc[
        lead_times["lead_time_weeks"].gt(horizon), "part_id"
    ].astype(str).tolist()
    mapped_parts = set(
        lead_times.loc[
            lead_times["lead_time_days"].notna(), "part_id"
        ].dropna().astype(str)
    )
    missing_lead_parts = sorted(
        set(demand["part_id"].dropna().astype(str)) - mapped_parts
    )
    tuning_origins = origins[:tuning_count]
    evaluation_origins = origins[tuning_count:]

    search_config = backtest_config["lightgbm_search"]
    configurations = _search_configurations(search_config, seed)
    tuning_rows: list[dict[str, object]] = []
    tuning_forecasts: list[pd.DataFrame] = []
    candidate_failures: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []

    for origin in tuning_origins:
        history, target_rows, calendar = _fold_data(demand, origin, horizon)
        metadata, scales = _origin_report_metadata(history, target_rows)
        training_design = build_fold_training_data(history, origin)
        for family in GLOBAL_MODEL_FAMILIES:
            for config_index, parameters in enumerate(configurations, start=1):
                config_id = f"random_{config_index}"
                started = time.perf_counter()
                try:
                    predictions, models, training_rows = _run_global_fold(
                        history, calendar, origin, family, parameters, seed,
                        training_design=training_design,
                    )
                    runtime = time.perf_counter() - started
                    decorated = _decorate_forecasts(
                        predictions,
                        target_rows,
                        metadata,
                        scales,
                        lead_times,
                        family,
                        config_id,
                        "tuning",
                        origin,
                        horizon,
                        runtime,
                    )
                    tuning_forecasts.append(decorated)
                    for quantile, model in models.items():
                        for feature_name, importance in zip(
                            model.feature_name_, model.feature_importances_
                        ):
                            importance_rows.append(
                                {
                                    "model": family,
                                    "config_id": config_id,
                                    "evaluation_split": "tuning",
                                    "origin_date": origin,
                                    "quantile": quantile,
                                    "feature": feature_name,
                                    "importance": int(importance),
                                }
                            )
                    tuning_rows.append(
                        {
                            "model": family,
                            "config_id": config_id,
                            "origin_date": origin,
                            "status": "success",
                            "parameters": json.dumps(parameters, sort_keys=True),
                            "training_rows": training_rows,
                            "runtime_seconds": runtime,
                        }
                    )
                except (ImportError, ValueError, RuntimeError) as error:
                    runtime = time.perf_counter() - started
                    tuning_rows.append(
                        {
                            "model": family,
                            "config_id": config_id,
                            "origin_date": origin,
                            "status": "failed",
                            "parameters": json.dumps(parameters, sort_keys=True),
                            "training_rows": 0,
                            "runtime_seconds": runtime,
                            "error_type": type(error).__name__,
                            "error_message": str(error),
                        }
                    )
                    candidate_failures.append(
                        _failure_row(family, config_id, "tuning", origin, error)
                    )

    tuning_forecast_frame = (
        pd.concat(tuning_forecasts, ignore_index=True)
        if tuning_forecasts else pd.DataFrame()
    )
    tuning_log = pd.DataFrame(tuning_rows)
    selected_config: dict[str, dict[str, Any]] = {}
    for family in GLOBAL_MODEL_FAMILIES:
        family_rows = tuning_log.loc[
            tuning_log["model"].eq(family) & tuning_log["status"].eq("success")
        ]
        if family_rows.empty:
            continue
        scores = _pooled_model_scores(
            tuning_forecast_frame.loc[tuning_forecast_frame["model"].eq(family)],
            horizon,
        )
        scores = scores.sort_values(
            ["lead_time_wape_pct", "weekly_wape_pct", "config_id"],
            na_position="last",
            kind="stable",
        )
        selected_id = str(scores.iloc[0]["config_id"])
        selected_config[family] = configurations[int(selected_id.rsplit("_", 1)[1]) - 1]
        tuning_log.loc[
            tuning_log["model"].eq(family)
            & tuning_log["config_id"].eq(selected_id),
            "selected_for_held_validation",
        ] = True
        for _, score in scores.iterrows():
            tuning_log.loc[
                tuning_log["model"].eq(family)
                & tuning_log["config_id"].eq(score["config_id"]),
                ["tuning_weekly_wape_pct", "tuning_lead_time_wape_pct"],
            ] = [
                score["weekly_wape_pct"],
                score["lead_time_wape_pct"],
            ]

    evaluation_forecasts: list[pd.DataFrame] = []
    for origin in evaluation_origins:
        history, target_rows, calendar = _fold_data(demand, origin, horizon)
        metadata, scales = _origin_report_metadata(history, target_rows)
        training_design = build_fold_training_data(history, origin)
        for baseline in BASELINE_MODELS:
            started = time.perf_counter()
            try:
                predictions = _baseline_forecasts(history, calendar, baseline)
                runtime = time.perf_counter() - started
                evaluation_forecasts.append(
                    _decorate_forecasts(
                        predictions, target_rows, metadata, scales, lead_times,
                        baseline, "baseline_default", "evaluation", origin,
                        horizon, runtime,
                    )
                )
            except (ValueError, RuntimeError) as error:
                candidate_failures.append(
                    _failure_row(baseline, "baseline_default", "evaluation", origin, error)
                )
        try:
            analogue_predictions = _analogue_fallback_forecasts(
                history, calendar, metadata
            )
            if not analogue_predictions.empty:
                evaluation_forecasts.append(
                    _decorate_forecasts(
                        analogue_predictions,
                        target_rows,
                        metadata,
                        scales,
                        lead_times,
                        ANALOGUE_FALLBACK,
                        "fallback_median_rate",
                        "evaluation",
                        origin,
                        horizon,
                        0.0,
                    )
                )
        except (ValueError, RuntimeError) as error:
            candidate_failures.append(
                _failure_row(
                    ANALOGUE_FALLBACK,
                    "fallback_median_rate",
                    "evaluation",
                    origin,
                    error,
                )
            )
        for family, parameters in selected_config.items():
            started = time.perf_counter()
            try:
                predictions, models, training_rows = _run_global_fold(
                    history, calendar, origin, family, parameters, seed,
                    training_design=training_design,
                )
                runtime = time.perf_counter() - started
                evaluation_forecasts.append(
                    _decorate_forecasts(
                        predictions,
                        target_rows,
                        metadata,
                        scales,
                        lead_times,
                        family,
                        f"random_{configurations.index(parameters) + 1}",
                        "evaluation",
                        origin,
                        horizon,
                        runtime,
                    )
                )
                for quantile, model in models.items():
                    for feature_name, importance in zip(
                        model.feature_name_, model.feature_importances_
                    ):
                        importance_rows.append(
                            {
                                "model": family,
                                "config_id": f"random_{configurations.index(parameters) + 1}",
                                "evaluation_split": "evaluation",
                                "origin_date": origin,
                                "quantile": quantile,
                                "feature": feature_name,
                                "importance": int(importance),
                            }
                        )
            except (ImportError, ValueError, RuntimeError) as error:
                candidate_failures.append(
                    _failure_row(
                        family,
                        f"random_{configurations.index(parameters) + 1}",
                        "evaluation",
                        origin,
                        error,
                    )
                )

    evaluation_forecast_frame = (
        pd.concat(evaluation_forecasts, ignore_index=True)
        if evaluation_forecasts else pd.DataFrame()
    )
    evaluation_forecast_frame = _append_composite_candidates(
        evaluation_forecast_frame,
        config["sbc_router"],
    )
    if "selected_for_held_validation" not in tuning_log:
        tuning_log["selected_for_held_validation"] = False
    else:
        tuning_log["selected_for_held_validation"] = (
            tuning_log["selected_for_held_validation"]
            .astype("boolean")
            .fillna(False)
            .astype(bool)
        )
    if not tuning_forecast_frame.empty:
        tuning_metrics = _metric_rows(tuning_forecast_frame, horizon)
    else:
        tuning_metrics = pd.DataFrame()
    if not evaluation_forecast_frame.empty:
        evaluation_metrics = _metric_rows(evaluation_forecast_frame, horizon)
    else:
        evaluation_metrics = pd.DataFrame()
    metrics = pd.concat(
        [tuning_metrics, evaluation_metrics], ignore_index=True
    )

    best_ml = None
    if not evaluation_forecast_frame.empty:
        ml_scores = _pooled_model_scores(
            evaluation_forecast_frame.loc[
                evaluation_forecast_frame["model"].isin(GLOBAL_MODEL_FAMILIES)
            ],
            horizon,
        ).sort_values(
            ["lead_time_wape_pct", "weekly_wape_pct", "model"],
            na_position="last",
            kind="stable",
        )
        if not ml_scores.empty:
            best_ml = ml_scores.iloc[0]

    ablation_rows: list[dict[str, object]] = []
    if best_ml is not None:
        best_family = str(best_ml["model"])
        best_config_id = str(best_ml["config_id"])
        best_parameters = selected_config[best_family]
        ablation_forecasts: list[pd.DataFrame] = []
        for origin in evaluation_origins:
            history, target_rows, calendar = _fold_data(demand, origin, horizon)
            metadata, scales = _origin_report_metadata(history, target_rows)
            training_design = build_fold_training_data(history, origin)
            for ablation_group in BASE_FEATURE_DROP_GROUPS:
                started = time.perf_counter()
                try:
                    predictions, _, _ = _run_global_fold(
                        history,
                        calendar,
                        origin,
                        best_family,
                        best_parameters,
                        seed,
                        dropped_groups=(ablation_group,),
                        training_design=training_design,
                    )
                    decorated = _decorate_forecasts(
                        predictions, target_rows, metadata, scales, lead_times,
                        best_family, best_config_id, "evaluation", origin, horizon,
                        time.perf_counter() - started,
                        ablation_group=ablation_group,
                    )
                    ablation_forecasts.append(decorated)
                except (ImportError, ValueError, RuntimeError) as error:
                    ablation_rows.append(
                        {
                            "best_ml_model": best_family,
                            "config_id": best_config_id,
                            "ablation_group": ablation_group,
                            "evaluation_level": "weekly",
                            "status": "failed",
                            "error_type": type(error).__name__,
                            "error_message": str(error),
                        }
                    )
        if ablation_forecasts:
            ablation_frame = pd.concat(ablation_forecasts, ignore_index=True)
            for data_level, data in (
                ("weekly", ablation_frame.loc[ablation_frame["actual_qty"].notna()]),
                ("lead_time", _lead_time_rows(ablation_frame, horizon)),
            ):
                for group_name, group in data.groupby(
                    "ablation_group", sort=False, observed=True
                ):
                    result = _metric_result(group, data_level, "overall", "all", "all")
                    result.update(
                        {
                            "best_ml_model": best_family,
                            "ablation_group": group_name,
                            "status": "success",
                            "origin_date": pd.NaT,
                            "pooled_origins": True,
                        }
                    )
                    ablation_rows.append(result)
        for group_name, reason in UNAVAILABLE_ABLATIONS.items():
            for data_level in ("weekly", "lead_time"):
                ablation_rows.append(
                    {
                        "best_ml_model": best_family,
                        "config_id": best_config_id,
                        "ablation_group": group_name,
                        "evaluation_level": data_level,
                        "status": "not_run_origin_availability",
                        "reason": reason,
                    }
                )
    ablations = pd.DataFrame(ablation_rows)
    if best_ml is None:
        for group_name in (*BASE_FEATURE_DROP_GROUPS, *UNAVAILABLE_ABLATIONS):
            ablation_rows.append(
                {
                    "best_ml_model": pd.NA,
                    "ablation_group": group_name,
                    "evaluation_level": "weekly",
                    "status": "not_run_no_successful_ml_candidate",
                    "reason": "No global ML model completed held validation.",
                }
            )
        ablations = pd.DataFrame(ablation_rows)

    if candidate_failures:
        failure_frame = pd.DataFrame(candidate_failures)
        metrics = pd.concat([metrics, failure_frame], ignore_index=True, sort=False)
    if not tuning_log.empty:
        metrics = metrics.merge(
            tuning_log[
                ["model", "config_id", "parameters", "selected_for_held_validation"]
            ].drop_duplicates(["model", "config_id"]),
            on=["model", "config_id"],
            how="left",
        )

    results_dir = root / "results"
    artifacts_dir = root / "artifacts"
    results_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(results_dir / "track_a_candidates.csv", index=False)
    tuning_log.to_csv(results_dir / "track_a_tuning_log.csv", index=False)
    ablations.to_csv(results_dir / "track_a_ablations.csv", index=False)
    if not evaluation_forecast_frame.empty:
        _forecast_artifact_frame(evaluation_forecast_frame).to_parquet(
            artifacts_dir / "track_a_backtest_forecasts.parquet",
            index=False,
            engine="fastparquet",
        )
    if importance_rows:
        importance = pd.DataFrame(importance_rows)
        eval_importance = importance.loc[
            importance["evaluation_split"].eq("evaluation")
        ]
        feature_importance = (
            eval_importance.groupby(["model", "feature"], observed=True)
            .agg(
                mean_importance=("importance", "mean"),
                folds=("origin_date", "nunique"),
            )
            .reset_index()
            .sort_values(["model", "mean_importance"], ascending=[True, False])
        )
        feature_importance.to_csv(
            results_dir / "track_a_feature_importance.csv", index=False
        )
    else:
        pd.DataFrame(
            columns=["model", "feature", "mean_importance", "folds"]
        ).to_csv(results_dir / "track_a_feature_importance.csv", index=False)
    if not evaluation_forecast_frame.empty:
        _write_charts(root, evaluation_forecast_frame, metrics)

    _build_candidate_report(
        root,
        metrics,
        tuning_log,
        ablations,
        evaluation_forecast_frame,
        origins,
        tuning_origins,
        evaluation_origins,
        lead_times,
        missing_lead_parts,
        over_horizon,
    )
    successful_models = (
        sorted(evaluation_forecast_frame["model"].unique().tolist())
        if not evaluation_forecast_frame.empty else []
    )
    return {
        "origins": [str(value.date()) for value in origins],
        "tuning_origins": [str(value.date()) for value in tuning_origins],
        "evaluation_origins": [str(value.date()) for value in evaluation_origins],
        "horizon_weeks": horizon,
        "successful_evaluation_models": successful_models,
        "tuning_configurations": len(tuning_rows),
        "tuning_failures": int(tuning_log["status"].eq("failed").sum())
        if not tuning_log.empty else 0,
        "candidate_failures": len(candidate_failures),
        "lead_time_parts": int(lead_times["lead_time_days"].notna().sum()),
        "parts_over_horizon": len(over_horizon),
        "forecast_rows": len(evaluation_forecast_frame),
        "candidate_metric_rows": len(metrics),
        "best_ml_model": str(best_ml["model"]) if best_ml is not None else None,
    }


if __name__ == "__main__":
    for name, result in run_phase3_backtest().items():
        print(f"{name}: {result}")
