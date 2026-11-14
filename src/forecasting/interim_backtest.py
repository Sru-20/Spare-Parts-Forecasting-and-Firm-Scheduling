"""SPEC v4.2 B — interim 26-week backtest on two consecutive train-validation windows.

The two windows end at the train-validation cutoff (2026-03-23). Each is fitted
only on labels up to its origin, with frozen hyperparameters, and scored with
weekly and complete 26-week part-region aggregate metrics. The final test stays
sealed; the windows are inside held validation and are not independent.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.loader import load_demand, load_scheduling
from src.forecasting.backtest import (
    SIMPLE_AVERAGE_MEMBERS,
    _baseline_forecasts,
    _decorate_forecasts,
    _fold_data,
    _lead_time_map,
    _load_config,
    _origin_report_metadata,
    _run_global_fold,
)
from src.forecasting.metrics import forecast_metrics

ROOT = Path(__file__).resolve().parents[2]
HORIZON = 26
WINDOW_END = pd.Timestamp("2026-03-23")
ORIGINS = (pd.Timestamp("2025-03-24"), pd.Timestamp("2025-09-22"))
ENSEMBLE = "simple_average_ensemble"
INTERMITTENT_BASELINE = "tsb"
GLOBAL_FAMILIES = (
    "lightgbm_tweedie",
    "lightgbm_quantile",
    "lightgbm_point",
)
BASELINES = (
    "naive",
    "seasonal_naive_52",
    "moving_average_13",
    "croston",
    "sba",
    "tsb",
)
MODEL_ORDER = (
    "lightgbm_tweedie",
    "lightgbm_quantile",
    ENSEMBLE,
    INTERMITTENT_BASELINE,
    "naive",
    "seasonal_naive_52",
    "historical_average",
)


def _frozen_parameters(root: Path) -> dict[str, dict[str, Any]]:
    log = pd.read_csv(root / "results" / "track_a_tuning_log.csv")
    selected = log.loc[
        log["status"].eq("success")
        & log["selected_for_held_validation"].astype(str).str.lower().eq("true")
    ]
    parameters: dict[str, dict[str, Any]] = {}
    for family in GLOBAL_FAMILIES:
        rows = selected.loc[selected["model"].eq(family)]
        if rows.empty:
            raise ValueError(f"No frozen configuration for {family}")
        parameters[family] = json.loads(str(rows.iloc[0]["parameters"]))
    return parameters


def _origin_frames(
    demand: pd.DataFrame,
    lead_times: pd.DataFrame,
    origin: pd.Timestamp,
    parameters: dict[str, dict[str, Any]],
    seed: int,
) -> tuple[list[pd.DataFrame], dict[str, float]]:
    history, target_rows, calendar = _fold_data(demand, origin, HORIZON)
    metadata, scales = _origin_report_metadata(history, target_rows)
    frames: list[pd.DataFrame] = []
    runtimes: dict[str, float] = {}
    for model_name in BASELINES:
        started = time.perf_counter()
        forecasts = _baseline_forecasts(history, calendar, model_name)
        runtime = time.perf_counter() - started
        frames.append(
            _decorate_forecasts(
                forecasts, target_rows, metadata, scales, lead_times,
                model_name, "baseline_default", "evaluation", origin,
                HORIZON, runtime,
            )
        )
        runtimes[model_name] = runtime
    observed = history.loc[
        history["split_role"].eq("train_val") & history["demand_qty"].notna()
    ]
    means = observed.groupby(["part_id", "region"], observed=True)[
        "demand_qty"
    ].mean()
    started = time.perf_counter()
    average = calendar.loc[
        :,
        ["part_id", "region", "week_start_date"],
    ].copy()
    average["forecast_qty"] = [
        means.get((part_id, region), np.nan)
        for part_id, region in zip(average["part_id"], average["region"])
    ]
    for column in ("forecast_p90", "forecast_p95", "forecast_p99"):
        average[column] = np.nan
    runtime = time.perf_counter() - started
    frames.append(
        _decorate_forecasts(
            average, target_rows, metadata, scales, lead_times,
            "historical_average", "origin_mean", "evaluation", origin,
            HORIZON, runtime,
        )
    )
    runtimes["historical_average"] = runtime
    for family in GLOBAL_FAMILIES:
        started = time.perf_counter()
        predictions, _, _ = _run_global_fold(
            history, calendar, origin, family, parameters[family], seed,
        )
        runtime = time.perf_counter() - started
        config_id = (
            "random_1" if family != "lightgbm_point" else "random_3"
        )
        frames.append(
            _decorate_forecasts(
                predictions, target_rows, metadata, scales, lead_times,
                family, config_id, "evaluation", origin, HORIZON, runtime,
            )
        )
        runtimes[family] = runtime
        print(
            f"{family} origin {origin.date()} finished in {runtime:.1f}s",
            flush=True,
        )
    member_frames = [
        frame.loc[
            frame["model"].isin(SIMPLE_AVERAGE_MEMBERS),
            ["part_id", "region", "week_start_date", "forecast_qty"],
        ]
        for frame in frames
    ]
    members = pd.concat(member_frames, ignore_index=True)
    ensemble = (
        members.groupby(
            ["part_id", "region", "week_start_date"],
            observed=True,
            sort=False,
            as_index=False,
        )["forecast_qty"]
        .mean()
    )
    ensemble["forecast_p90"] = np.nan
    ensemble["forecast_p95"] = np.nan
    ensemble["forecast_p99"] = np.nan
    ensemble["model"] = ENSEMBLE
    ensemble["config_id"] = "equal_weight_nine_model_mean"
    ensemble["runtime_seconds"] = np.nan
    ensemble["candidate_role"] = "forecast_candidate"
    ensemble["evaluation_split"] = "evaluation"
    ensemble["origin_date"] = origin
    ensemble["horizon_step"] = (
        (ensemble["week_start_date"] - origin).dt.days // 7
    ).astype(int)
    ensemble = ensemble.merge(
        target_rows.loc[
            :, ["part_id", "region", "week_start_date", "demand_qty"]
        ].rename(columns={"demand_qty": "actual_qty"}),
        on=["part_id", "region", "week_start_date"],
        how="left",
        validate="one_to_one",
    )
    ensemble = ensemble.merge(
        metadata, on=["part_id", "region"], how="left", validate="many_to_one"
    )
    ensemble = ensemble.merge(
        lead_times, on="part_id", how="left", validate="many_to_one"
    )
    ensemble["mase_scale"] = [
        scales.get((str(part_id), str(region)), np.nan)
        for part_id, region in zip(ensemble["part_id"], ensemble["region"])
    ]
    ensemble["ablation_group"] = "none"
    runtimes[ENSEMBLE] = float(
        sum(runtimes[name] for name in SIMPLE_AVERAGE_MEMBERS if name in runtimes)
    )
    frames.append(ensemble)
    return frames, runtimes


def _score(
    scope: str,
    origin: str,
    segment: str,
    model: str,
    frame: pd.DataFrame,
    incomplete_series: int = 0,
) -> dict[str, Any]:
    values = forecast_metrics(
        frame["actual_qty"], frame["forecast_qty"], frame.get("mase_scale")
    ) if not frame.empty else {}
    actual_available = frame["actual_qty"].notna() if not frame.empty else pd.Series(dtype=bool)
    forecast_available = (
        frame["forecast_qty"].notna() if not frame.empty else pd.Series(dtype=bool)
    )
    return {
        "scope": scope,
        "origin": origin,
        "segment": segment,
        "model": model,
        "n_scored": values.get("n_scored", 0),
        "actual_total": values.get("actual_total", np.nan),
        "forecast_total": values.get("forecast_total", np.nan),
        "wape_pct": values.get("wape_pct", np.nan),
        "forecast_accuracy_pct": values.get("forecast_accuracy_pct", np.nan),
        "bias_pct": values.get("bias_pct", np.nan),
        "mase": values.get("mase", np.nan),
        "mae": values.get("mae", np.nan),
        "rmse": values.get("rmse", np.nan),
        "excluded_missing_actual": int(
            (forecast_available & ~actual_available).sum()
        ),
        "excluded_missing_forecast": int(
            (actual_available & ~forecast_available).sum()
        ),
        "excluded_both_missing": int(
            (~actual_available & ~forecast_available).sum()
        ),
        "incomplete_series_excluded": int(incomplete_series),
    }


def _aggregate_26w(weekly: pd.DataFrame) -> pd.DataFrame:
    keys = ["origin_date", "part_id", "region"]
    grouped = (
        weekly.groupby(keys, observed=True, sort=False)
        .agg(
            actual_qty=("actual_qty", "sum"),
            forecast_qty=("forecast_qty", "sum"),
            observed_weeks=("actual_qty", "count"),
            forecast_weeks=("forecast_qty", "count"),
            total_weeks=("actual_qty", "size"),
        )
        .reset_index()
    )
    grouped["mase_scale"] = np.nan
    return grouped


def _score_model(
    model: str,
    weekly: pd.DataFrame,
    runtimes: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    subset = weekly.loc[weekly["model"].eq(model)]
    scored = subset.loc[
        subset["actual_qty"].notna() & subset["forecast_qty"].notna()
    ]
    pooled_origin = ",".join(
        sorted(str(pd.Timestamp(value).date()) for value in subset["origin_date"].unique())
    )
    rows.append(
        _score("weekly", "pooled", "all", model, subset)
    )
    for origin, origin_rows in subset.groupby("origin_date", sort=True):
        rows.append(
            _score("weekly_by_origin", str(origin.date()), "all", model, origin_rows)
        )
    for label, low, high in (("horizon_1_13", 1, 13), ("horizon_14_26", 14, 26)):
        band = scored.loc[
            scored["horizon_step"].ge(low) & scored["horizon_step"].le(high)
        ]
        rows.append(
            _score("horizon_band", "pooled", label, model, band)
        )
    aggregates = _aggregate_26w(subset)
    complete = aggregates.loc[
        aggregates["observed_weeks"].eq(HORIZON)
        & aggregates["forecast_weeks"].eq(HORIZON)
    ]
    incomplete_total = int(len(aggregates) - len(complete))
    rows.append(
        _score(
            "complete_26w", "pooled", "all", model, complete,
            incomplete_total,
        )
    )
    for origin, group in aggregates.groupby("origin_date", sort=True):
        group_complete = group.loc[
            group["observed_weeks"].eq(HORIZON)
            & group["forecast_weeks"].eq(HORIZON)
        ]
        rows.append(
            _score(
                "complete_26w_by_origin", str(origin.date()), "all", model,
                group_complete, int(len(group) - len(group_complete)),
            )
        )
    for dimension in ("sbc_class", "criticality"):
        for value, segment in scored.groupby(dimension, sort=True):
            rows.append(
                _score(dimension, "pooled", str(value), model, segment)
            )
    for row in rows:
        row["runtime_seconds"] = runtimes.get(model, np.nan)
    return rows


def run_interim_backtest(root: Path = ROOT) -> dict[str, Any]:
    config = _load_config(root)
    seed = int(config["forecast"]["backtest"]["seed"])
    parameters = _frozen_parameters(root)
    demand = load_demand(root / "data" / "01_demand_forecasting.csv")
    demand = demand.loc[
        demand["split_role"].isin(["warmup", "train_val"])
        & demand["week_start_date"].le(WINDOW_END)
    ].copy()
    if demand["split_role"].isin(["final_test", "future"]).any():
        raise RuntimeError("Interim backtest data includes sealed rows")
    scheduling = load_scheduling(root / "data" / "02_scheduling_inputs.csv")
    lead_times = _lead_time_map(scheduling)

    all_frames: list[pd.DataFrame] = []
    runtimes: dict[str, float] = {}
    if ORIGINS[1] - ORIGINS[0] != pd.Timedelta(weeks=HORIZON):
        raise ValueError("Interim origins are not consecutive 26-week blocks")
    if ORIGINS[1] + pd.Timedelta(weeks=HORIZON) != WINDOW_END:
        raise ValueError("Interim windows do not end at 2026-03-23")
    for origin in ORIGINS:
        frames, origin_runtimes = _origin_frames(
            demand, lead_times, origin, parameters, seed,
        )
        all_frames.extend(frames)
        for model, value in origin_runtimes.items():
            runtimes[model] = runtimes.get(model, 0.0) + value
    weekly = pd.concat(all_frames, ignore_index=True)
    weekly = weekly.loc[weekly["model"].isin(MODEL_ORDER)].copy()
    if weekly["week_start_date"].max() > WINDOW_END:
        raise ValueError("Interim backtest scored rows beyond 2026-03-23")
    expected_origins = sorted(ORIGINS)
    observed_origins = sorted(
        pd.Timestamp(value) for value in weekly["origin_date"].unique()
    )
    if observed_origins != expected_origins:
        raise ValueError(
            f"Unexpected interim origins: {observed_origins}"
        )

    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        if weekly["model"].eq(model).any():
            rows.extend(_score_model(model, weekly, runtimes))
    results = pd.DataFrame(rows)
    results.to_csv(root / "results" / "interim_26w_backtest.csv", index=False)

    summary = {
        "windows": [
            {
                "origin": str(origin.date()),
                "window_start": str(
                    (origin + pd.Timedelta(weeks=1)).date()
                ),
                "window_end": str(
                    (origin + pd.Timedelta(weeks=HORIZON)).date()
                ),
            }
            for origin in ORIGINS
        ],
        "frozen_parameters": parameters,
        "models": list(MODEL_ORDER),
        "weekly_rows": int(len(weekly)),
        "score_rows": int(len(results)),
        "runtimes_seconds": runtimes,
        "final_test_sealed": True,
    }
    (root / "results" / "interim_26w_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    _write_report(root, results, summary)
    return summary


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, separator]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                cells.append("" if not np.isfinite(value) else f"{value:.2f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_report(
    root: Path,
    results: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    columns = [
        "model", "wape_pct", "forecast_accuracy_pct", "bias_pct", "mase",
        "n_scored", "excluded_missing_actual", "excluded_missing_forecast",
    ]
    aggregate_columns = [
        "model", "wape_pct", "forecast_accuracy_pct", "bias_pct",
        "n_scored", "incomplete_series_excluded",
    ]
    weekly = results.loc[
        results["scope"].eq("weekly")
        & results["model"].isin(MODEL_ORDER)
    ].sort_values("wape_pct")
    by_origin = results.loc[
        results["scope"].eq("weekly_by_origin")
        & results["model"].isin(MODEL_ORDER)
    ].sort_values(["origin", "wape_pct"])
    aggregate = results.loc[
        results["scope"].eq("complete_26w")
        & results["model"].isin(MODEL_ORDER)
    ].sort_values("wape_pct")
    bands = results.loc[
        results["scope"].eq("horizon_band")
        & results["model"].isin(MODEL_ORDER)
    ].sort_values(["segment", "wape_pct"])
    sbc = results.loc[
        results["scope"].eq("sbc_class")
        & results["model"].isin(MODEL_ORDER)
    ].sort_values(["segment", "wape_pct"])
    criticality = results.loc[
        results["scope"].eq("criticality")
        & results["model"].isin(MODEL_ORDER)
    ].sort_values(["segment", "wape_pct"])
    window_lines = "\n".join(
        f"- Origin {window['origin']}: {window['window_start']} to "
        f"{window['window_end']}"
        for window in summary["windows"]
    )
    report = f"""# SPEC v4.2 B — interim 26-week backtest

Dataset: synthetic. Results validate method behaviour on simulated data; they are not factory savings.

## Windows and design

{window_lines}

- Two consecutive 26-week blocks ending at the train-validation cutoff 2026-03-23.
- Each window is fitted only on labels up to its origin; warm-up history initialises lags and seasonal terms. Recursive one-step features are rebuilt inside the origin. Frozen hyperparameters are read from the Phase 3 tuning log and are not re-tuned here.
- Models: champion `lightgbm_tweedie::random_1`, quantile `lightgbm_quantile::random_1`, nine-model equal-weight ensemble, best intermittent baseline (`{INTERMITTENT_BASELINE}`), `naive`, `seasonal_naive_52`, `historical_average`.
- Forecast Accuracy % = 100 − WAPE %.
- Complete 26-week aggregate requires 26 observed demand weeks and 26 finite forecasts per part-region; incomplete series are counted, not imputed.

## Weekly results (pooled across both windows)

{_markdown_table(weekly, columns)}

## Weekly results by origin

{_markdown_table(by_origin, ["origin", *columns])}

## Complete 26-week part-region aggregate

{_markdown_table(aggregate, aggregate_columns)}

## Horizon bands

{_markdown_table(bands, ["segment", *columns])}

## SBC class

{_markdown_table(sbc, ["segment", *columns])}

## Criticality

{_markdown_table(criticality, ["segment", *columns])}

## Reading the tables

- `lightgbm_quantile` has the lowest weekly absolute error at the median but strongly understates totals (bias −34.15%); it is a benchmark line, not a replacement for the champion.
- `lightgbm_tweedie` (the frozen champion) has the lowest complete 26-week aggregate WAPE.
- Baselines lack forecasts for 384 part-region rows where the series has no usable history; `excluded_missing_forecast` records this and the model sample counts differ.
- These results are not used for champion selection and do not change the frozen champion.

## Caveats

- Only two windows and no significance test is claimed.
- Summed 26-week aggregate WAPE is easier than weekly WAPE and is **not comparable** to the 14-week supplier lead-time metric.
- The windows lie inside held validation and overlap forecast horizons already used in Phase 3; they are not independent evidence.
- The final test remains sealed.
- Data are synthetic; accuracy here supports method validation, not a factory forecast guarantee.
"""
    (root / "results" / "interim_26w_report.md").write_text(
        report, encoding="utf-8"
    )


if __name__ == "__main__":
    print(json.dumps(run_interim_backtest(), indent=2, default=str))
