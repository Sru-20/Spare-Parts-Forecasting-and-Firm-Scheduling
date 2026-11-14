"""SPEC v4.2 A1 — tuning-origin comparison and pre-declared advancement gate.

Runs moving average, exponential smoothing (ETS(A,N,N)) and XGBoost on the two
tuning origins with the frozen 3-configuration seed-42 budget, scores them next
to the frozen champion, and only opens a held evaluation when a candidate beats
the champion by at least 0.50 percentage points of tuning lead-time WAPE.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.loader import load_demand, load_scheduling
from src.forecasting.backtest import (
    _baseline_forecasts,
    _decorate_forecasts,
    _fold_data,
    _lead_time_map,
    _load_config,
    _origin_report_metadata,
    _run_global_fold,
    _search_configurations,
)
from src.forecasting.phase4_v41 import (
    _lead_time_rows,
    _metric_values,
    _paired_wape_difference_ci,
)

ROOT = Path(__file__).resolve().parents[2]
CHAMPION_MODEL = "lightgbm_tweedie"
CHAMPION_CONFIG = "random_1"
CHAMPION_LABEL = f"{CHAMPION_MODEL}::{CHAMPION_CONFIG}"
TUNING_FORECAST_ARTIFACT = "artifacts/bias_calibration_tuning_forecasts.csv.gz"
HELD_FORECAST_ARTIFACT = "artifacts/track_a_backtest_forecasts_v41.parquet"
BASELINE_CANDIDATES = (
    "moving_average_13",
    "exponential_smoothing",
)
REPORTED_BASELINES = (
    "naive",
    "seasonal_naive_52",
    "tsb",
)
TUNING_GATE_PP = 0.50
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 42
BOOTSTRAP_CONFIDENCE = 0.95


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _total_runtime(frame: pd.DataFrame) -> float:
    if "runtime_seconds" not in frame:
        return float("nan")
    runtimes = pd.to_numeric(frame["runtime_seconds"], errors="coerce")
    if runtimes.notna().sum() == 0:
        return float("nan")
    origins = frame["origin_date"]
    return float(runtimes.groupby(origins, sort=False).max().sum())


def _xgboost_configurations(search: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    """Map the frozen LightGBM search budget onto XGBoost equivalents."""
    configurations = _search_configurations(search, seed)
    mapped: list[dict[str, Any]] = []
    for parameters in configurations:
        mapped.append(
            {
                "n_estimators": int(parameters["n_estimators"]),
                "learning_rate": float(parameters["learning_rate"]),
                "max_depth": max(2, int(math.ceil(math.log2(parameters["num_leaves"])))),
                "min_child_weight": float(parameters["min_child_samples"]),
                "colsample_bytree": float(parameters["feature_fraction"]),
                "subsample": 1.0,
            }
        )
    return mapped


def _metric_row(
    label: str,
    family: str,
    role: str,
    evaluation_split: str,
    level: str,
    frame: pd.DataFrame,
    runtime_seconds: float,
) -> dict[str, Any]:
    values = _metric_values(frame) if not frame.empty else {}
    return {
        "candidate_id": label,
        "model": family,
        "candidate_role": role,
        "evaluation_split": evaluation_split,
        "evaluation_level": level,
        "n_scored": values.get("n_scored", 0),
        "actual_total": values.get("actual_total", np.nan),
        "forecast_total": values.get("forecast_total", np.nan),
        "wape_pct": values.get("wape_pct", np.nan),
        "forecast_accuracy_pct": values.get("forecast_accuracy_pct", np.nan),
        "bias_pct": values.get("bias_pct", np.nan),
        "mase": values.get("mase", np.nan),
        "mae": values.get("mae", np.nan),
        "rmse": values.get("rmse", np.nan),
        "mape_pct": values.get("mape_pct", np.nan),
        "mape_nonzero_n": values.get("mape_nonzero_n", np.nan),
        "mape_zero_excluded": values.get("mape_zero_excluded", np.nan),
        "runtime_seconds": runtime_seconds,
    }


def _baseline_frame(
    history: pd.DataFrame,
    target_rows: pd.DataFrame,
    calendar: pd.DataFrame,
    model_name: str,
    lead_times: pd.DataFrame,
    metadata: pd.DataFrame,
    scales: dict[tuple[str, str], float],
    origin: pd.Timestamp,
    horizon: int,
) -> tuple[pd.DataFrame, float]:
    started = time.perf_counter()
    forecasts = _baseline_forecasts(history, calendar, model_name)
    runtime = time.perf_counter() - started
    decorated = _decorate_forecasts(
        forecasts,
        target_rows,
        metadata,
        scales,
        lead_times,
        model_name,
        "baseline_default",
        "tuning",
        origin,
        horizon,
        runtime,
    )
    return decorated, runtime


def _historical_average_frame(
    history: pd.DataFrame,
    target_rows: pd.DataFrame,
    calendar: pd.DataFrame,
    lead_times: pd.DataFrame,
    metadata: pd.DataFrame,
    scales: dict[tuple[str, str], float],
    origin: pd.Timestamp,
    horizon: int,
) -> tuple[pd.DataFrame, float]:
    started = time.perf_counter()
    observed = history.loc[
        history["split_role"].eq("train_val") & history["demand_qty"].notna()
    ]
    means = observed.groupby(["part_id", "region"], observed=True)[
        "demand_qty"
    ].mean()
    forecasts = calendar.copy()
    forecasts["forecast_qty"] = [
        means.get((part_id, region), np.nan)
        for part_id, region in zip(forecasts["part_id"], forecasts["region"])
    ]
    for column in ("forecast_p90", "forecast_p95", "forecast_p99"):
        forecasts[column] = np.nan
    runtime = time.perf_counter() - started
    forecasts = forecasts.loc[
        :,
        [
            "part_id", "region", "week_start_date", "forecast_qty",
            "forecast_p90", "forecast_p95", "forecast_p99",
        ],
    ]
    decorated = _decorate_forecasts(
        forecasts,
        target_rows,
        metadata,
        scales,
        lead_times,
        "historical_average",
        "origin_mean",
        "tuning",
        origin,
        horizon,
        runtime,
    )
    return decorated, runtime


def _held_rows(forecasts: pd.DataFrame, model: str, config_id: str) -> pd.DataFrame:
    return forecasts.loc[
        forecasts["model"].eq(model) & forecasts["config_id"].eq(config_id)
    ].copy()


def _held_forecasts_for(
    root: Path,
    demand: pd.DataFrame,
    lead_times: pd.DataFrame,
    origin: pd.Timestamp,
    parameters: dict[str, Any] | None,
    model_name: str,
    config_id: str,
    horizon: int,
    seed: int,
) -> tuple[pd.DataFrame, float]:
    history, target_rows, calendar = _fold_data(demand, origin, horizon)
    metadata, scales = _origin_report_metadata(history, target_rows)
    if parameters is None:
        forecasts, runtime = _baseline_frame(
            history, target_rows, calendar, model_name, lead_times,
            metadata, scales, origin, horizon,
        )
        forecasts["config_id"] = config_id
        return forecasts, runtime
    started = time.perf_counter()
    predictions, _, _ = _run_global_fold(
        history, calendar, origin, model_name, parameters, seed,
    )
    runtime = time.perf_counter() - started
    decorated = _decorate_forecasts(
        predictions,
        target_rows,
        metadata,
        scales,
        lead_times,
        model_name,
        config_id,
        "evaluation",
        origin,
        horizon,
        runtime,
    )
    return decorated, runtime


def _origin_wins(
    candidate_lead: pd.DataFrame,
    champion_lead: pd.DataFrame,
) -> tuple[int, int]:
    keys = ["origin_date", "part_id", "region"]
    merged = candidate_lead.loc[:, [*keys, "actual_qty", "forecast_qty"]].merge(
        champion_lead.loc[:, [*keys, "actual_qty", "forecast_qty"]],
        on=keys,
        suffixes=("_candidate", "_champion"),
        validate="one_to_one",
    )
    wins = 0
    total = 0
    for _, group in merged.groupby("origin_date", sort=True):
        candidate_wape = 100.0 * group["forecast_qty_candidate"].sub(
            group["actual_qty"]
        ).abs().sum() / group["actual_qty"].sum()
        champion_wape = 100.0 * group["forecast_qty_champion"].sub(
            group["actual_qty"]
        ).abs().sum() / group["actual_qty"].sum()
        wins += int(candidate_wape < champion_wape)
        total += 1
    return wins, total


def _decide(
    tuning_table: pd.DataFrame,
    tuning_gate_pp: float,
) -> dict[str, Any]:
    champion = tuning_table.loc[
        tuning_table["candidate_id"].eq(CHAMPION_LABEL)
        & tuning_table["evaluation_level"].eq("lead_time")
    ]
    if champion.empty:
        raise RuntimeError("Champion tuning metrics are missing")
    champion_wape = float(champion.iloc[0]["wape_pct"])
    candidates = tuning_table.loc[
        tuning_table["candidate_role"].eq("new_candidate")
        & tuning_table["evaluation_level"].eq("lead_time")
    ].copy()
    candidates = candidates.loc[candidates["wape_pct"].notna()]
    if candidates.empty:
        return {
            "champion_tuning_lead_wape_pct": champion_wape,
            "best_candidate": None,
            "best_candidate_tuning_lead_wape_pct": np.nan,
            "tuning_gain_pp": np.nan,
            "tuning_gate_passed": False,
            "held_evaluation_opened": False,
            "decision": "champion_retained_no_finite_candidate",
        }
    candidates = candidates.sort_values("wape_pct")
    best = candidates.iloc[0]
    gain = champion_wape - float(best["wape_pct"])
    return {
        "champion_tuning_lead_wape_pct": champion_wape,
        "best_candidate": str(best["candidate_id"]),
        "best_candidate_tuning_lead_wape_pct": float(best["wape_pct"]),
        "tuning_gain_pp": float(gain),
        "tuning_gate_threshold_pp": tuning_gate_pp,
        "tuning_gate_passed": bool(gain >= tuning_gate_pp),
        "held_evaluation_opened": False,
        "decision": "champion_retained_tuning_gate_not_met",
    }


def _held_evaluation(
    root: Path,
    demand: pd.DataFrame,
    lead_times: pd.DataFrame,
    held_forecasts: pd.DataFrame,
    held_origins: list[pd.Timestamp],
    candidate_label: str,
    candidate_model: str,
    candidate_config: str,
    xgboost_parameters: dict[str, Any] | None,
    horizon: int,
    extension_weeks: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    champion = _held_rows(held_forecasts, CHAMPION_MODEL, CHAMPION_CONFIG)
    if candidate_label == CHAMPION_LABEL:
        candidate = champion
    elif candidate_model == "exponential_smoothing":
        frames: list[pd.DataFrame] = []
        runtimes: list[float] = []
        for origin in held_origins:
            frame, runtime = _held_forecasts_for(
                root, demand, lead_times, origin, None,
                candidate_model, candidate_config, horizon, seed,
            )
            frame["evaluation_split"] = "evaluation"
            frames.append(frame)
            runtimes.append(runtime)
        candidate = pd.concat(frames, ignore_index=True)
        candidate["runtime_seconds"] = sum(runtimes)
    elif xgboost_parameters is not None:
        frames = []
        runtimes = []
        for origin in held_origins:
            frame, runtime = _held_forecasts_for(
                root, demand, lead_times, origin, xgboost_parameters,
                candidate_model, candidate_config, horizon, seed,
            )
            frames.append(frame)
            runtimes.append(runtime)
        candidate = pd.concat(frames, ignore_index=True)
        candidate["runtime_seconds"] = sum(runtimes)
    else:
        candidate = _held_rows(held_forecasts, candidate_model, candidate_config)
    if candidate.empty:
        raise RuntimeError(f"No held forecast rows for {candidate_label}")

    candidate_lead = _lead_time_rows(candidate, extension_weeks)
    champion_lead = _lead_time_rows(champion, extension_weeks)
    rows = [
        _metric_row(
            candidate_label, candidate_model, "advanced_candidate",
            "evaluation", "weekly", candidate, _total_runtime(candidate),
        ),
        _metric_row(
            candidate_label, candidate_model, "advanced_candidate",
            "evaluation", "lead_time", candidate_lead, _total_runtime(candidate),
        ),
        _metric_row(
            CHAMPION_LABEL, CHAMPION_MODEL, "champion", "evaluation",
            "weekly", champion, _total_runtime(champion),
        ),
        _metric_row(
            CHAMPION_LABEL, CHAMPION_MODEL, "champion", "evaluation",
            "lead_time", champion_lead, _total_runtime(champion),
        ),
    ]
    baseline_rows: list[dict[str, Any]] = []
    for baseline_model, baseline_config in (
        ("seasonal_naive_52", "baseline_default"),
        ("historical_average", "origin_mean"),
    ):
        reference = _held_rows(held_forecasts, baseline_model, baseline_config)
        if reference.empty:
            continue
        reference_lead = _lead_time_rows(reference, extension_weeks)
        baseline_rows.append(
            _metric_row(
                f"{baseline_model}::{baseline_config}", baseline_model,
                "reported_reference", "evaluation", "lead_time",
                reference_lead, _total_runtime(reference),
            )
        )
    lead_table = pd.DataFrame([*rows, *baseline_rows])
    point, low, high = _paired_wape_difference_ci(
        candidate_lead, champion_lead, BOOTSTRAP_REPLICATES,
        BOOTSTRAP_SEED, BOOTSTRAP_CONFIDENCE,
    )
    wins, origins = _origin_wins(candidate_lead, champion_lead)
    candidate_lead_row = lead_table.loc[
        lead_table["candidate_id"].eq(candidate_label)
        & lead_table["evaluation_level"].eq("lead_time")
    ].iloc[0]
    baseline_wapes = lead_table.loc[
        lead_table["candidate_role"].eq("reported_reference")
        & lead_table["evaluation_level"].eq("lead_time"),
        "wape_pct",
    ].astype(float)
    baseline_beating = bool(
        not baseline_wapes.empty
        and float(candidate_lead_row["wape_pct"]) < float(baseline_wapes.min())
    )
    gates = {
        "paired_ci_point_pp": point,
        "paired_ci_low_pp": low,
        "paired_ci_high_pp": high,
        "paired_ci_upper_below_zero": bool(np.isfinite(high) and high < 0),
        "origin_wins": wins,
        "origin_count": origins,
        "origin_wins_required": 3,
        "origin_win_gate_passed": bool(wins >= 3),
        "absolute_bias_pct": float(candidate_lead_row["bias_pct"]),
        "bias_within_10pp": bool(
            abs(float(candidate_lead_row["bias_pct"])) <= 10.0
        ),
        "baseline_beating": baseline_beating,
    }
    normal_gates_passed = bool(
        gates["paired_ci_upper_below_zero"]
        and gates["origin_win_gate_passed"]
        and baseline_beating
    )
    gates["normal_gates_passed"] = normal_gates_passed
    gates["decision"] = (
        "champion_displaced_by_normal_gates"
        if normal_gates_passed
        else "champion_retained_after_held_evaluation"
    )
    return lead_table, gates


def run_model_comparison(root: Path = ROOT) -> dict[str, Any]:
    config = _load_config(root)
    forecast_config = config["forecast"]
    backtest_config = forecast_config["backtest"]
    horizon = int(backtest_config["horizon_weeks"])
    extension_weeks = int(
        backtest_config["over_horizon_extension_period_weeks"]
    )
    seed = int(backtest_config["seed"])
    origins = [pd.Timestamp(value) for value in backtest_config["origins"]]
    tuning_count = int(backtest_config["tuning_origin_count"])
    tuning_origins = origins[:tuning_count]
    held_origins = origins[tuning_count:]
    search = backtest_config["lightgbm_search"]
    xgboost_configs = _xgboost_configurations(search, seed)

    demand = load_demand(root / "data" / "01_demand_forecasting.csv")
    demand = demand.loc[
        demand["split_role"].isin(["warmup", "train_val"])
        & demand["week_start_date"].le(pd.Timestamp("2026-03-23"))
    ].copy()
    scheduling = load_scheduling(root / "data" / "02_scheduling_inputs.csv")
    lead_times = _lead_time_map(scheduling)

    tuning_path = root / TUNING_FORECAST_ARTIFACT
    if not tuning_path.is_file():
        raise FileNotFoundError(
            f"Missing frozen champion tuning forecasts: {tuning_path}"
        )
    champion_tuning = pd.read_csv(tuning_path)
    champion_tuning["origin_date"] = pd.to_datetime(champion_tuning["origin_date"])
    champion_tuning["week_start_date"] = pd.to_datetime(
        champion_tuning["week_start_date"]
    )
    if champion_tuning["evaluation_split"].ne("tuning").any():
        raise ValueError("Champion tuning artifact must contain tuning rows only")

    frames: list[pd.DataFrame] = [champion_tuning]
    for origin in tuning_origins:
        history, target_rows, calendar = _fold_data(demand, origin, horizon)
        metadata, scales = _origin_report_metadata(history, target_rows)
        for model_name in (*BASELINE_CANDIDATES, *REPORTED_BASELINES):
            decorated, runtime = _baseline_frame(
                history, target_rows, calendar, model_name, lead_times,
                metadata, scales, origin, horizon,
            )
            frames.append(decorated)
        average, runtime = _historical_average_frame(
            history, target_rows, calendar, lead_times, metadata, scales,
            origin, horizon,
        )
        frames.append(average)
        for index, parameters in enumerate(xgboost_configs, start=1):
            config_id = f"random_{index}"
            started = time.perf_counter()
            predictions, _, _ = _run_global_fold(
                history, calendar, origin, "xgboost", parameters, seed,
            )
            runtime = time.perf_counter() - started
            decorated = _decorate_forecasts(
                predictions,
                target_rows,
                metadata,
                scales,
                lead_times,
                "xgboost",
                config_id,
                "tuning",
                origin,
                horizon,
                runtime,
            )
            frames.append(decorated)
            print(
                f"xgboost {config_id} origin {origin.date()} "
                f"finished in {runtime:.1f}s",
                flush=True,
            )

    tuning = pd.concat(frames, ignore_index=True)
    tuning["evaluation_level"] = "weekly"
    tuning_lead = _lead_time_rows(tuning, extension_weeks)

    tuning_rows: list[dict[str, Any]] = []
    for (model, config_id), group in tuning.groupby(
        ["model", "config_id"], observed=True, sort=False
    ):
        label = f"{model}::{config_id}"
        role = (
            "champion"
            if label == CHAMPION_LABEL
            else "new_candidate"
            if model in {*BASELINE_CANDIDATES, "xgboost"}
            else "reported_reference"
        )
        lead_group = tuning_lead.loc[
            tuning_lead["model"].eq(model)
            & tuning_lead["config_id"].eq(config_id)
        ]
        tuning_rows.append(
            _metric_row(
                label, model, role, "tuning", "weekly", group,
                _total_runtime(group),
            )
        )
        tuning_rows.append(
            _metric_row(
                label, model, role, "tuning", "lead_time", lead_group,
                _total_runtime(group),
            )
        )
    tuning_table = pd.DataFrame(tuning_rows).sort_values(
        ["evaluation_level", "wape_pct"], na_position="last"
    )
    tuning_table.to_csv(root / "results" / "model_comparison_tuning.csv", index=False)

    decision = _decide(tuning_table, TUNING_GATE_PP)
    held_table = pd.DataFrame(
        columns=[
            "candidate_id", "model", "candidate_role", "evaluation_split",
            "evaluation_level", "n_scored", "actual_total", "forecast_total",
            "wape_pct", "forecast_accuracy_pct", "bias_pct", "mase",
            "mae", "rmse", "mape_pct", "mape_nonzero_n",
            "mape_zero_excluded", "runtime_seconds",
        ]
    )
    held_gates: dict[str, Any] = {}
    if decision["tuning_gate_passed"]:
        best_label = str(decision["best_candidate"])
        model_name, config_id = best_label.split("::")
        parameters = None
        if model_name == "xgboost":
            parameters = xgboost_configs[int(config_id.split("_")[1]) - 1]
        held_path = root / HELD_FORECAST_ARTIFACT
        held_forecasts = pd.read_parquet(held_path)
        held_forecasts["origin_date"] = pd.to_datetime(
            held_forecasts["origin_date"]
        )
        held_forecasts["week_start_date"] = pd.to_datetime(
            held_forecasts["week_start_date"]
        )
        held_table, held_gates = _held_evaluation(
            root, demand, lead_times, held_forecasts, held_origins,
            best_label, model_name, config_id, parameters, horizon,
            extension_weeks, seed,
        )
        decision["held_evaluation_opened"] = True
        decision["decision"] = held_gates["decision"]
        decision["held_gates"] = held_gates
        held_table.to_csv(
            root / "results" / "model_comparison_held.csv", index=False
        )
    else:
        decision["held_evaluation_opened"] = False

    summary = {
        "tuning_gate_pp": TUNING_GATE_PP,
        "tuning_origins": [str(origin.date()) for origin in tuning_origins],
        "held_origins_not_used": [
            str(origin.date()) for origin in held_origins
        ] if not decision["held_evaluation_opened"] else [],
        "xgboost_configs": xgboost_configs,
        "deferred_models": {
            "sarima_arima": (
                "616 mostly intermittent series, synthetic data and a 14-week "
                "horizon; per-series fits add cost without decision value"
            ),
            "lstm": (
                "no GPU, small synthetic sample, recursive tabular challengers "
                "cover the non-linear comparison at lower cost"
            ),
        },
        "decision": decision,
        "held_gates": held_gates,
    }
    (root / "results" / "model_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    _write_report(root, tuning_table, held_table, decision, held_gates)
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
    tuning_table: pd.DataFrame,
    held_table: pd.DataFrame,
    decision: dict[str, Any],
    held_gates: dict[str, Any],
) -> None:
    weekly = tuning_table.loc[tuning_table["evaluation_level"].eq("weekly")]
    lead = tuning_table.loc[tuning_table["evaluation_level"].eq("lead_time")]
    weekly_view = weekly.loc[
        :,
        [
            "candidate_id", "candidate_role", "wape_pct", "bias_pct",
            "mase", "mae", "rmse", "mape_pct", "mape_nonzero_n",
            "mape_zero_excluded", "runtime_seconds",
        ],
    ]
    lead_view = lead.loc[
        :,
        [
            "candidate_id", "candidate_role", "wape_pct", "bias_pct",
            "mase", "mae", "rmse", "mape_pct", "mape_nonzero_n",
            "mape_zero_excluded", "runtime_seconds",
        ],
    ]
    decision_text = (
        f"Tuning gate: **{'PASSED' if decision['tuning_gate_passed'] else 'NOT MET'}** "
        f"— best new candidate `{decision['best_candidate']}` reached "
        f"{decision['best_candidate_tuning_lead_wape_pct']:.2f}% lead-time WAPE "
        f"against the champion's {decision['champion_tuning_lead_wape_pct']:.2f}% "
        f"(gain {decision['tuning_gain_pp']:.2f} pp).\n\n"
    )
    if decision["held_evaluation_opened"]:
        decision_text += (
            f"One held evaluation was opened once for `{decision['best_candidate']}`. "
            f"Paired lead-time WAPE difference: {held_gates['paired_ci_point_pp']:.2f} pp "
            f"(95% CI {held_gates['paired_ci_low_pp']:.2f} to "
            f"{held_gates['paired_ci_high_pp']:.2f}); origin wins "
            f"{held_gates['origin_wins']}/{held_gates['origin_count']}; "
            f"baseline beating {held_gates['baseline_beating']}; absolute bias "
            f"{held_gates['absolute_bias_pct']:.2f}%. Normal gates passed: "
            f"{held_gates['normal_gates_passed']}.\n\n"
            f"**Decision: `{decision['decision']}`.**\n"
        )
    else:
        decision_text += (
            "The held origins were not opened. The champion stays frozen and "
            "no held-forecast look was added to the ledger.\n\n"
            f"**Decision: `{decision['decision']}`.**\n"
        )
    held_section = (
        _markdown_table(
            held_table.loc[
                held_table["evaluation_level"].eq("lead_time"),
                [
                    "candidate_id", "wape_pct", "bias_pct", "mase", "mae",
                    "rmse", "mape_pct", "mape_nonzero_n",
                    "mape_zero_excluded",
                ],
            ],
            [
                "candidate_id", "wape_pct", "bias_pct", "mase", "mae",
                "rmse", "mape_pct", "mape_nonzero_n", "mape_zero_excluded",
            ],
        )
        if not held_table.empty
        else "No held evaluation was opened."
    )
    report = f"""# SPEC v4.2 A1 — model comparison at the tuning origins

Dataset: synthetic. Results validate method behaviour on simulated data; they are not factory savings.

## Scope and budget

- Tuning origins: 2024-07-01 and 2024-10-07; fixed 14-week horizon.
- Champion: `{CHAMPION_LABEL}` with the frozen hyperparameters and seed 42, read from the frozen tuning-forecast artifact.
- New candidates: `moving_average_13`, `exponential_smoothing` (ETS(A,N,N) with fitted alpha, bounded 0.01–0.99), and `xgboost` (one of XGBoost/Random Forest required by the amendment; Random Forest already had a Tier 2 run in Phase 3). XGBoost uses the same three seed-42 configurations as the frozen LightGBM search, mapped as: `num_leaves -> max_depth = ceil(log2(num_leaves))`, `min_child_samples -> min_child_weight`, `feature_fraction -> colsample_bytree`, same `n_estimators` and `learning_rate`. Both baseline candidates are refit at each origin; they do not use future labels.
- Reported references: `naive`, `seasonal_naive_52`, `tsb`, `historical_average`.
- Same features and recursive one-step strategy as the champion. No label after an origin is used.
- MAPE is computed on non-zero actuals only; the excluded-zero count is shown. MAPE is reported, not scored.
- SARIMA/ARIMA deferred: 616 mostly intermittent series, synthetic data and a 14-week horizon make per-series fits costly without decision value. LSTM deferred: no GPU, small synthetic sample, and recursive tabular challengers cover the non-linear comparison.

## Tuning results — weekly level

{_markdown_table(weekly_view, list(weekly_view.columns))}

## Tuning results — lead-time windows

{_markdown_table(lead_view, list(lead_view.columns))}

## Advancement gate

{decision_text}
## One held evaluation (only if the gate passed)

{held_section}

## Limits

- Two tuning origins only; no significance test is claimed from tuning alone.
- XGBoost is one seed and three configurations; its result is conditional on that budget.
- Lead-time WAPE sums a supplier's full window per part-region, so it is easier than weekly WAPE and not comparable across levels.
- The final test remains sealed; nothing here opens it.
"""
    (root / "results" / "model_comparison_report.md").write_text(
        report, encoding="utf-8"
    )


if __name__ == "__main__":
    print(json.dumps(run_model_comparison(), indent=2, default=str))
