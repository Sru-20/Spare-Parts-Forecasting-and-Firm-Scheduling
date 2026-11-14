"""Structural forecasting experiments selected on tuning origins only.

Candidates change the model form (origin-safe trend features, Poisson loss) or
combine existing candidates. Tuning origins decide which candidates advance;
the four held origins are scored once for those that pass. The final test
remains sealed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.data.loader import ROOT, load_demand, load_scheduling
from src.forecasting.backtest import (
    _decorate_forecasts,
    _fold_data,
    _lead_time_map,
    _origin_report_metadata,
)
from src.forecasting.bias_calibration import (
    _json_safe,
    evaluate_candidate,
)
from src.forecasting.global_models import fit_global_models
from src.forecasting.phase4_v41 import (
    _bootstrap_inputs,
    _lead_time_rows,
    _metric_values,
    _paired_wape_difference_ci,
    _pooled_wape,
)


SEED = 42
TUNING_WAPE_MARGIN = 0.005
MAX_ABS_BIAS = 0.10
BLEND_WEIGHTS = tuple(round(value, 2) for value in np.arange(0.0, 1.01, 0.05))
MODEL_CANDIDATES = {
    "tweedie_augmented": {
        "family": "lightgbm_tweedie",
        "config_id": "augmented",
        "extra_features": True,
        "description": "Tweedie plus origin-safe trend, zero-rate and history features",
    },
    "poisson": {
        "family": "lightgbm_poisson",
        "config_id": "base",
        "extra_features": False,
        "description": "LightGBM Poisson loss on the frozen feature set",
    },
    "poisson_augmented": {
        "family": "lightgbm_poisson",
        "config_id": "augmented",
        "extra_features": True,
        "description": "Poisson loss plus origin-safe trend, zero-rate and history features",
    },
}
BLEND_NAME = "blend_tweedie_quantile_p50"


def _read_config(root: Path) -> dict[str, Any]:
    with (root / "configs" / "project.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Project configuration must be a mapping")
    return config


def _tuning_parameters(root: Path, model: str, config_id: str) -> dict[str, float | int]:
    log = pd.read_csv(root / "results" / "track_a_tuning_log.csv")
    rows = log.loc[
        log["model"].eq(model)
        & log["config_id"].eq(config_id)
        & log["status"].eq("success")
    ]
    if rows.empty:
        raise ValueError(f"Parameters for {model}::{config_id} are absent")
    parameters = json.loads(str(rows.iloc[0]["parameters"]))
    return {
        name: float(value) if isinstance(value, float) else int(value)
        for name, value in parameters.items()
    }


def _metrics(frame: pd.DataFrame) -> dict[str, float]:
    values = _metric_values(_bootstrap_inputs(frame))
    return {
        "wape": float(values["wape_ratio"]),
        "bias": float(values["bias_ratio"]),
        "p95_coverage": float(values.get("p95_coverage", np.nan)),
    }


def _evaluate_forecasts(
    weekly: pd.DataFrame,
    extension_weeks: int,
) -> dict[str, float]:
    lead = _lead_time_rows(weekly, extension_weeks)
    weekly_metrics = _metrics(weekly)
    lead_metrics = _metrics(lead)
    return {
        "weekly_wape": weekly_metrics["wape"],
        "weekly_bias": weekly_metrics["bias"],
        "lead_time_wape": lead_metrics["wape"],
        "lead_time_bias": lead_metrics["bias"],
        "p95_coverage": lead_metrics["p95_coverage"],
        "lead_rows": int(len(lead)),
    }


def _fit_origin(
    demand: pd.DataFrame,
    lead_times: pd.DataFrame,
    origin: pd.Timestamp,
    family: str,
    parameters: dict[str, float | int],
    config_id: str,
    split: str,
    horizon: int,
    extra_features: bool,
) -> pd.DataFrame:
    history, targets, calendar = _fold_data(demand, origin, horizon)
    metadata, scales = _origin_report_metadata(history, targets)
    started = time.perf_counter()
    predictions, _, _ = fit_global_models(
        history, origin, calendar, parameters, family, SEED,
        extra_features=extra_features,
    )
    runtime = time.perf_counter() - started
    decorated = _decorate_forecasts(
        predictions, targets, metadata, scales, lead_times,
        family, config_id, split, origin, horizon, runtime,
    )
    return decorated.loc[decorated["actual_qty"].notna()].copy()


def _pair_blend(
    tweedie_weekly: pd.DataFrame,
    quantile_weekly: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["origin_date", "part_id", "region", "week_start_date"]
    quantile = quantile_weekly[keys + ["forecast_qty"]].rename(
        columns={"forecast_qty": "forecast_qty_quantile"}
    )
    paired = tweedie_weekly.merge(
        quantile, on=keys, how="inner", validate="one_to_one"
    )
    if paired["actual_qty"].isna().any():
        raise ValueError("Blend inputs must carry observed actuals")
    return paired


def _blend_weights(
    tweedie_weekly: pd.DataFrame,
    quantile_weekly: pd.DataFrame,
    extension_weeks: int,
) -> tuple[float, dict[str, float]]:
    paired = _pair_blend(tweedie_weekly, quantile_weekly)
    best_weight = 1.0
    best_metrics: dict[str, float] = {}
    for weight in BLEND_WEIGHTS:
        candidate = paired.copy()
        candidate["forecast_qty"] = (
            weight * candidate["forecast_qty"]
            + (1.0 - weight) * candidate["forecast_qty_quantile"]
        )
        metrics = _evaluate_forecasts(candidate, extension_weeks)
        if not best_metrics or metrics["lead_time_wape"] < best_metrics["lead_time_wape"]:
            best_weight = weight
            best_metrics = metrics
    if not best_metrics:
        raise ValueError("No blend weight could be scored")
    return best_weight, best_metrics


def _apply_blend(
    tweedie_weekly: pd.DataFrame,
    quantile_weekly: pd.DataFrame,
    weight: float,
) -> pd.DataFrame:
    paired = _pair_blend(tweedie_weekly, quantile_weekly)
    paired["forecast_qty"] = (
        weight * paired["forecast_qty"]
        + (1.0 - weight) * paired["forecast_qty_quantile"]
    )
    return paired


def run_model_experiments(root: Path = ROOT) -> dict[str, object]:
    started = time.perf_counter()
    config = _read_config(root)
    forecast_config = config["forecast"]
    origins = [pd.Timestamp(value) for value in forecast_config["backtest"]["origins"]]
    tuning_count = int(forecast_config["backtest"]["tuning_origin_count"])
    extension_weeks = int(
        forecast_config["backtest"]["over_horizon_extension_period_weeks"]
    )
    horizon = int(forecast_config["horizon_weeks"])
    tuning_origins = origins[:tuning_count]
    held_origins = origins[tuning_count:]
    if len(held_origins) != 4:
        raise ValueError(f"Expected four held origins, found {len(held_origins)}")

    demand = load_demand(root / "data" / "01_demand_forecasting.csv")
    scheduling = load_scheduling(root / "data" / "02_scheduling_inputs.csv")
    lead_times = _lead_time_map(scheduling)
    champion_parameters = _tuning_parameters(
        root, "lightgbm_tweedie", "random_1"
    )
    quantile_parameters = _tuning_parameters(
        root, "lightgbm_quantile", "random_1"
    )

    tuning_artifact = pd.read_csv(
        root / "artifacts" / "bias_calibration_tuning_forecasts.csv.gz",
        parse_dates=["week_start_date", "origin_date"],
    )
    tuning_tweedie = tuning_artifact.loc[
        tuning_artifact["model"].eq("lightgbm_tweedie")
        & tuning_artifact["config_id"].eq("random_1")
    ].copy()
    baseline_tuning = _evaluate_forecasts(tuning_tweedie, extension_weeks)

    tuning_rows: list[dict[str, object]] = [
        {
            "candidate": "lightgbm_tweedie::random_1 (frozen)",
            "stage": "baseline",
            **baseline_tuning,
            "advanced_to_held": True,
            "reason": "",
        }
    ]

    tuning_quantile_frames: list[pd.DataFrame] = []
    for origin in tuning_origins:
        tuning_quantile_frames.append(
            _fit_origin(
                demand, lead_times, origin, "lightgbm_quantile",
                quantile_parameters, "random_1", "tuning", horizon, False,
            )
        )
    tuning_quantile = pd.concat(tuning_quantile_frames, ignore_index=True)

    blend_weight, blend_tuning = _blend_weights(
        tuning_tweedie, tuning_quantile, extension_weeks
    )
    blend_gain = (
        baseline_tuning["lead_time_wape"] - blend_tuning["lead_time_wape"]
    )
    blend_advanced = blend_gain >= TUNING_WAPE_MARGIN
    tuning_rows.append(
        {
            "candidate": BLEND_NAME,
            "stage": "tuning",
            **blend_tuning,
            "advanced_to_held": blend_advanced,
            "reason": (
                f"weight={blend_weight}; tuning lead WAPE gain {blend_gain:.4f}"
                if blend_advanced
                else f"weight={blend_weight}; tuning lead WAPE gain "
                f"{blend_gain:.4f} below {TUNING_WAPE_MARGIN}"
            ),
        }
    )

    tuning_frames: dict[str, pd.DataFrame] = {}
    advanced: list[str] = [BLEND_NAME] if blend_advanced else []
    for name, candidate in MODEL_CANDIDATES.items():
        frames = []
        for origin in tuning_origins:
            frames.append(
                _fit_origin(
                    demand, lead_times, origin, candidate["family"],
                    champion_parameters, candidate["config_id"], "tuning",
                    horizon, bool(candidate["extra_features"]),
                )
            )
        weekly = pd.concat(frames, ignore_index=True)
        tuning_frames[name] = weekly
        metrics = _evaluate_forecasts(weekly, extension_weeks)
        gain = baseline_tuning["lead_time_wape"] - metrics["lead_time_wape"]
        reasons = []
        if gain < TUNING_WAPE_MARGIN:
            reasons.append(f"tuning lead WAPE gain {gain:.4f} below {TUNING_WAPE_MARGIN}")
        passed = not reasons
        if passed:
            advanced.append(name)
        tuning_rows.append(
            {
                "candidate": name,
                "stage": "tuning",
                **metrics,
                "advanced_to_held": passed,
                "reason": "; ".join(reasons),
            }
        )

    held_source = pd.read_parquet(
        root / "artifacts" / "track_a_backtest_forecasts_v41.parquet"
    )
    held_tweedie = held_source.loc[
        held_source["model"].eq("lightgbm_tweedie")
        & held_source["config_id"].eq("random_1")
        & held_source["actual_qty"].notna()
    ].copy()
    held_quantile = held_source.loc[
        held_source["model"].eq("lightgbm_quantile")
        & held_source["config_id"].eq("random_1")
        & held_source["actual_qty"].notna()
    ].copy()
    baseline_held = _evaluate_forecasts(held_tweedie, extension_weeks)

    held_rows: list[dict[str, object]] = [
        {
            "candidate": "lightgbm_tweedie::random_1 (frozen)",
            "stage": "baseline",
            **baseline_held,
            "tuning_gate_passed": True,
            "gate_passed": True,
            "admissible": True,
            "gate_reasons": "",
        }
    ]
    gate_baseline = {
        "lead_time_wape": baseline_held["lead_time_wape"],
        "lead_time_bias": baseline_held["lead_time_bias"],
        "p95_coverage": baseline_held["p95_coverage"],
    }

    blended_weekly = _apply_blend(held_tweedie, held_quantile, blend_weight)
    blend_held = _evaluate_forecasts(blended_weekly, extension_weeks)
    blend_lead = _lead_time_rows(blended_weekly, extension_weeks)
    baseline_lead = _lead_time_rows(held_tweedie, extension_weeks)
    delta_point, delta_low, delta_high = _paired_wape_difference_ci(
        blend_lead, baseline_lead, 2000, SEED, 0.95
    )
    blend_per_origin = {
        str(pd.Timestamp(origin).date()): float(_pooled_wape(group))
        for origin, group in blend_lead.groupby("origin_date", sort=True)
    }
    blend_gate = evaluate_candidate(gate_baseline, blend_held)
    held_rows.append(
        {
            "candidate": BLEND_NAME,
            "stage": "held",
            **blend_held,
            "tuning_gate_passed": blend_advanced,
            "gate_passed": blend_gate["passed"],
            "admissible": blend_advanced and blend_gate["passed"],
            "gate_reasons": "; ".join(blend_gate["reasons"]),
            "paired_delta_vs_champion": delta_point,
            "paired_delta_ci_low": delta_low,
            "paired_delta_ci_high": delta_high,
            "per_origin_lead_wape": json.dumps(blend_per_origin, sort_keys=True),
        }
    )

    for name in advanced:
        if name == BLEND_NAME:
            continue
        candidate = MODEL_CANDIDATES[name]
        frames = []
        for origin in held_origins:
            frames.append(
                _fit_origin(
                    demand, lead_times, origin, candidate["family"],
                    champion_parameters, candidate["config_id"], "held",
                    horizon, bool(candidate["extra_features"]),
                )
            )
        metrics = _evaluate_forecasts(
            pd.concat(frames, ignore_index=True), extension_weeks
        )
        gate = evaluate_candidate(gate_baseline, metrics)
        held_rows.append(
            {
                "candidate": name,
                "stage": "held",
                **metrics,
                "tuning_gate_passed": True,
                "gate_passed": gate["passed"],
                "admissible": gate["passed"],
                "gate_reasons": "; ".join(gate["reasons"]),
            }
        )

    accepted = [
        row["candidate"]
        for row in held_rows
        if row["stage"] == "held" and row["admissible"]
    ]
    summary = {
        "baseline_tuning": baseline_tuning,
        "baseline_held": baseline_held,
        "blend_weight": blend_weight,
        "tuning_rows": tuning_rows,
        "held_rows": held_rows,
        "accepted_candidates": accepted,
        "decision": accepted[0] if accepted else "keep_frozen_champion",
        "final_test_sealed": True,
        "runtime_seconds": time.perf_counter() - started,
    }

    results_dir = root / "results"
    pd.DataFrame(tuning_rows).to_csv(
        results_dir / "model_experiments_tuning.csv", index=False
    )
    pd.DataFrame(held_rows).to_csv(
        results_dir / "model_experiments_held.csv", index=False
    )
    with (results_dir / "model_experiments_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(_json_safe(summary), stream, indent=2, sort_keys=True)
    return summary


if __name__ == "__main__":
    print(
        json.dumps(
            _json_safe(run_model_experiments()), indent=2, sort_keys=True
        )
    )
