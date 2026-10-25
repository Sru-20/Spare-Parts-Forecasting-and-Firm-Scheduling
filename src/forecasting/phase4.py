"""Phase 4 Track A scoring, conditional Tier 2 evaluation, and inference."""
from __future__ import annotations

import itertools
import json
import math
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import t as student_t

from src.data.loader import ROOT, load_demand, load_scheduling
from src.features.time_features import CALENDAR_FEATURES, STATIC_FEATURES
from src.forecasting.backtest import (
    ANALOGUE_FALLBACK,
    GLOBAL_MODEL_FAMILIES,
    _decorate_forecasts,
    _fold_data,
    _lead_time_map,
    _lead_time_rows,
    _metric_rows,
    _origin_report_metadata,
    _pooled_model_scores,
    _run_global_fold,
)
from src.forecasting.baselines import BASELINE_MODELS, forecast_baseline
from src.forecasting.global_models import build_fold_training_data, fit_global_models
from src.forecasting.metrics import forecast_metrics


FORECAST_ARTIFACT = "artifacts/track_a_backtest_forecasts.parquet"
TIER2_MODEL = "random_forest"
COMPONENTS = (
    "wape",
    "absolute_bias",
    "a_class_critical_parts",
    "intermittent_lumpy",
    "calibration",
    "stability_across_origins",
    "runtime",
    "explainability",
)
def _read_config(root: Path) -> dict[str, Any]:
    with (root / "configs" / "project.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Project configuration must be a mapping")
    return config


def _metric_values(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "wape_pct": np.nan,
            "bias_pct": np.nan,
            "pinball_loss": np.nan,
            "n_scored": 0,
            **{
                name: np.nan
                for q in (0.90, 0.95, 0.99)
                for name in (
                    f"p{int(q * 100)}_coverage",
                    f"p{int(q * 100)}_coverage_error",
                )
            },
        }
    quantiles = {
        quantile: frame[f"forecast_p{int(quantile * 100)}"].to_numpy(dtype=float)
        for quantile in (0.90, 0.95, 0.99)
        if f"forecast_p{int(quantile * 100)}" in frame
        and frame[f"forecast_p{int(quantile * 100)}"].notna().any()
    }
    values = forecast_metrics(
        frame["actual_qty"].to_numpy(dtype=float),
        frame["forecast_qty"].to_numpy(dtype=float),
        quantile_forecasts=quantiles,
    )
    return {
        "wape_pct": float(values["wape_pct"]),
        "bias_pct": float(values["bias_pct"]),
        "pinball_loss": float(values["pinball_loss"]),
        "n_scored": int(values["n_scored"]),
        **{
            f"p{int(q * 100)}_coverage": float(
                values[f"p{int(q * 100)}_coverage"]
            )
            for q in (0.90, 0.95, 0.99)
        },
        **{
            f"p{int(q * 100)}_coverage_error": float(
                values[f"p{int(q * 100)}_coverage_error"]
            )
            for q in (0.90, 0.95, 0.99)
        },
    }


def _mixed_metric(
    weekly: pd.DataFrame,
    lead: pd.DataFrame,
    metric: str,
    lead_mix: float,
    weekly_mix: float,
) -> float:
    weekly_value = _metric_values(weekly).get(metric, np.nan)
    lead_value = _metric_values(lead).get(metric, np.nan)
    if not np.isfinite(weekly_value) or not np.isfinite(lead_value):
        return np.nan
    return float(lead_mix * lead_value + weekly_mix * weekly_value)


def _group_metric_rows(
    forecasts: pd.DataFrame,
    horizon: int,
    lead_mix: float,
    weekly_mix: float,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (model, config_id), group in forecasts.groupby(
        ["model", "config_id"], sort=True, observed=True
    ):
        weekly = group.loc[group["actual_qty"].notna()]
        lead = _lead_time_rows(group, horizon)
        weekly_values = _metric_values(weekly)
        lead_values = _metric_values(lead)
        record: dict[str, object] = {
            "model": str(model),
            "config_id": str(config_id),
            "candidate_id": f"{model}::{config_id}",
            "weekly_wape_pct": weekly_values["wape_pct"],
            "weekly_bias_pct": weekly_values["bias_pct"],
            "lead_time_wape_pct": lead_values["wape_pct"],
            "lead_time_bias_pct": lead_values["bias_pct"],
            "wape_mixed_pct": _mixed_metric(
                weekly, lead, "wape_pct", lead_mix, weekly_mix
            ),
            "absolute_bias_mixed_pct": (
                lead_mix * abs(lead_values["bias_pct"])
                + weekly_mix * abs(weekly_values["bias_pct"])
                if np.isfinite(lead_values["bias_pct"])
                and np.isfinite(weekly_values["bias_pct"])
                else np.nan
            ),
            "n_weekly": weekly_values["n_scored"],
            "n_lead_time": lead_values["n_scored"],
            "n_series_weekly": int(
                weekly.loc[weekly["forecast_qty"].notna(), ["part_id", "region"]]
                .drop_duplicates()
                .shape[0]
            ),
            "n_series_lead_time": int(lead[["part_id", "region"]].drop_duplicates().shape[0])
            if not lead.empty
            else 0,
        }

        segment_values: list[float] = []
        for column, value in (("abc_class", "A"), ("criticality", "High")):
            weekly_segment = weekly.loc[weekly[column].astype("string").eq(value)]
            lead_segment = lead.loc[lead[column].astype("string").eq(value)]
            segment_values.append(
                _mixed_metric(
                    weekly_segment,
                    lead_segment,
                    "wape_pct",
                    lead_mix,
                    weekly_mix,
                )
            )
        record["a_class_critical_parts_wape_pct"] = (
            float(np.mean(segment_values))
            if segment_values and all(np.isfinite(segment_values))
            else np.nan
        )

        intermittent_values: list[float] = []
        for value in ("Intermittent", "Lumpy"):
            weekly_segment = weekly.loc[weekly["sbc_class"].astype("string").eq(value)]
            lead_segment = lead.loc[lead["sbc_class"].astype("string").eq(value)]
            intermittent_values.append(
                _mixed_metric(
                    weekly_segment,
                    lead_segment,
                    "wape_pct",
                    lead_mix,
                    weekly_mix,
                )
            )
        record["intermittent_lumpy_wape_pct"] = (
            float(np.mean(intermittent_values))
            if intermittent_values and all(np.isfinite(intermittent_values))
            else np.nan
        )

        coverage_errors = [
            weekly_values.get(f"p{int(q * 100)}_coverage_error", np.nan)
            for q in (0.90, 0.95, 0.99)
        ]
        record["mean_coverage_error"] = (
            float(np.mean(coverage_errors))
            if coverage_errors and all(np.isfinite(coverage_errors))
            else np.nan
        )
        record["pinball_loss"] = weekly_values["pinball_loss"]
        lead_coverage_errors = [
            lead_values.get(f"p{int(q * 100)}_coverage_error", np.nan)
            for q in (0.90, 0.95, 0.99)
        ]
        record["lead_time_mean_coverage_error"] = (
            float(np.mean(lead_coverage_errors))
            if lead_coverage_errors and all(np.isfinite(lead_coverage_errors))
            else np.nan
        )
        record["lead_time_pinball_loss"] = lead_values["pinball_loss"]

        origin_wapes: list[float] = []
        for _, origin_group in group.groupby("origin_date", sort=True, observed=True):
            origin_lead = _lead_time_rows(origin_group, horizon)
            origin_metric = _metric_values(origin_lead)["wape_pct"]
            if np.isfinite(origin_metric):
                origin_wapes.append(origin_metric)
        record["origin_lead_time_wape_std_pct"] = (
            float(np.std(origin_wapes, ddof=0)) if origin_wapes else np.nan
        )
        record["origin_count"] = len(origin_wapes)
        record["runtime_seconds"] = float(
            group[["origin_date", "runtime_seconds"]]
            .drop_duplicates("origin_date")["runtime_seconds"]
            .sum()
        )
        records.append(record)
    return pd.DataFrame(records)


def _rank_score(values: pd.Series, lower_is_better: bool) -> pd.Series:
    finite = values.notna() & np.isfinite(values.astype(float))
    result = pd.Series(0.0, index=values.index)
    count = int(finite.sum())
    if count == 1:
        result.loc[finite] = 50.0
    elif count > 1:
        ranks = values.loc[finite].rank(
            method="average", ascending=lower_is_better
        )
        result.loc[finite] = 100.0 * (count - ranks) / (count - 1)
    return result


def _candidate_tiers(model: str, tier_config: dict[str, int]) -> tuple[int, int]:
    if model in BASELINE_MODELS:
        explanation = (
            "intermittent_baseline"
            if model in {"croston", "sba", "tsb"}
            else "formula_baseline"
        )
        return 1, int(tier_config[explanation])
    if model in {"simple_average_ensemble", "sbc_segment_router"}:
        return 3, int(tier_config["ensemble_or_router"])
    return 2, int(tier_config["global_tree_model"])


def _weighted_selection(
    metric_rows: pd.DataFrame,
    weights: dict[str, float],
    tier_config: dict[str, int],
    lead_calibration_share: float,
    weekly_calibration_share: float,
) -> pd.DataFrame:
    frame = metric_rows.copy()
    raw_directions = {
        "wape": ("wape_mixed_pct", True),
        "absolute_bias": ("absolute_bias_mixed_pct", True),
        "a_class_critical_parts": ("a_class_critical_parts_wape_pct", True),
        "intermittent_lumpy": ("intermittent_lumpy_wape_pct", True),
        "stability_across_origins": ("origin_lead_time_wape_std_pct", True),
        "runtime": ("runtime_seconds", True),
        "explainability": ("explainability_tier", False),
    }
    frame["explainability_tier"] = [
        _candidate_tiers(str(model), tier_config)[1] for model in frame["model"]
    ]
    frame["simplicity_tier"] = [
        _candidate_tiers(str(model), tier_config)[0] for model in frame["model"]
    ]
    score_columns: dict[str, str] = {}
    for component, (raw_column, lower_is_better) in raw_directions.items():
        score_column = f"{component}_score"
        frame[score_column] = _rank_score(
            frame[raw_column], lower_is_better=lower_is_better
        )
        score_columns[component] = score_column

    calibration_scores: dict[str, pd.Series] = {}
    for level, coverage_column, pinball_column in (
        ("weekly", "mean_coverage_error", "pinball_loss"),
        (
            "lead_time",
            "lead_time_mean_coverage_error",
            "lead_time_pinball_loss",
        ),
    ):
        coverage_score = _rank_score(
            frame[coverage_column], lower_is_better=True
        )
        pinball_score = _rank_score(frame[pinball_column], lower_is_better=True)
        complete = frame[coverage_column].notna() & frame[pinball_column].notna()
        calibration_scores[level] = pd.Series(
            np.where(complete, (coverage_score + pinball_score) / 2.0, 0.0),
            index=frame.index,
        )
    frame["weekly_calibration_score"] = calibration_scores["weekly"]
    frame["lead_time_calibration_score"] = calibration_scores["lead_time"]
    frame["calibration_score"] = (
        lead_calibration_share * frame["lead_time_calibration_score"]
        + weekly_calibration_share * frame["weekly_calibration_score"]
    )
    score_columns["calibration"] = "calibration_score"

    for component, score_column in score_columns.items():
        if component not in weights:
            raise ValueError(f"No configured selection weight for {component}")
    weight_sum = float(sum(weights.values()))
    if not math.isclose(weight_sum, 1.0, abs_tol=1e-9):
        raise ValueError(f"Track A weights must sum to 1.0, found {weight_sum}")
    frame["selection_score"] = sum(
        frame[score_columns[component]] * weight
        for component, weight in weights.items()
    )
    frame = frame.sort_values(
        [
            "selection_score",
            "simplicity_tier",
            "explainability_tier",
            "runtime_seconds",
            "candidate_id",
        ],
        ascending=[False, True, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    frame["selection_rank"] = np.arange(1, len(frame) + 1)
    return frame


def _sensitivity(
    ranked: pd.DataFrame,
    weights: dict[str, float],
    tier_config: dict[str, int],
    perturbation: float,
) -> pd.DataFrame:
    component_scores = {
        "wape": "wape_score",
        "absolute_bias": "absolute_bias_score",
        "a_class_critical_parts": "a_class_critical_parts_score",
        "intermittent_lumpy": "intermittent_lumpy_score",
        "calibration": "calibration_score",
        "stability_across_origins": "stability_across_origins_score",
        "runtime": "runtime_score",
        "explainability": "explainability_score",
    }
    default_winner = str(ranked.iloc[0]["candidate_id"])
    results: list[dict[str, object]] = []
    for component in COMPONENTS:
        for direction, amount in (("plus", perturbation), ("minus", -perturbation)):
            adjusted = dict(weights)
            adjusted[component] = max(0.0, adjusted[component] + amount)
            total = sum(adjusted.values())
            adjusted = {key: value / total for key, value in adjusted.items()}
            scores = sum(
                ranked[component_scores[name]] * weight
                for name, weight in adjusted.items()
            )
            ordering = ranked.assign(sensitivity_score=scores).sort_values(
                [
                    "sensitivity_score",
                    "simplicity_tier",
                    "explainability_tier",
                    "runtime_seconds",
                    "candidate_id",
                ],
                ascending=[False, True, False, True, True],
                kind="stable",
            )
            results.append(
                {
                    "perturbed_component": component,
                    "direction": direction,
                    "perturbation_points": amount * 100.0,
                    "winner": str(ordering.iloc[0]["candidate_id"]),
                    "default_winner": default_winner,
                    "winner_changed": str(ordering.iloc[0]["candidate_id"])
                    != default_winner,
                    "top_three": " | ".join(ordering["candidate_id"].head(3)),
                    **{
                        f"weight_{key}": value
                        for key, value in adjusted.items()
                    },
                }
            )
    return pd.DataFrame(results)


def _weekly_pair_rows(
    forecasts: pd.DataFrame,
    model_a: str,
    config_a: str,
    model_b: str,
    config_b: str,
) -> pd.DataFrame:
    keys = ["part_id", "region", "week_start_date", "origin_date"]
    columns = [*keys, "actual_qty", "forecast_qty"]
    left = forecasts.loc[
        forecasts["model"].eq(model_a) & forecasts["config_id"].eq(config_a),
        columns,
    ].rename(columns={"forecast_qty": "forecast_a", "actual_qty": "actual_a"})
    right = forecasts.loc[
        forecasts["model"].eq(model_b) & forecasts["config_id"].eq(config_b),
        columns,
    ].rename(columns={"forecast_qty": "forecast_b", "actual_qty": "actual_b"})
    paired = left.merge(right, on=keys, how="inner", validate="one_to_one")
    paired["actual_qty"] = paired["actual_a"]
    valid = (
        paired["actual_a"].notna()
        & paired["actual_b"].notna()
        & paired["forecast_a"].notna()
        & paired["forecast_b"].notna()
    )
    return paired.loc[valid].copy()


def _weekly_dm(pair: pd.DataFrame, horizon: int, max_lag: int) -> dict[str, float | int]:
    if pair.empty:
        return {
            "dm_statistic": np.nan,
            "dm_p_value": np.nan,
            "dm_periods": 0,
            "dm_mean_wape_difference_pct_points": np.nan,
        }
    weekly_losses: list[dict[str, float]] = []
    for _, group in pair.groupby("week_start_date", sort=True):
        actual = group["actual_qty"].to_numpy(dtype=float)
        denominator = float(np.sum(actual))
        if denominator <= 0:
            continue
        loss_a = 100.0 * float(
            np.sum(np.abs(group["forecast_a"].to_numpy(dtype=float) - actual))
        ) / denominator
        loss_b = 100.0 * float(
            np.sum(np.abs(group["forecast_b"].to_numpy(dtype=float) - actual))
        ) / denominator
        weekly_losses.append({"a": loss_a, "b": loss_b})
    losses = pd.DataFrame(weekly_losses)
    difference = (losses["b"] - losses["a"]).to_numpy(dtype=float)
    count = len(difference)
    if count < 3:
        return {
            "dm_statistic": np.nan,
            "dm_p_value": np.nan,
            "dm_periods": count,
            "dm_mean_wape_difference_pct_points": float(np.mean(difference))
            if count
            else np.nan,
        }
    centered = difference - np.mean(difference)
    gamma_zero = float(np.dot(centered, centered) / count)
    long_run_variance = gamma_zero
    effective_lag = min(max_lag, count - 1)
    for lag in range(1, effective_lag + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / count)
        bartlett_weight = 1.0 - lag / (effective_lag + 1.0)
        long_run_variance += 2.0 * bartlett_weight * covariance
    if long_run_variance <= 0 or not np.isfinite(long_run_variance):
        statistic = np.nan
        p_value = np.nan
    else:
        uncorrected = float(np.mean(difference) / math.sqrt(long_run_variance / count))
        hln_squared = (
            count + 1.0 - 2.0 * horizon + horizon * (horizon - 1.0) / count
        ) / count
        statistic = uncorrected * math.sqrt(max(0.0, hln_squared))
        p_value = float(2.0 * student_t.sf(abs(statistic), df=count - 1))
    return {
        "dm_statistic": statistic,
        "dm_p_value": p_value,
        "dm_periods": count,
        "dm_mean_wape_difference_pct_points": float(np.mean(difference)),
    }


def _bootstrap_wape_difference(
    left: pd.DataFrame,
    right: pd.DataFrame,
    cluster_columns: list[str],
    replicates: int,
    seed: int,
) -> tuple[float, float, float, int]:
    keys = [*cluster_columns, "origin_date"]
    if "week_start_date" in left and "week_start_date" in right:
        keys.append("week_start_date")
    paired = left.merge(
        right,
        on=keys,
        how="inner",
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    paired = paired.loc[
        paired["actual_qty_a"].notna()
        & paired["actual_qty_b"].notna()
        & paired["forecast_qty_a"].notna()
        & paired["forecast_qty_b"].notna()
    ].copy()
    if paired.empty:
        return np.nan, np.nan, np.nan, 0
    paired["abs_error_a"] = (
        paired["forecast_qty_a"] - paired["actual_qty_a"]
    ).abs()
    paired["abs_error_b"] = (
        paired["forecast_qty_b"] - paired["actual_qty_b"]
    ).abs()
    sums = paired.groupby(cluster_columns, sort=False, observed=True).agg(
        abs_error_a=("abs_error_a", "sum"),
        abs_error_b=("abs_error_b", "sum"),
        actual=("actual_qty_a", "sum"),
    )
    sums = sums.loc[sums["actual"].gt(0)]
    cluster_count = len(sums)
    if not cluster_count:
        return np.nan, np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, cluster_count, size=(replicates, cluster_count))
    a_error = sums["abs_error_a"].to_numpy(dtype=float)[draws].sum(axis=1)
    b_error = sums["abs_error_b"].to_numpy(dtype=float)[draws].sum(axis=1)
    actual = sums["actual"].to_numpy(dtype=float)[draws].sum(axis=1)
    differences = np.divide(
        100.0 * (b_error - a_error),
        actual,
        out=np.full(replicates, np.nan),
        where=actual > 0,
    )
    estimate = float(
        100.0
        * (
            sums["abs_error_b"].sum() - sums["abs_error_a"].sum()
        )
        / sums["actual"].sum()
    )
    lower, upper = np.nanquantile(differences, [0.025, 0.975])
    return estimate, float(lower), float(upper), cluster_count


def _statistical_comparisons(
    ranked: pd.DataFrame,
    forecasts: pd.DataFrame,
    horizon: int,
    statistics_config: dict[str, Any],
    root_seed: int,
) -> pd.DataFrame:
    finalists = ranked.head(int(statistics_config["finalist_count"]))
    pairs: list[dict[str, object]] = []
    for pair_number, (candidate_a, candidate_b) in enumerate(
        itertools.combinations(finalists.to_dict("records"), 2)
    ):
        paired_weekly = _weekly_pair_rows(
            forecasts,
            str(candidate_a["model"]),
            str(candidate_a["config_id"]),
            str(candidate_b["model"]),
            str(candidate_b["config_id"]),
        )
        dm = _weekly_dm(
            paired_weekly,
            horizon=horizon,
            max_lag=int(statistics_config["dm_hac_max_lag_weeks"]),
        )
        left_weekly = forecasts.loc[
            forecasts["model"].eq(candidate_a["model"])
            & forecasts["config_id"].eq(candidate_a["config_id"])
            & forecasts["actual_qty"].notna()
        ]
        right_weekly = forecasts.loc[
            forecasts["model"].eq(candidate_b["model"])
            & forecasts["config_id"].eq(candidate_b["config_id"])
            & forecasts["actual_qty"].notna()
        ]
        weekly = _bootstrap_wape_difference(
            left_weekly,
            right_weekly,
            ["part_id", "region"],
            int(statistics_config["bootstrap_replicates"]),
            root_seed + pair_number,
        )
        left_lead = _lead_time_rows(left_weekly, horizon)
        right_lead = _lead_time_rows(right_weekly, horizon)
        left_lead = left_lead.rename(
            columns={"actual_qty": "actual_qty_a", "forecast_qty": "forecast_qty_a"}
        )
        right_lead = right_lead.rename(
            columns={"actual_qty": "actual_qty_b", "forecast_qty": "forecast_qty_b"}
        )
        lead = _bootstrap_wape_difference(
            left_lead,
            right_lead,
            ["part_id", "region"],
            int(statistics_config["bootstrap_replicates"]),
            root_seed + pair_number + 1000,
        )
        pairs.append(
            {
                "candidate_a": candidate_a["candidate_id"],
                "candidate_b": candidate_b["candidate_id"],
                "difference_direction": "B minus A; positive means B has higher WAPE",
                **dm,
                "weekly_wape_difference_pct_points": weekly[0],
                "weekly_bootstrap_ci_lower_pct_points": weekly[1],
                "weekly_bootstrap_ci_upper_pct_points": weekly[2],
                "weekly_bootstrap_series_clusters": weekly[3],
                "lead_time_wape_difference_pct_points": lead[0],
                "lead_time_bootstrap_ci_lower_pct_points": lead[1],
                "lead_time_bootstrap_ci_upper_pct_points": lead[2],
                "lead_time_bootstrap_series_clusters": lead[3],
            }
        )
    result = pd.DataFrame(pairs)
    if result.empty:
        return result
    finite = result["dm_p_value"].notna()
    adjusted = pd.Series(np.nan, index=result.index, dtype=float)
    sorted_indices = result.loc[finite, "dm_p_value"].sort_values().index
    running = 0.0
    total = len(sorted_indices)
    for rank, index in enumerate(sorted_indices):
        running = max(running, min(1.0, (total - rank) * result.at[index, "dm_p_value"]))
        adjusted.at[index] = running
    result["dm_holm_adjusted_p_value"] = adjusted
    return result


def _phase3_correction(
    forecasts: pd.DataFrame,
    root: Path,
    horizon: int,
) -> pd.DataFrame:
    old_path = root / "results" / "track_a_candidates.csv"
    if not old_path.is_file():
        return pd.DataFrame()
    old = pd.read_csv(old_path)
    old = old.loc[
        old["evaluation_split"].eq("evaluation")
        & old["pooled_origins"].fillna(False)
        & old["evaluation_level"].eq("lead_time")
        & old["segment_dimension"].eq("overall")
        & old["horizon_step"].eq("all")
    ]
    new = _metric_rows(forecasts, horizon)
    new = new.loc[
        new["evaluation_split"].eq("evaluation")
        & new["pooled_origins"].fillna(False)
        & new["evaluation_level"].eq("lead_time")
        & new["segment_dimension"].eq("overall")
        & new["horizon_step"].eq("all")
    ]
    if old.empty or new.empty:
        return pd.DataFrame()
    joined = old.merge(
        new,
        on=["model", "config_id"],
        how="inner",
        suffixes=("_phase3", "_corrected"),
        validate="one_to_one",
    )
    columns = [
        "model",
        "config_id",
        "wape_pct_phase3",
        "wape_pct_corrected",
        "n_scored_phase3",
        "n_scored_corrected",
    ]
    result = joined.loc[:, columns].copy()
    result["reason"] = (
        "Phase 3 lead-time aggregation treated missing forecasts as zero for "
        "series with observed actuals; Phase 4 excludes incomplete lead windows."
    )
    return result


def _run_tier2(
    root: Path,
    config: dict[str, Any],
    demand: pd.DataFrame,
    scheduling: pd.DataFrame,
    horizon: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str | None]:
    settings = config["phase4"]["tier2_random_forest"]
    lead_times = _lead_time_map(scheduling)
    tuning_forecasts: list[pd.DataFrame] = []
    evaluation_forecasts: list[pd.DataFrame] = []
    log_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    configurations = settings["configurations"]

    for origin in map(pd.Timestamp, settings["tuning_origins"]):
        history, targets, calendar = _fold_data(demand, origin, horizon)
        metadata, scales = _origin_report_metadata(history, targets)
        training_design = build_fold_training_data(history, origin)
        for index, parameters in enumerate(configurations, start=1):
            config_id = f"tier2_random_{index}"
            started = time.perf_counter()
            try:
                predictions, _, training_rows = _run_global_fold(
                    history,
                    calendar,
                    origin,
                    TIER2_MODEL,
                    parameters,
                    seed,
                    training_design=training_design,
                )
                runtime = time.perf_counter() - started
                tuning_forecasts.append(
                    _decorate_forecasts(
                        predictions,
                        targets,
                        metadata,
                        scales,
                        lead_times,
                        TIER2_MODEL,
                        config_id,
                        "tuning",
                        origin,
                        horizon,
                        runtime,
                    )
                )
                log_rows.append(
                    {
                        "model": TIER2_MODEL,
                        "config_id": config_id,
                        "origin_date": origin,
                        "status": "success",
                        "parameters": json.dumps(parameters, sort_keys=True),
                        "training_rows": training_rows,
                        "runtime_seconds": runtime,
                    }
                )
            except (ImportError, ValueError, RuntimeError, TypeError) as error:
                runtime = time.perf_counter() - started
                log_rows.append(
                    {
                        "model": TIER2_MODEL,
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
                failures.append(
                    {
                        "model": TIER2_MODEL,
                        "config_id": config_id,
                        "evaluation_split": "tuning",
                        "origin_date": origin,
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )

    tuning_frame = (
        pd.concat(tuning_forecasts, ignore_index=True)
        if tuning_forecasts
        else pd.DataFrame()
    )
    selected_config_id: str | None = None
    selected_parameters: dict[str, Any] | None = None
    if not tuning_frame.empty:
        tuning_scores = _pooled_model_scores(tuning_frame, horizon)
        tuning_scores = tuning_scores.sort_values(
            ["lead_time_wape_pct", "weekly_wape_pct", "config_id"],
            na_position="last",
            kind="stable",
        )
        if not tuning_scores.empty:
            selected_config_id = str(tuning_scores.iloc[0]["config_id"])
            index = int(selected_config_id.rsplit("_", 1)[1]) - 1
            selected_parameters = dict(configurations[index])
    for row in log_rows:
        row["selected_for_held_validation"] = (
            row["config_id"] == selected_config_id
        )
    log = pd.DataFrame(log_rows)

    if selected_config_id is not None and selected_parameters is not None:
        for origin in map(pd.Timestamp, settings["held_validation_origins"]):
            history, targets, calendar = _fold_data(demand, origin, horizon)
            metadata, scales = _origin_report_metadata(history, targets)
            training_design = build_fold_training_data(history, origin)
            started = time.perf_counter()
            try:
                predictions, _, _ = _run_global_fold(
                    history,
                    calendar,
                    origin,
                    TIER2_MODEL,
                    selected_parameters,
                    seed,
                    training_design=training_design,
                )
                runtime = time.perf_counter() - started
                evaluation_forecasts.append(
                    _decorate_forecasts(
                        predictions,
                        targets,
                        metadata,
                        scales,
                        lead_times,
                        TIER2_MODEL,
                        selected_config_id,
                        "evaluation",
                        origin,
                        horizon,
                        runtime,
                    )
                )
            except (ImportError, ValueError, RuntimeError, TypeError) as error:
                failures.append(
                    {
                        "model": TIER2_MODEL,
                        "config_id": selected_config_id,
                        "evaluation_split": "evaluation",
                        "origin_date": origin,
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )
    evaluation_frame = (
        pd.concat(evaluation_forecasts, ignore_index=True)
        if evaluation_forecasts
        else pd.DataFrame()
    )
    evaluation_failures = [
        failure for failure in failures if failure.get("evaluation_split") == "evaluation"
    ]
    if evaluation_failures:
        log = pd.concat([log, pd.DataFrame(evaluation_failures)], ignore_index=True)
    return tuning_frame, evaluation_frame, log, selected_config_id


def _future_calendar(demand: pd.DataFrame, training_end: pd.Timestamp) -> pd.DataFrame:
    allowed = demand.loc[
        demand["split_role"].isin(["final_test", "future"])
        & demand["week_start_date"].gt(training_end)
    ]
    columns = list(
        dict.fromkeys(
            column
            for column in (
                "part_id",
                "region",
                "week_start_date",
                *STATIC_FEATURES,
                *CALENDAR_FEATURES,
            )
            if column in allowed.columns
        )
    )
    calendar = allowed.loc[:, columns].copy()
    if calendar.duplicated(["part_id", "region", "week_start_date"]).any():
        raise ValueError("Future feature calendar contains duplicate series-week keys")
    return calendar.sort_values(
        ["week_start_date", "part_id", "region"], kind="stable"
    ).reset_index(drop=True)


def _forecast_final_model(
    root: Path,
    config: dict[str, Any],
    demand: pd.DataFrame,
    selected: pd.Series,
    tier2_config_id: str | None,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    training_end = pd.Timestamp(config["forecast"]["split_dates"]["train_validation_end"])
    history = demand.loc[
        demand["split_role"].isin(["warmup", "train_val"])
        & demand["week_start_date"].le(training_end)
    ].copy()
    calendar = _future_calendar(demand, training_end)
    future_keys = set(
        pd.to_datetime(
            demand.loc[demand["split_role"].eq("future"), "week_start_date"]
        ).unique()
    )
    model_name = str(selected["model"])
    config_id = str(selected["config_id"])
    seed_model = {
        "model": model_name,
        "config_id": config_id,
        "random_seed": seed,
        "training_end": training_end.date().isoformat(),
        "training_data_roles": ["warmup_for_history_only", "train_val_for_fit"],
        "future_output_role": "future",
        "test_labels_used": False,
        "quantile_output": False,
    }

    if model_name in BASELINE_MODELS:
        records: list[pd.DataFrame] = []
        for (part_id, region), group in history.groupby(
            ["part_id", "region"], sort=False, observed=True
        ):
            target_calendar = calendar.loc[
                calendar["part_id"].eq(part_id) & calendar["region"].eq(region)
            ]
            target_dates = pd.to_datetime(target_calendar["week_start_date"])
            predictions = forecast_baseline(group, target_dates, model_name)
            records.append(
                pd.DataFrame(
                    {
                        "part_id": part_id,
                        "region": region,
                        "week_start_date": target_dates,
                        "forecast_qty": predictions,
                        "forecast_p90": np.nan,
                        "forecast_p95": np.nan,
                        "forecast_p99": np.nan,
                    }
                )
            )
        forecast = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
        seed_model["forecast_parameters"] = "baseline default"
        return forecast.loc[
            pd.to_datetime(forecast["week_start_date"]).isin(future_keys)
        ].copy(), seed_model

    if model_name in GLOBAL_MODEL_FAMILIES:
        tuning = pd.read_csv(root / "results" / "track_a_tuning_log.csv")
        selected_rows = tuning.loc[
            tuning["model"].eq(model_name)
            & tuning["config_id"].eq(config_id)
            & tuning["selected_for_held_validation"].fillna(False).astype(bool)
        ]
        if selected_rows.empty:
            raise ValueError(f"Frozen tuning parameters missing for {model_name}:{config_id}")
        parameters = json.loads(str(selected_rows.iloc[0]["parameters"]))
    elif model_name == TIER2_MODEL:
        if tier2_config_id != config_id:
            raise ValueError("Selected Random Forest configuration is not the tuned Tier 2 model")
        tier2_settings = config["phase4"]["tier2_random_forest"]
        index = int(config_id.rsplit("_", 1)[1]) - 1
        parameters = dict(tier2_settings["configurations"][index])
    else:
        raise ValueError(
            f"Future model freezing for composite candidate {model_name!r} "
            "requires a component-wise freeze; no substitute was generated."
        )

    predictions, models, training_rows = fit_global_models(
        history,
        training_end,
        calendar,
        parameters,
        model_name if model_name in GLOBAL_MODEL_FAMILIES else TIER2_MODEL,
        seed,
    )
    predictions = predictions.loc[
        pd.to_datetime(predictions["week_start_date"]).isin(future_keys)
    ].copy()
    seed_model.update(
        {
            "parameters": parameters,
            "training_rows": training_rows,
            "quantile_output": model_name == "lightgbm_quantile",
            "fit_model_family": (
                model_name if model_name in GLOBAL_MODEL_FAMILIES else TIER2_MODEL
            ),
            "models": models,
        }
    )
    return predictions, seed_model


def _write_reports(
    root: Path,
    selection: pd.DataFrame,
    sensitivity: pd.DataFrame,
    comparisons: pd.DataFrame,
    calibration: pd.DataFrame,
    correction: pd.DataFrame,
    tier2_log: pd.DataFrame,
    stable: bool,
    model_artifact: str | None,
    future_count: int,
    blocker: str | None,
) -> None:
    reports_dir = root / "results"
    tier2_success = (
        int(tier2_log["status"].eq("success").sum())
        if not tier2_log.empty and "status" in tier2_log
        else 0
    )
    lines = [
        "# Phase 4 — Track A selection",
        "",
        "This evaluation uses simulated data and train-validation evidence only. "
        "The final-test labels remain sealed.",
        "",
        f"- Tier 2 Random Forest fitting rows logged successfully: {tier2_success}.",
        f"- Sensitivity-stable winner: {'yes' if stable else 'no'}.",
        f"- Frozen model artifact: {model_artifact or 'not created'}.",
        f"- Future output rows saved: {future_count}.",
        f"- Phase 3 lead-time correction rows: {len(correction)}.",
        "",
        "## Selection",
        "",
        "| Rank | Candidate | Score | Lead-time WAPE % | Weekly WAPE % | Calibration score |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for _, row in selection.head(10).iterrows():
        lines.append(
            f"| {int(row['selection_rank'])} | {row['candidate_id']} | "
            f"{row['selection_score']:.3f} | {row['lead_time_wape_pct']:.3f} | "
            f"{row['weekly_wape_pct']:.3f} | {row['calibration_score']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Conditional Tier 2 result",
            "",
        ]
    )
    tier2 = selection.loc[selection["model"].eq(TIER2_MODEL)]
    if not tier2.empty:
        forest = tier2.iloc[0]
        lines.append(
            f"Random Forest ranked {int(forest['selection_rank'])} of "
            f"{len(selection)} candidates (selection score "
            f"{forest['selection_score']:.3f}; lead-time WAPE "
            f"{forest['lead_time_wape_pct']:.3f}%; weekly WAPE "
            f"{forest['weekly_wape_pct']:.3f}%; mixed intermittent/lumpy WAPE "
            f"{forest['intermittent_lumpy_wape_pct']:.3f}%; runtime "
            f"{forest['runtime_seconds']:.1f} seconds). It did not improve on the "
            "best Tier 1 intermittent/lumpy result."
        )
    else:
        lines.append("No Tier 2 Random Forest configuration produced eligible metrics.")
    lines.extend(
        [
            "",
            "## Weight sensitivity",
            "",
            f"Winner changed in {int(sensitivity['winner_changed'].sum())} of "
            f"{len(sensitivity)} one-at-a-time ±10-point perturbations.",
            "",
            "## Statistical comparisons",
            "",
            f"Pairwise comparisons among the top five produced {len(comparisons)} rows. "
            "DM uses weekly WAPE by target week, Bartlett HAC through lag 13, "
            "and the 14-week HLN correction. Bootstrap intervals resample part-region "
            "clusters with 2,000 paired replicates.",
            "",
            "| Candidate A | Candidate B | DM mean WAPE diff. (B−A) | Holm p | Weekly WAPE diff. 95% CI | Lead-time WAPE diff. 95% CI |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in comparisons.iterrows():
        lines.append(
            f"| {row['candidate_a']} | {row['candidate_b']} | "
            f"{row['dm_mean_wape_difference_pct_points']:.3f} | "
            f"{row['dm_holm_adjusted_p_value']:.4g} | "
            f"[{row['weekly_bootstrap_ci_lower_pct_points']:.3f}, "
            f"{row['weekly_bootstrap_ci_upper_pct_points']:.3f}] | "
            f"[{row['lead_time_bootstrap_ci_lower_pct_points']:.3f}, "
            f"{row['lead_time_bootstrap_ci_upper_pct_points']:.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Calibration",
            "",
            "| Candidate | Weekly P90/P95/P99 | Weekly coverage error | Weekly pinball | Lead-time P90/P95/P99 | Lead-time coverage error | Lead-time pinball |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in calibration.iterrows():
        def fmt(value: object) -> str:
            return f"{float(value):.4f}" if pd.notna(value) else "n/a"

        lines.append(
            f"| {row['candidate_id']} | "
            f"{fmt(row.get('weekly_p90_coverage'))} / "
            f"{fmt(row.get('weekly_p95_coverage'))} / "
            f"{fmt(row.get('weekly_p99_coverage'))} | "
            f"{fmt(row.get('mean_coverage_error'))} | "
            f"{fmt(row.get('pinball_loss'))} | "
            f"{fmt(row.get('lead_time_p90_coverage'))} / "
            f"{fmt(row.get('lead_time_p95_coverage'))} / "
            f"{fmt(row.get('lead_time_p99_coverage'))} | "
            f"{fmt(row.get('lead_time_mean_coverage_error'))} | "
            f"{fmt(row.get('lead_time_pinball_loss'))} |"
        )
    if blocker:
        lines.extend(["", "## Exit-gate blocker", "", blocker])
    lines.extend(
        [
            "",
            "Tier 3 is deferred: the source is synthetic, only four held origins are "
            "available, and more complex models have no demonstrated decision value "
            "on this validation sample.",
            "",
            "These results are method validation on simulated data, not evidence of "
            "factory savings. Four held origins and dependent part-region observations "
            "limit inferential power.",
        ]
    )
    (reports_dir / "phase4_track_a_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    calibration_lines = [
        "# Track A calibration report",
        "",
        "Calibration is scored on mean absolute P90/P95/P99 coverage error and "
        "mean pinball loss. A candidate without all required quantile outputs receives "
        "the minimum calibration component score. The calibration component combines "
        "lead-time score at 70% and weekly score at 30%.",
        "",
        "| Candidate | Weekly P90/P95/P99 coverage | Weekly coverage error | Weekly pinball | Lead-time P90/P95/P99 coverage | Lead-time coverage error | Lead-time pinball | Calibration score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in calibration.iterrows():
        calibration_lines.append(
            f"| {row['candidate_id']} | "
            f"{row['weekly_p90_coverage']} / {row['weekly_p95_coverage']} / "
            f"{row['weekly_p99_coverage']} | {row['mean_coverage_error']} | "
            f"{row['pinball_loss']} | {row['lead_time_p90_coverage']} / "
            f"{row['lead_time_p95_coverage']} / {row['lead_time_p99_coverage']} | "
            f"{row['lead_time_mean_coverage_error']} | "
            f"{row['lead_time_pinball_loss']} | {row['calibration_score']} |"
        )
    (reports_dir / "track_a_calibration_report.md").write_text(
        "\n".join(calibration_lines) + "\n", encoding="utf-8"
    )
    if not correction.empty:
        phase3_report = reports_dir / "phase3_track_a_report.md"
        correction_note = (
            "\n\n## Phase 4 correction to lead-time aggregation\n\n"
            "The original lead-time aggregator could count missing baseline forecasts "
            "as zero. Phase 4 excludes incomplete lead-time windows and recomputes "
            "metrics from the saved held-validation forecasts. The corrected values "
            "supersede the earlier pooled lead-time rows; see "
            "`track_a_phase4_metric_correction.csv` for before/after values. The maximum "
            "absolute WAPE change was "
            f"{(correction['wape_pct_corrected'] - correction['wape_pct_phase3']).abs().max():.6f} "
            "percentage points.\n"
        )
        current_report = phase3_report.read_text(encoding="utf-8")
        if "## Phase 4 correction to lead-time aggregation" not in current_report:
            phase3_report.write_text(
                current_report.rstrip() + correction_note,
                encoding="utf-8",
            )


def run_phase4(root: Path = ROOT) -> dict[str, object]:
    """Run the frozen Track A selection on the four held-validation origins."""
    config = _read_config(root)
    forecast_config = config["forecast"]
    horizon = int(forecast_config["horizon_weeks"])
    seed = int(forecast_config["backtest"]["seed"])
    phase4_config = config["phase4"]
    training_end = pd.Timestamp(forecast_config["split_dates"]["train_validation_end"])

    demand = load_demand(root / "data" / "01_demand_forecasting.csv")
    demand = demand.loc[
        (
            demand["split_role"].isin(["warmup", "train_val"])
            & demand["week_start_date"].le(training_end)
        )
        | demand["split_role"].isin(["final_test", "future"])
    ].copy()
    scheduling = load_scheduling(root / "data" / "02_scheduling_inputs.csv")
    phase3 = pd.read_parquet(root / FORECAST_ARTIFACT)
    phase3 = phase3.loc[
        phase3["evaluation_split"].eq("evaluation")
        & phase3["candidate_role"].eq("forecast_candidate")
    ].copy()
    correction = _phase3_correction(phase3, root, horizon)
    correction.to_csv(root / "results" / "track_a_phase4_metric_correction.csv", index=False)

    tier2_forecasts_path = (
        root / "artifacts" / "track_a_phase4_validation_forecasts.parquet"
    )
    tier2_log_path = root / "results" / "track_a_tier2_tuning.csv"
    tier2_metrics_path = root / "results" / "track_a_tier2_candidates.csv"
    if (
        tier2_forecasts_path.is_file()
        and tier2_log_path.is_file()
        and tier2_metrics_path.is_file()
    ):
        tier2_tuning = pd.DataFrame()
        cached_validation = pd.read_parquet(tier2_forecasts_path)
        cached_validation["origin_date"] = pd.to_datetime(
            cached_validation["origin_date"]
        )
        tier2_evaluation = cached_validation.loc[
            cached_validation["model"].eq(TIER2_MODEL)
            & cached_validation["evaluation_split"].eq("evaluation")
        ].copy()
        tier2_log = pd.read_csv(tier2_log_path)
        selected_tier2 = tier2_log.loc[
            tier2_log["model"].eq(TIER2_MODEL)
            & tier2_log["selected_for_held_validation"]
            .astype("string")
            .str.lower()
            .eq("true")
        ]
        tier2_selected_id = (
            str(selected_tier2.iloc[0]["config_id"])
            if not selected_tier2.empty
            else None
        )
        tier2_metrics = pd.read_csv(tier2_metrics_path)
        tier2_cached = True
    else:
        tier2_tuning, tier2_evaluation, tier2_log, tier2_selected_id = _run_tier2(
            root,
            config,
            demand,
            scheduling,
            horizon,
            seed,
        )
        tier2_cached = False
    if tier2_evaluation.empty:
        combined = phase3.copy()
    else:
        combined = pd.concat(
            [phase3, tier2_evaluation], ignore_index=True, sort=False
        )
    combined["origin_date"] = pd.to_datetime(combined["origin_date"])
    artifact_frame = combined.copy()
    artifact_frame["origin_date"] = artifact_frame["origin_date"].dt.strftime(
        "%Y-%m-%d"
    )
    artifact_frame.to_parquet(
        root / "artifacts" / "track_a_phase4_validation_forecasts.parquet",
        index=False,
        engine="fastparquet",
    )
    tier2_log.to_csv(root / "results" / "track_a_tier2_tuning.csv", index=False)
    if not tier2_cached:
        tier2_metrics = (
            _metric_rows(
                pd.concat(
                    [tier2_tuning, tier2_evaluation],
                    ignore_index=True,
                    sort=False,
                ),
                horizon,
            )
            if not tier2_tuning.empty or not tier2_evaluation.empty
            else pd.DataFrame()
        )
        tier2_failures = tier2_log.loc[
            tier2_log["status"].eq("failed")
        ].copy()
        if not tier2_failures.empty:
            tier2_failures["candidate_role"] = "forecast_candidate"
            tier2_failures["evaluation_level"] = "failure"
            tier2_failures["segment_dimension"] = "overall"
            tier2_failures["segment_value"] = "all"
            tier2_metrics = pd.concat(
                [tier2_metrics, tier2_failures],
                ignore_index=True,
                sort=False,
            )
        tier2_metrics.to_csv(tier2_metrics_path, index=False)

    track_scoring = config["scoring"]["tracks"]["forecasting"]
    lead_mix = float(track_scoring["lead_time_wape_share"])
    weekly_mix = float(track_scoring["weekly_wape_share"])
    calibration_mix = float(track_scoring["lead_time_calibration_share"])
    weekly_calibration_mix = float(track_scoring["weekly_calibration_share"])
    metrics = _group_metric_rows(combined, horizon, lead_mix, weekly_mix)
    eligible = metrics.loc[
        metrics["n_weekly"].gt(0) & metrics["n_lead_time"].gt(0)
    ].copy()
    if eligible.empty:
        raise RuntimeError("No Track A candidate has both weekly and lead-time metrics")
    weights = {
        key: float(value)
        for key, value in config["scoring"]["tracks"]["forecasting"]["weights"].items()
    }
    ranking = _weighted_selection(
        eligible,
        weights,
        phase4_config["explainability_tier"],
        calibration_mix,
        weekly_calibration_mix,
    )
    sensitivity = _sensitivity(
        ranking,
        weights,
        phase4_config["explainability_tier"],
        float(config["scoring"]["weight_sensitivity_points"]) / 100.0,
    )
    stable = not bool(sensitivity["winner_changed"].any())
    ranking["sensitivity_stable_winner"] = stable
    sensitivity.to_csv(
        root / "results" / "track_a_weight_sensitivity.csv", index=False
    )
    ranking.to_csv(
        root / "results" / "track_a_selection_matrix.csv", index=False
    )

    statistics_config = dict(phase4_config["statistical_comparisons"])
    statistics_config["finalist_count"] = int(phase4_config["finalist_count"])
    comparisons = _statistical_comparisons(
        ranking,
        combined,
        horizon,
        statistics_config,
        int(statistics_config["random_seed"]),
    )
    comparisons.to_csv(
        root / "results" / "track_a_statistical_comparisons.csv", index=False
    )

    calibration_rows = []
    for _, row in metrics.iterrows():
        group = combined.loc[
            combined["model"].eq(row["model"])
            & combined["config_id"].eq(row["config_id"])
            & combined["actual_qty"].notna()
        ]
        values = _metric_values(group)
        lead_values = _metric_values(_lead_time_rows(group, horizon))
        calibration_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "weekly_p90_coverage": values["p90_coverage"],
                "weekly_p95_coverage": values["p95_coverage"],
                "weekly_p99_coverage": values["p99_coverage"],
                "mean_coverage_error": row["mean_coverage_error"],
                "pinball_loss": row["pinball_loss"],
                "lead_time_p90_coverage": lead_values["p90_coverage"],
                "lead_time_p95_coverage": lead_values["p95_coverage"],
                "lead_time_p99_coverage": lead_values["p99_coverage"],
                "lead_time_mean_coverage_error": row[
                    "lead_time_mean_coverage_error"
                ],
                "lead_time_pinball_loss": row["lead_time_pinball_loss"],
            }
        )
    calibration = pd.DataFrame(calibration_rows)
    calibration = calibration.merge(
        ranking[["candidate_id", "calibration_score"]],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    calibration.to_csv(
        root / "results" / "track_a_calibration_metrics.csv", index=False
    )

    model_artifact = None
    future_count = 0
    blocker = None
    if stable:
        champion = ranking.iloc[0]
        try:
            future_forecast, model_bundle = _forecast_final_model(
                root,
                config,
                demand,
                champion,
                tier2_selected_id,
                seed,
            )
            model_dir = root / "artifacts" / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = model_dir / "track_a_champion.joblib"
            joblib.dump(model_bundle, artifact_path)
            model_artifact = str(artifact_path.relative_to(root))
            if str(champion["model"]) != "lightgbm_quantile":
                blocker = (
                    f"Sensitivity-stable top candidate {champion['candidate_id']} has "
                    "no native probabilistic output. Per the frozen contract, no "
                    "quantiles were manufactured; the Phase 4 quantile forecast "
                    "deliverable is blocked."
                )
            else:
                if future_forecast.empty:
                    raise RuntimeError("Champion produced no split_role=future forecasts")
                future_forecast = future_forecast.merge(
                    demand.loc[
                        demand["split_role"].eq("future"),
                        ["part_id", "region", "week_start_date", "criticality"],
                    ].drop_duplicates(),
                    on=["part_id", "region", "week_start_date"],
                    how="left",
                    validate="one_to_one",
                )
                future_forecast["model"] = champion["model"]
                future_forecast["config_id"] = champion["config_id"]
                future_forecast["forecast_origin"] = training_end.date().isoformat()
                future_forecast["split_role"] = "future"
                future_forecast["forecast_qty"] = future_forecast[
                    "forecast_qty"
                ].clip(lower=0)
                future_forecast.to_csv(
                    root / "results" / "forecast_future_14w.csv", index=False
                )
                future_count = len(future_forecast)
        except (ImportError, ValueError, RuntimeError, TypeError) as error:
            blocker = (
                f"Stable candidate {ranking.iloc[0]['candidate_id']} could not be "
                f"frozen for future prediction: {type(error).__name__}: {error}"
            )
    else:
        blocker = (
            "No stable winner: the top weighted candidate changes in the one-at-a-time "
            "weight-sensitivity analysis. Top contenders are retained in the selection "
            "matrix; a champion was not forced."
        )

    if blocker:
        if (root / "results" / "forecast_future_14w.csv").exists():
            (root / "results" / "forecast_future_14w.csv").unlink()
    _write_reports(
        root,
        ranking,
        sensitivity,
        comparisons,
        calibration,
        correction,
        tier2_log,
        stable,
        model_artifact,
        future_count,
        blocker,
    )
    return {
        "tier2_selected_config_id": tier2_selected_id,
        "tier2_tuning_successes": int(
            tier2_log["status"].eq("success").sum()
        )
        if not tier2_log.empty
        else 0,
        "candidate_count": len(ranking),
        "sensitivity_changes": int(sensitivity["winner_changed"].sum()),
        "stable_winner": stable,
        "top_candidate": str(ranking.iloc[0]["candidate_id"]),
        "phase3_metric_corrections": len(correction),
        "statistical_comparisons": len(comparisons),
        "model_artifact": model_artifact,
        "future_forecast_rows": future_count,
        "blocker": blocker,
    }
