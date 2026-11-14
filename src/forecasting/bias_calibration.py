"""Tuning-only multiplicative bias calibration for the frozen Track A champion.

SPEC v4.1 admits a change only when it beats the frozen candidate on held
validation. Factors are fitted on the two tuning origins, applied unchanged to
the four held origins, and judged against pre-declared gates. Held labels are
used for scoring only.
"""
from __future__ import annotations

import gzip
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
    _run_global_fold,
)
from src.forecasting.phase4_v41 import (
    _bootstrap_inputs,
    _lead_time_rows,
    _metric_values,
    _pooled_wape,
)


CHAMPION_MODEL = "lightgbm_tweedie"
CHAMPION_CONFIG = "random_1"
FACTOR_BOUNDS = (0.50, 1.50)
MIN_GROUP_ROWS = 30
MIN_GROUP_ACTUAL_TOTAL = 1.0
LEAD_WAPE_MARGIN = 0.005
MAX_ABS_BIAS = 0.10
P95_COVERAGE_TOLERANCE = 0.01
SEED = 42
VOLUME_QUARTILES = ("Q1", "Q2", "Q3", "Q4")


def _finite_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    actual = pd.to_numeric(frame["actual_qty"], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    forecast = pd.to_numeric(frame["forecast_qty"], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    valid = np.isfinite(actual) & np.isfinite(forecast)
    return actual[valid], forecast[valid]


def fit_multiplicative_factor(
    frame: pd.DataFrame,
    *,
    min_rows: int = MIN_GROUP_ROWS,
    min_actual_total: float = MIN_GROUP_ACTUAL_TOTAL,
    bounds: tuple[float, float] = FACTOR_BOUNDS,
) -> float:
    """Fit one correction factor from tuning evidence only."""
    actual, forecast = _finite_arrays(frame)
    if len(actual) < min_rows:
        return 1.0
    actual_total = float(actual.sum())
    forecast_total = float(forecast.sum())
    if actual_total <= min_actual_total or forecast_total <= 0:
        return 1.0
    return float(np.clip(actual_total / forecast_total, *bounds))


def fit_group_factors(
    frame: pd.DataFrame,
    group_column: str,
    *,
    min_rows: int = MIN_GROUP_ROWS,
    min_actual_total: float = MIN_GROUP_ACTUAL_TOTAL,
    bounds: tuple[float, float] = FACTOR_BOUNDS,
) -> dict[str, float]:
    """Fit a factor per group, falling back to 1.0 where evidence is thin."""
    if group_column not in frame:
        raise ValueError(f"Missing grouping column: {group_column}")
    groups = frame[group_column].astype(str)
    return {
        str(value): fit_multiplicative_factor(
            group,
            min_rows=min_rows,
            min_actual_total=min_actual_total,
            bounds=bounds,
        )
        for value, group in frame.groupby(groups, observed=True)
    }


def apply_factors(
    frame: pd.DataFrame,
    *,
    factor: float = 1.0,
    factors: dict[str, float] | None = None,
    group_column: str | None = None,
) -> pd.DataFrame:
    """Scale forecast_qty only; quantiles are intentionally left unchanged."""
    if "forecast_qty" not in frame:
        raise ValueError("Frame lacks forecast_qty")
    if group_column is None and factors is not None:
        raise ValueError("Group factors require a group column")
    if group_column is not None and factors is None:
        raise ValueError("A group column requires group factors")
    output = frame.copy()
    if group_column is None:
        multipliers = pd.Series(float(factor), index=output.index)
    else:
        if group_column not in output:
            raise ValueError(f"Missing grouping column: {group_column}")
        multipliers = (
            output[group_column].astype(str).map(factors).fillna(1.0)
        )
    output["forecast_qty"] = (
        pd.to_numeric(output["forecast_qty"], errors="raise")
        * multipliers.astype(float)
    )
    return output


def evaluate_candidate(
    baseline: dict[str, float],
    candidate: dict[str, float],
    *,
    lead_wape_margin: float = LEAD_WAPE_MARGIN,
    max_abs_bias: float = MAX_ABS_BIAS,
    p95_tolerance: float = P95_COVERAGE_TOLERANCE,
) -> dict[str, object]:
    """Apply the pre-declared acceptance gates to one held-origin candidate."""
    reasons: list[str] = []
    wape_gain = float(baseline["lead_time_wape"]) - float(
        candidate["lead_time_wape"]
    )
    if not wape_gain >= lead_wape_margin:
        reasons.append(
            f"lead_time_wape_gain {wape_gain:.4f} below required {lead_wape_margin:.4f}"
        )
    bias = float(candidate["lead_time_bias"])
    if not abs(bias) <= max_abs_bias:
        reasons.append(
            f"abs_lead_time_bias {abs(bias):.4f} above limit {max_abs_bias:.4f}"
        )
    baseline_coverage = baseline.get("p95_coverage", np.nan)
    candidate_coverage = candidate.get("p95_coverage", np.nan)
    if np.isfinite(baseline_coverage) and np.isfinite(candidate_coverage):
        if candidate_coverage < baseline_coverage - p95_tolerance:
            reasons.append(
                f"p95_coverage {candidate_coverage:.4f} below baseline "
                f"{baseline_coverage:.4f} minus tolerance {p95_tolerance:.4f}"
            )
    return {
        "passed": not reasons,
        "reasons": reasons,
        "lead_time_wape_gain": wape_gain,
        "lead_time_bias": bias,
    }


def _read_config(root: Path) -> dict[str, Any]:
    with (root / "configs" / "project.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Project configuration must be a mapping")
    return config


def _champion_parameters(root: Path) -> dict[str, float | int]:
    log = pd.read_csv(root / "results" / "track_a_tuning_log.csv")
    rows = log.loc[
        log["model"].eq(CHAMPION_MODEL)
        & log["config_id"].eq(CHAMPION_CONFIG)
        & log["status"].eq("success")
    ]
    if rows.empty:
        raise ValueError("Frozen champion parameters are absent from the tuning log")
    parameters = json.loads(str(rows.iloc[0]["parameters"]))
    return {name: float(value) if isinstance(value, float) else int(value)
            for name, value in parameters.items()}


def _metric_summary(frame: pd.DataFrame) -> dict[str, float]:
    values = _metric_values(_bootstrap_inputs(frame))
    return {
        "wape": float(values["wape_ratio"]),
        "bias": float(values["bias_ratio"]),
        "p95_coverage": float(values.get("p95_coverage", np.nan)),
    }


def _aggregate_bias(frame: pd.DataFrame) -> float:
    actual = pd.to_numeric(frame["actual_qty"], errors="coerce").sum()
    forecast = pd.to_numeric(frame["forecast_qty"], errors="coerce").sum()
    if not np.isfinite(actual) or actual == 0:
        return float("nan")
    return float((forecast - actual) / actual)


def _high_criticality_bias(lead_rows: pd.DataFrame) -> float:
    return _aggregate_bias(
        lead_rows.loc[lead_rows["criticality"].astype(str).eq("High")]
    )


def _per_origin_wape(lead_rows: pd.DataFrame) -> dict[str, float]:
    return {
        pd.Timestamp(origin).date().isoformat(): float(_pooled_wape(group))
        for origin, group in lead_rows.groupby("origin_date", sort=True)
    }


def _volume_assignment(
    tuning_lead: pd.DataFrame,
    frame: pd.DataFrame,
) -> pd.Series:
    """Assign volume quartiles from tuning-only per-series actual totals."""
    totals = (
        tuning_lead.groupby(["part_id", "region"], observed=True)["actual_qty"]
        .sum()
        .rename("series_volume")
    )
    edges = [-np.inf, *totals.quantile([0.25, 0.50, 0.75]).tolist(), np.inf]
    merged = frame.merge(
        totals.reset_index(), on=["part_id", "region"], how="left",
        validate="many_to_one",
    )
    labels = pd.cut(
        merged["series_volume"], bins=edges, labels=list(VOLUME_QUARTILES)
    )
    return pd.Series(
        labels.astype(object).fillna("unmapped").to_numpy(), index=frame.index
    )


def _variant_metrics(
    held_weekly: pd.DataFrame,
    held_lead: pd.DataFrame,
    *,
    factor: float = 1.0,
    factors: dict[str, float] | None = None,
    group_column: str | None = None,
) -> dict[str, object]:
    corrected_weekly = apply_factors(
        held_weekly, factor=factor, factors=factors, group_column=group_column
    )
    corrected_lead = apply_factors(
        held_lead, factor=factor, factors=factors, group_column=group_column
    )
    weekly = _metric_summary(corrected_weekly)
    lead = _metric_summary(corrected_lead)
    return {
        "weekly_wape": weekly["wape"],
        "weekly_bias": weekly["bias"],
        "lead_time_wape": lead["wape"],
        "lead_time_bias": lead["bias"],
        "p95_coverage": lead["p95_coverage"],
        "high_criticality_bias": _high_criticality_bias(corrected_lead),
        "per_origin_lead_wape": _per_origin_wape(corrected_lead),
    }


def _json_safe(value: object) -> object:
    """Replace non-finite floats so the summary remains strict JSON."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def run_bias_calibration(root: Path = ROOT) -> dict[str, object]:
    """Fit calibration on tuning origins, score on held origins, and report."""
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
    parameters = _champion_parameters(root)

    demand = load_demand(root / "data" / "01_demand_forecasting.csv")
    scheduling = load_scheduling(root / "data" / "02_scheduling_inputs.csv")
    lead_times = _lead_time_map(scheduling)

    tuning_frames: list[pd.DataFrame] = []
    tuning_lead_frames: list[pd.DataFrame] = []
    for origin in tuning_origins:
        history, targets, calendar = _fold_data(demand, origin, horizon)
        metadata, scales = _origin_report_metadata(history, targets)
        fold_started = time.perf_counter()
        predictions, _, _ = _run_global_fold(
            history, calendar, origin, CHAMPION_MODEL, parameters, SEED
        )
        runtime = time.perf_counter() - fold_started
        decorated = _decorate_forecasts(
            predictions, targets, metadata, scales, lead_times,
            CHAMPION_MODEL, CHAMPION_CONFIG, "tuning", origin, horizon, runtime,
        )
        decorated = decorated.loc[decorated["actual_qty"].notna()].copy()
        tuning_frames.append(decorated)
        tuning_lead_frames.append(_lead_time_rows(decorated, extension_weeks))
    tuning_weekly = pd.concat(tuning_frames, ignore_index=True)
    tuning_lead = pd.concat(tuning_lead_frames, ignore_index=True)

    artifact = root / "artifacts" / "bias_calibration_tuning_forecasts.csv.gz"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(artifact, "wb", mtime=0) as stream:
        tuning_weekly.drop(columns=["runtime_seconds"], errors="ignore").to_csv(
            stream, index=False
        )

    held_source = pd.read_parquet(
        root / "artifacts" / "track_a_backtest_forecasts_v41.parquet"
    )
    held_weekly = held_source.loc[
        held_source["model"].eq(CHAMPION_MODEL)
        & held_source["config_id"].eq(CHAMPION_CONFIG)
        & held_source["actual_qty"].notna()
    ].copy()
    if held_weekly.empty:
        raise ValueError("Held champion forecasts are absent from the artifact")
    held_weekly["origin_date"] = pd.to_datetime(held_weekly["origin_date"])
    observed_origins = [pd.Timestamp(value) for value in held_weekly["origin_date"].unique()]
    if (
        len(observed_origins) != 4
        or set(observed_origins) != set(held_origins)
        or max(observed_origins) > pd.Timestamp("2026-03-23")
    ):
        raise ValueError("Held artifact does not match the frozen four-origin window")
    held_lead = _lead_time_rows(held_weekly, extension_weeks)
    held_lead["criticality"] = held_lead["criticality"].astype(str)

    baseline = _variant_metrics(held_weekly, held_lead)
    baseline_gate = {
        "lead_time_wape": baseline["lead_time_wape"],
        "lead_time_bias": baseline["lead_time_bias"],
        "p95_coverage": baseline["p95_coverage"],
    }

    variants: dict[str, dict[str, object]] = {}

    def record(
        name: str,
        *,
        factor: float = 1.0,
        factors: dict[str, float] | None = None,
        group_column: str | None = None,
        evidence: str,
    ) -> None:
        metrics = _variant_metrics(
            held_weekly, held_lead, factor=factor, factors=factors,
            group_column=group_column,
        )
        gate = evaluate_candidate(baseline_gate, metrics)
        variants[name] = {
            "evidence": evidence,
            "factor": factor,
            "factors": factors,
            "group_column": group_column,
            **metrics,
            "gate_passed": gate["passed"],
            "gate_reasons": gate["reasons"],
            "lead_time_wape_gain": gate["lead_time_wape_gain"],
        }

    pooled_lead_factor = fit_multiplicative_factor(tuning_lead)
    record(
        "lead_pooled", factor=pooled_lead_factor,
        evidence="all tuning lead-time rows",
    )
    pooled_weekly_factor = fit_multiplicative_factor(tuning_weekly)
    record(
        "weekly_pooled", factor=pooled_weekly_factor,
        evidence="all tuning weekly rows",
    )
    recent_origin = max(tuning_origins)
    recent = tuning_lead.loc[tuning_lead["origin_date"].eq(recent_origin)]
    record(
        "recent_origin", factor=fit_multiplicative_factor(recent),
        evidence=f"lead-time rows at {recent_origin.date()}",
    )

    for name, column in (
        ("lead_criticality", "criticality"),
        ("lead_sbc", "sbc_class"),
        ("lead_abc", "abc_class"),
    ):
        record(
            name,
            factors=fit_group_factors(tuning_lead, column),
            group_column=column,
            evidence=f"tuning lead-time rows grouped by {column}",
        )

    tuning_volume = _volume_assignment(tuning_lead, tuning_lead)
    tuning_with_volume = tuning_lead.assign(volume_quartile=tuning_volume)
    volume_factors = fit_group_factors(tuning_with_volume, "volume_quartile")
    held_with_volume = held_lead.assign(
        volume_quartile=_volume_assignment(tuning_lead, held_lead)
    )
    volume_metrics = _variant_metrics(
        held_weekly.assign(
            volume_quartile=_volume_assignment(tuning_lead, held_weekly)
        ),
        held_with_volume,
        factors=volume_factors,
        group_column="volume_quartile",
    )
    volume_gate = evaluate_candidate(baseline_gate, volume_metrics)
    variants["lead_volume_quartile"] = {
        "evidence": "tuning lead-time rows grouped by tuning-only volume quartile",
        "factor": 1.0,
        "factors": volume_factors,
        "group_column": "volume_quartile",
        **volume_metrics,
        "gate_passed": volume_gate["passed"],
        "gate_reasons": volume_gate["reasons"],
        "lead_time_wape_gain": volume_gate["lead_time_wape_gain"],
    }

    segment_transfer = []
    for column in ("sbc_class", "abc_class", "criticality"):
        for value in sorted(tuning_lead[column].astype(str).unique()):
            tuning_slice = tuning_lead.loc[
                tuning_lead[column].astype(str).eq(value)
            ]
            held_slice = held_lead.loc[held_lead[column].astype(str).eq(value)]
            if len(tuning_slice) < MIN_GROUP_ROWS or len(held_slice) < MIN_GROUP_ROWS:
                continue
            tuning_bias = _aggregate_bias(tuning_slice)
            held_bias = _aggregate_bias(held_slice)
            segment_transfer.append(
                {
                    "dimension": column,
                    "segment": value,
                    "tuning_rows": len(tuning_slice),
                    "held_rows": len(held_slice),
                    "tuning_bias": tuning_bias,
                    "held_bias": held_bias,
                    "tuning_factor": fit_multiplicative_factor(tuning_slice),
                }
            )
    for value in VOLUME_QUARTILES:
        tuning_slice = tuning_with_volume.loc[
            tuning_with_volume["volume_quartile"].eq(value)
        ]
        held_slice = held_with_volume.loc[
            held_with_volume["volume_quartile"].eq(value)
        ]
        if len(tuning_slice) < MIN_GROUP_ROWS or len(held_slice) < MIN_GROUP_ROWS:
            continue
        segment_transfer.append(
            {
                "dimension": "volume_quartile",
                "segment": value,
                "tuning_rows": len(tuning_slice),
                "held_rows": len(held_slice),
                "tuning_bias": _aggregate_bias(tuning_slice),
                "held_bias": _aggregate_bias(held_slice),
                "tuning_factor": fit_multiplicative_factor(tuning_slice),
            }
        )

    passed = [name for name, value in variants.items() if value["gate_passed"]]
    decision = (
        "adopt_" + passed[0]
        if passed
        else "keep_frozen_champion_uncorrected"
    )
    summary = {
        "champion": f"{CHAMPION_MODEL}::{CHAMPION_CONFIG}",
        "champion_parameters": parameters,
        "seed": SEED,
        "tuning_origins": [value.date().isoformat() for value in tuning_origins],
        "held_origins": [value.date().isoformat() for value in held_origins],
        "extension_period_weeks": extension_weeks,
        "horizon_weeks": horizon,
        "factor_bounds": list(FACTOR_BOUNDS),
        "min_group_rows": MIN_GROUP_ROWS,
        "acceptance_gates": {
            "lead_time_wape_margin": LEAD_WAPE_MARGIN,
            "max_abs_lead_time_bias": MAX_ABS_BIAS,
            "p95_coverage_tolerance": P95_COVERAGE_TOLERANCE,
        },
        "baseline": baseline,
        "variants": variants,
        "segment_transfer": segment_transfer,
        "passed_variants": passed,
        "decision": decision,
        "final_test_sealed": True,
        "runtime_seconds": time.perf_counter() - started,
    }

    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, value in variants.items():
        rows.append(
            {
                "variant": name,
                "evidence": value["evidence"],
                "factor": value["factor"],
                "factors": json.dumps(value["factors"], sort_keys=True),
                "weekly_wape_ratio": value["weekly_wape"],
                "weekly_bias_ratio": value["weekly_bias"],
                "lead_time_wape_ratio": value["lead_time_wape"],
                "lead_time_bias_ratio": value["lead_time_bias"],
                "high_criticality_bias_ratio": value["high_criticality_bias"],
                "p95_coverage": value["p95_coverage"],
                "lead_time_wape_gain": value["lead_time_wape_gain"],
                "per_origin_lead_wape": json.dumps(
                    value["per_origin_lead_wape"], sort_keys=True
                ),
                "gate_passed": value["gate_passed"],
                "gate_reasons": json.dumps(value["gate_reasons"]),
            }
        )
    pd.DataFrame(rows).to_csv(
        results_dir / "bias_calibration_candidates.csv", index=False
    )
    with (results_dir / "bias_calibration_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(_json_safe(summary), stream, indent=2, sort_keys=True)
    pd.DataFrame(segment_transfer).to_csv(
        results_dir / "bias_calibration_segment_transfer.csv", index=False
    )

    return summary


if __name__ == "__main__":
    print(json.dumps(_json_safe(run_bias_calibration()), indent=2, sort_keys=True))
