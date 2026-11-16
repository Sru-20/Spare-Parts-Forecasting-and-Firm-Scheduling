"""Fail-closed Phase 7 final-test runner.

Preflight reads only split/date metadata. The final-test attempt marker is
claimed before any demand target column is loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from src.data.loader import load_scheduling
from src.features.segmentation import build_segment_table
from src.features.time_features import CALENDAR_FEATURES, STATIC_FEATURES
from src.forecasting.backtest import _lead_time_map, _lead_time_rows
from src.forecasting.global_models import fit_global_models
from src.forecasting.metrics import forecast_metrics, naive_scale_by_series
from src.scheduling.checker import check_schedule
from src.scheduling.phase5 import (
    HARD_CONSTRAINTS,
    PRIMARY_INITIALIZATION,
    _supplier_snapshot,
    load_origin_visible_history,
)
from src.scheduling.phase6 import (
    _build_provisional_schedule,
    _checker_parts,
    _initial_snapshot,
    _run_plan,
)
from src.simulation.weekly import realize_supplier_orders


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = Path("configs/phase7.yaml")
PROJECT_CONFIG_PATH = Path("configs/project.yaml")
BUSINESS_CONFIG_PATH = Path("configs/business.yaml")
FINAL_TEST_CONFIRMATION = "CONSUME-PHASE7-FINAL-TEST-ONCE"
FINALIZE_CONFIRMATION = "CREATE-PHASE7-FINAL-FORECAST-ONCE"
FINAL_TEST_RUN_ID = "phase7-final-test-2026-03-23-v1"
PHASE5_GATE_EVIDENCE_PATH = Path("results/phase5_exit_gate_evidence_v1.json")
PHASE5_REQUIRED_EVIDENCE = {
    "approved_analogue_exception_recorded": True,
    "python_3_10_tests_passed": True,
    "python_3_13_tests_passed": True,
    "validator_passes": 32,
    "validator_failures": 0,
    "firm_order_hard_constraint_violations": 0,
    "inventory_reconciliation_passed": True,
    "repeat_run_identical": True,
    "assumptions_frozen": True,
}
FROZEN_PROJECT_SHA256 = (
    "46a6183450abad85c84ea3df41e412709e189b9314d42c8d1fb0c34a533314cc"
)
FROZEN_BUSINESS_SHA256 = (
    "ed0e3df890504ed02f2aef27ee3ba6ca4a431ef6e40d07951dc4d86b231e6540"
)

FROZEN_PROTOCOL: dict[str, Any] = {
    "protocol_id": "phase7-final-test-2026-03-23-v1",
    "evaluation": {
        "origin": "2026-03-23",
        "split_role": "final_test",
        "start": "2026-03-30",
        "end": "2026-09-21",
        "horizon_weeks": 26,
        "primary_horizon_weeks": 14,
        "extended_horizon_start_week": 15,
    },
    "models": {
        "end_to_end_champion": {
            "candidate_id": "lightgbm_quantile::random_1",
            "family": "lightgbm_quantile",
            "config_id": "random_1",
            "seed": 42,
            "parameters": {
                "feature_fraction": 1.0,
                "learning_rate": 0.03,
                "min_child_samples": 20,
                "n_estimators": 300,
                "num_leaves": 63,
            },
            "quantiles": [0.5, 0.9, 0.95, 0.99],
        },
        "point_forecast_benchmark": {
            "candidate_id": "lightgbm_tweedie::random_1",
            "family": "lightgbm_tweedie",
            "config_id": "random_1",
            "seed": 42,
            "parameters": {
                "feature_fraction": 1.0,
                "learning_rate": 0.03,
                "min_child_samples": 20,
                "n_estimators": 300,
                "num_leaves": 63,
            },
        },
    },
    "scheduling": {
        "winning_policy": "risk_adjusted_mrp",
        "traditional_baseline_policy": "traditional_baseline",
        "traditional_baseline_forecast": "historical_average",
    },
    "future": {
        "start": "2026-09-28",
        "end": "2026-12-28",
        "horizon_weeks": 14,
    },
}

SAFE_CALENDAR_COLUMNS = (
    "year",
    "month",
    "quarter_fy",
    "fiscal_year",
    "week_of_year",
    "week_of_month",
    "festival_flag",
    "festival_name",
    "festival_intensity",
    "weeks_to_next_festival",
    "weeks_since_last_festival",
    "public_holiday_flag",
    "monsoon_flag",
    "summer_peak_flag",
    "fiscal_year_end_flag",
    "dealer_order_cycle_flag",
)
EVALUATION_OUTPUTS = (
    "results/phase7_forecasts_v1.csv",
    "results/phase7_forecast_metrics_v1.csv",
    "results/phase7_schedule_checks_v1.csv",
    "results/phase7_orders_v1.csv",
    "results/phase7_simulation_weekly_v1.csv",
    "results/phase7_scheduler_kpis_v1.csv",
    "results/phase7_exclusions_v1.csv",
    "reports/phase7_final_test_report.md",
    "results/phase7_run_metadata_v1.json",
)
ATTEMPT_MARKER = Path("results/phase7_final_test_consumption_v1.json")
FINALIZATION_OUTPUTS = (
    "results/forecast_future_14w_final_v1.csv",
    "results/firm_schedule_14w_final_v1.csv",
    "results/phase7_finalization_metadata_v1.json",
)
DEMAND_READ_COLUMNS = tuple(
    dict.fromkeys(
        [
            "week_start_date",
            "part_id",
            "region",
            "split_role",
            "is_missing_week",
            "demand_qty",
            *STATIC_FEATURES,
            *SAFE_CALENDAR_COLUMNS,
        ]
    )
)
FUTURE_PREDICTOR_COLUMNS = tuple(
    column
    for column in DEMAND_READ_COLUMNS
    if column not in {"demand_qty", "is_missing_week"}
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def _selected_tuning_parameters(root: Path, model: str) -> dict[str, Any]:
    tuning_path = root / "results" / "track_a_tuning_log.csv"
    tuning = pd.read_csv(tuning_path)
    selected = tuning.loc[
        tuning["model"].eq(model)
        & tuning["config_id"].eq("random_1")
        & tuning["status"].eq("success")
        & tuning["selected_for_held_validation"].astype(str).str.lower().eq("true")
    ]
    if selected.empty:
        raise ValueError(f"No frozen successful random_1 tuning row for {model}")
    parameter_sets = {
        json.dumps(json.loads(value), sort_keys=True)
        for value in selected["parameters"].dropna().astype(str)
    }
    if len(parameter_sets) != 1:
        raise ValueError(f"Frozen random_1 parameters disagree for {model}")
    return json.loads(next(iter(parameter_sets)))


def validate_protocol(root: Path = ROOT) -> dict[str, Any]:
    """Validate the immutable Phase 7 contract and its saved model handoff."""
    protocol_path = root / PROTOCOL_PATH
    protocol = _read_yaml(protocol_path)
    if protocol != FROZEN_PROTOCOL:
        raise ValueError(
            "Phase 7 protocol differs from its frozen contract; evaluation is blocked"
        )

    project_path = root / PROJECT_CONFIG_PATH
    business_path = root / BUSINESS_CONFIG_PATH
    project = _read_yaml(project_path)
    if _sha256(project_path) != FROZEN_PROJECT_SHA256:
        raise ValueError("project.yaml differs from the frozen Phase 6 configuration")
    if _sha256(business_path) != FROZEN_BUSINESS_SHA256:
        raise ValueError("business.yaml differs from the frozen simulator assumptions")
    if project.get("specification", {}).get("version") != "4.2":
        raise ValueError("Phase 7 requires governing specification v4.2")
    forecast_config = project.get("forecast", {})
    expected_dates = {
        "train_validation_end": "2026-03-23",
        "final_test_start": "2026-03-30",
        "final_test_end": "2026-09-21",
        "future_start": "2026-09-28",
        "future_end": "2026-12-28",
    }
    split_dates = forecast_config.get("split_dates", {})
    if any(split_dates.get(key) != value for key, value in expected_dates.items()):
        raise ValueError("project.yaml split boundaries differ from Phase 7 protocol")
    if (
        forecast_config.get("training_target_column") != "demand_qty"
        or forecast_config.get("scoring_target_column") != "demand_qty"
        or forecast_config.get("exclude_missing_week_from_fit_and_score") is not True
        or forecast_config.get("genuine_zero_demand_is_observed") is not True
    ):
        raise ValueError("Project target/missing/zero-demand contract has changed")
    if forecast_config.get("final_test_open_phase") != 7:
        raise ValueError("project.yaml does not seal the final test for Phase 7")

    champion = protocol["models"]["end_to_end_champion"]
    benchmark = protocol["models"]["point_forecast_benchmark"]
    if champion["candidate_id"] == benchmark["candidate_id"]:
        raise ValueError("The end-to-end champion and point benchmark must stay distinct")
    for model_entry in (champion, benchmark):
        actual_parameters = _selected_tuning_parameters(root, model_entry["family"])
        if actual_parameters != model_entry["parameters"]:
            raise ValueError(
                f"Frozen tuning parameters changed for {model_entry['candidate_id']}"
            )

    handoff_path = root / "reports" / "phase6_comparison_report.md"
    handoff = handoff_path.read_text(encoding="utf-8")
    expected_winner = (
        "`lightgbm_quantile::random_1` + `risk_adjusted_mrp`"
    )
    if expected_winner not in handoff or FROZEN_PROJECT_SHA256 not in handoff:
        raise ValueError("Saved Phase 6 handoff does not match the frozen winner/config")
    required_handoff_files = (
        "results/phase6_scheduler_comparison_v1.csv",
        "results/phase6_kpi_comparison_v1.csv",
        "results/phase6_checker_summary_v1.csv",
        "results/phase6_provisional_firm_schedule_v1.csv",
    )
    missing_handoff = [
        path for path in required_handoff_files if not (root / path).is_file()
    ]
    if missing_handoff:
        raise FileNotFoundError(f"Phase 6 handoff artifacts missing: {missing_handoff}")

    return {
        "protocol_id": protocol["protocol_id"],
        "phase7_config_sha256": _sha256(protocol_path),
        "project_config_sha256": _sha256(project_path),
        "business_config_sha256": _sha256(business_path),
        "phase6_report_sha256": _sha256(handoff_path),
        "track_a_tuning_sha256": _sha256(
            root / "results" / "track_a_tuning_log.csv"
        ),
        "champion_id": champion["candidate_id"],
        "point_benchmark_id": benchmark["candidate_id"],
        "scheduler_id": protocol["scheduling"]["winning_policy"],
        "baseline_id": (
            f'{protocol["scheduling"]["traditional_baseline_forecast"]}::'
            f'{protocol["scheduling"]["traditional_baseline_policy"]}'
        ),
    }


def _read_split_metadata(path: Path) -> pd.DataFrame:
    """Read only split/date columns; never request a target or outcome column."""
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
    required = {"week_start_date", "split_role"}
    if not required.issubset(header.columns):
        raise ValueError(f"Demand CSV is missing split metadata: {sorted(required)}")
    metadata = pd.read_csv(
        path,
        usecols=["week_start_date", "split_role"],
        dtype={"week_start_date": "string", "split_role": "string"},
        keep_default_na=False,
        encoding="utf-8-sig",
        low_memory=False,
    )
    metadata["week_start_date"] = pd.to_datetime(
        metadata["week_start_date"], errors="raise"
    )
    if metadata[["week_start_date", "split_role"]].isna().any().any():
        raise ValueError("Split/date metadata contains missing values")
    return metadata


def _validate_split_metadata(metadata: pd.DataFrame) -> dict[str, Any]:
    expected = FROZEN_PROTOCOL["evaluation"]
    origin = pd.Timestamp(expected["origin"])
    test_start = pd.Timestamp(expected["start"])
    test_end = pd.Timestamp(expected["end"])
    test_dates = pd.date_range(test_start, test_end, freq="W-MON")
    future_start = pd.Timestamp(FROZEN_PROTOCOL["future"]["start"])
    future_end = pd.Timestamp(FROZEN_PROTOCOL["future"]["end"])
    future_dates = pd.date_range(future_start, future_end, freq="W-MON")

    role_dates = {
        role: pd.DatetimeIndex(
            metadata.loc[metadata["split_role"].eq(role), "week_start_date"]
            .drop_duplicates()
            .sort_values()
        )
        for role in ("train_val", "final_test", "future")
    }
    expected_roles = {"warmup", "train_val", "final_test", "future"}
    actual_roles = set(metadata["split_role"].dropna().astype(str))
    if actual_roles != expected_roles:
        raise ValueError(
            f"Split roles differ from the frozen contract: {sorted(actual_roles)}"
        )
    if not len(role_dates["train_val"]) or role_dates["train_val"].max() != origin:
        raise ValueError("Train-validation metadata does not end at the frozen origin")
    if not role_dates["final_test"].equals(test_dates):
        raise ValueError("Final-test split dates differ from the frozen 26-week window")
    if not role_dates["future"].equals(future_dates):
        raise ValueError("Future split dates differ from the frozen 14-week window")
    split_summary = (
        metadata.groupby("split_role", observed=True)
        .agg(rows=("week_start_date", "size"), weeks=("week_start_date", "nunique"))
        .sort_index()
    )
    canonical = {
        "roles": {
            str(role): {
                "rows": int(row.rows),
                "weeks": int(row.weeks),
                "first_date": str(
                    metadata.loc[
                        metadata["split_role"].eq(role), "week_start_date"
                    ].min().date()
                ),
                "last_date": str(
                    metadata.loc[
                        metadata["split_role"].eq(role), "week_start_date"
                    ].max().date()
                ),
            }
            for role, row in split_summary.iterrows()
        }
    }
    manifest_hash = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "split_manifest": canonical,
        "split_manifest_sha256": manifest_hash,
        "final_test_weeks": len(test_dates),
        "future_weeks": len(future_dates),
    }


def _phase5_gate_blockers(root: Path) -> list[str]:
    state = (root / "STATE.md").read_text(encoding="utf-8")
    current_status_lines = [
        line.strip()
        for line in state.splitlines()
        if line.startswith("- **Phase 5:")
    ]
    expected_status = (
        "- **Phase 5: COMPLETE WITH APPROVED DATA-LIMITATION EXCEPTION**"
    )
    if len(current_status_lines) != 1 or not current_status_lines[0].startswith(
        expected_status
    ):
        return [
            "The current Phase 5 status is not COMPLETE WITH APPROVED "
            "DATA-LIMITATION EXCEPTION."
        ]

    evidence_path = root / PHASE5_GATE_EVIDENCE_PATH
    if not evidence_path.is_file():
        return [f"Phase 5 exit-gate evidence is missing: {PHASE5_GATE_EVIDENCE_PATH}"]
    with evidence_path.open(encoding="utf-8") as stream:
        evidence = json.load(stream)
    if not isinstance(evidence, dict):
        raise ValueError("Phase 5 exit-gate evidence must be a JSON object")
    if (
        evidence.get("phase") != 5
        or evidence.get("status")
        != "COMPLETE WITH APPROVED DATA-LIMITATION EXCEPTION"
    ):
        return ["Phase 5 exit-gate evidence has the wrong phase or status."]
    criteria = evidence.get("criteria")
    if not isinstance(criteria, dict) or any(
        criteria.get(name) != expected
        for name, expected in PHASE5_REQUIRED_EVIDENCE.items()
    ):
        return ["Phase 5 exit-gate evidence is incomplete or does not pass."]
    return []


def _phase5_is_closed(root: Path) -> bool:
    return not _phase5_gate_blockers(root)


def _ensure_destinations_are_new(root: Path) -> None:
    outputs = (*EVALUATION_OUTPUTS, str(ATTEMPT_MARKER), *FINALIZATION_OUTPUTS)
    existing = [path for path in outputs if (root / path).exists()]
    if existing:
        raise FileExistsError(
            "Phase 7 destinations already exist; refusing to overwrite: "
            + ", ".join(existing)
        )
    missing_dirs = sorted(
        {
            str((root / path).parent.relative_to(root))
            for path in outputs
            if not (root / path).parent.is_dir()
        }
    )
    if missing_dirs:
        raise FileNotFoundError(
            f"Phase 7 output directories must already exist: {missing_dirs}"
        )


def safe_preflight(root: Path = ROOT) -> dict[str, Any]:
    """Validate configuration and split metadata without reading target values."""
    hashes = validate_protocol(root)
    demand_path = root / "data" / "01_demand_forecasting.csv"
    metadata = _read_split_metadata(demand_path)
    split = _validate_split_metadata(metadata)
    d2_date = pd.read_csv(
        root / "data" / "02_scheduling_inputs.csv",
        usecols=["as_of_date"],
        dtype={"as_of_date": "string"},
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    d2_dates = sorted(set(d2_date["as_of_date"].dropna().astype(str)))
    if len(d2_dates) != 1:
        raise ValueError("Scheduling snapshot must have one as_of_date")
    _ensure_destinations_are_new(root)
    blockers = _phase5_gate_blockers(root)
    warnings = []
    if pd.Timestamp(d2_dates[0]) > pd.Timestamp(
        FROZEN_PROTOCOL["evaluation"]["end"]
    ):
        warnings.append(
            "D2 snapshot postdates the final-test window; scheduling results are "
            "a simulated policy comparison, not historical operating-state evidence."
        )
    return {
        "status": "blocked" if blockers else "safe_preflight_passed",
        "target_values_read": False,
        "target_columns_requested": [],
        "outcome_columns_requested": [],
        "evaluation_consumption_state": "NOT_CONSUMED",
        "d2_snapshot_as_of_date": d2_dates[0],
        "blockers": blockers,
        "warnings": warnings,
        **hashes,
        **split,
    }


def _read_csv_rows(
    path: Path,
    row_mask: pd.Series,
    *,
    usecols: tuple[str, ...] | list[str],
    dtype: Mapping[str, str],
) -> pd.DataFrame:
    selected_rows = frozenset(
        int(position) for position in np.flatnonzero(row_mask.to_numpy())
    )
    selected_dtypes = {name: kind for name, kind in dtype.items() if name in usecols}
    return pd.read_csv(
        path,
        usecols=list(usecols),
        skiprows=lambda line: line > 0 and (line - 1) not in selected_rows,
        dtype=selected_dtypes,
        keep_default_na=True,
        encoding="utf-8-sig",
        low_memory=False,
    )


def _read_demand_for_authorized_operation(
    root: Path,
    *,
    expected_split_manifest_sha256: str,
) -> pd.DataFrame:
    """Validate metadata, then read only approved target-bearing rows."""
    path = root / "data" / "01_demand_forecasting.csv"
    metadata = _read_split_metadata(path)
    split = _validate_split_metadata(metadata)
    if split["split_manifest_sha256"] != expected_split_manifest_sha256:
        raise ValueError("Demand split metadata changed after protocol preflight")

    future_rows = metadata["split_role"].eq("future")
    future_targets = _read_csv_rows(
        path,
        future_rows,
        usecols=["demand_qty"],
        dtype={"demand_qty": "Float64"},
    )
    if len(future_targets) != int(future_rows.sum()):
        raise ValueError("Future target row count differs from split metadata")
    if future_targets["demand_qty"].notna().any():
        raise ValueError("Future split contains populated demand targets")

    origin = pd.Timestamp(FROZEN_PROTOCOL["evaluation"]["origin"])
    test_start = pd.Timestamp(FROZEN_PROTOCOL["evaluation"]["start"])
    test_end = pd.Timestamp(FROZEN_PROTOCOL["evaluation"]["end"])
    dates = metadata["week_start_date"]
    historical_rows = (
        metadata["split_role"].isin(["warmup", "train_val"])
        & dates.le(origin)
    )
    final_test_rows = (
        metadata["split_role"].eq("final_test")
        & dates.between(test_start, test_end)
    )
    allowed_rows = historical_rows | final_test_rows
    if not (allowed_rows | future_rows).all():
        raise ValueError("Demand rows fall outside the approved historical/test/future windows")
    frame = _read_csv_rows(
        path,
        allowed_rows,
        usecols=DEMAND_READ_COLUMNS,
        dtype={
            "week_start_date": "string",
            "split_role": "string",
            "demand_qty": "Float64",
            "is_missing_week": "Int64",
        },
    )
    expected_rows = metadata.loc[allowed_rows].reset_index(drop=True)
    if len(frame) != len(expected_rows):
        raise ValueError("Target-bearing row count differs from split metadata")
    frame["week_start_date"] = pd.to_datetime(
        frame["week_start_date"], errors="raise"
    )
    if not (
        frame["week_start_date"].reset_index(drop=True).equals(
            expected_rows["week_start_date"].reset_index(drop=True)
        )
        and frame["split_role"].astype("string").reset_index(drop=True).equals(
            expected_rows["split_role"].astype("string").reset_index(drop=True)
        )
    ):
        raise ValueError("Demand split metadata changed during target-bearing read")
    missing_flags = frame["is_missing_week"]
    if missing_flags.isna().any() or not missing_flags.isin([0, 1]).all():
        raise ValueError("is_missing_week must contain only non-null 0/1 values")
    if (missing_flags.eq(1) & frame["demand_qty"].notna()).any():
        raise ValueError("Missing-week rows must not contain demand targets")

    future_predictors = _read_csv_rows(
        path,
        future_rows,
        usecols=FUTURE_PREDICTOR_COLUMNS,
        dtype={"week_start_date": "string", "split_role": "string"},
    )
    expected_future = metadata.loc[future_rows].reset_index(drop=True)
    if len(future_predictors) != len(expected_future):
        raise ValueError("Future predictor row count differs from split metadata")
    future_predictors["week_start_date"] = pd.to_datetime(
        future_predictors["week_start_date"], errors="raise"
    )
    if not (
        future_predictors["week_start_date"].reset_index(drop=True).equals(
            expected_future["week_start_date"].reset_index(drop=True)
        )
        and future_predictors["split_role"].astype("string").reset_index(
            drop=True
        ).equals(expected_future["split_role"].astype("string").reset_index(drop=True))
    ):
        raise ValueError("Future metadata changed during predictor-only read")
    return pd.concat([frame, future_predictors], ignore_index=True, sort=False)


def _target_calendar(rows: pd.DataFrame) -> pd.DataFrame:
    """Create predictor-only calendar rows with no targets or outcomes."""
    columns = list(
        dict.fromkeys(
            [
                "part_id",
                "region",
                "week_start_date",
                *STATIC_FEATURES,
                *SAFE_CALENDAR_COLUMNS,
            ]
        )
    )
    calendar = rows.loc[:, [column for column in columns if column in rows]].copy()
    forbidden = [
        column
        for column in calendar
        if column in {"demand_qty", "demand_qty_capped", "is_missing_week"}
        or column.startswith("out_")
    ]
    if forbidden:
        raise ValueError(f"Forecast calendar contains forbidden columns: {forbidden}")
    return calendar


def _attach_actuals(
    forecast: pd.DataFrame, target_rows: pd.DataFrame
) -> pd.DataFrame:
    labels = target_rows[
        ["part_id", "region", "week_start_date", "demand_qty", "is_missing_week"]
    ].copy()
    labels["actual_qty"] = labels["demand_qty"].where(
        labels["is_missing_week"].eq(0) & labels["demand_qty"].notna()
    )
    labels = labels.drop(columns=["demand_qty", "is_missing_week"])
    return forecast.merge(
        labels,
        on=["part_id", "region", "week_start_date"],
        how="left",
        validate="one_to_one",
    )


def _origin_metadata(
    history: pd.DataFrame, target_rows: pd.DataFrame
) -> tuple[pd.DataFrame, dict[tuple[str, str], float]]:
    segments = build_segment_table(history)
    segment_columns = [
        "part_id",
        "region",
        "sbc_class",
        "abc_class",
        "xyz_class",
        "classification_status",
    ]
    segments = segments.loc[:, segment_columns]
    target_metadata = target_rows[
        ["part_id", "region", "part_category", "criticality"]
    ].drop_duplicates(["part_id", "region"])
    metadata = target_metadata.merge(
        segments,
        on=["part_id", "region"],
        how="left",
        validate="one_to_one",
    )
    for column in ("part_category", "criticality", "sbc_class", "abc_class", "xyz_class"):
        metadata[column] = metadata[column].astype("string").fillna("Unclassified")
    return metadata, naive_scale_by_series(history)


def _fit_final_test_forecasts(
    history: pd.DataFrame,
    target_rows: pd.DataFrame,
    scheduling: pd.DataFrame,
) -> pd.DataFrame:
    protocol_models = FROZEN_PROTOCOL["models"]
    origin = pd.Timestamp(FROZEN_PROTOCOL["evaluation"]["origin"])
    calendar = _target_calendar(target_rows)
    outputs = []
    for key in ("end_to_end_champion", "point_forecast_benchmark"):
        model = protocol_models[key]
        predictions, _, training_rows = fit_global_models(
            history,
            origin,
            calendar,
            model["parameters"],
            model["family"],
            int(model["seed"]),
            quantiles=tuple(model.get("quantiles", (0.5, 0.9, 0.95, 0.99))),
        )
        if len(predictions) != len(calendar):
            raise ValueError(f"Incomplete predictions from {model['candidate_id']}")
        predictions["candidate_id"] = model["candidate_id"]
        predictions["model"] = model["family"]
        predictions["config_id"] = model["config_id"]
        predictions["candidate_role"] = key
        predictions["training_rows"] = training_rows
        outputs.append(predictions)

    forecasts = pd.concat(outputs, ignore_index=True)
    average = (
        history.loc[
            history["split_role"].eq("train_val")
            & history["week_start_date"].le(origin)
            & history["demand_qty"].notna()
        ]
        .groupby(["part_id", "region"], observed=True)["demand_qty"]
        .mean()
    )
    average_rows = calendar[["part_id", "region", "week_start_date"]].copy()
    average_rows["forecast_qty"] = [
        float(average.get((part_id, region), np.nan))
        for part_id, region in zip(
            average_rows["part_id"], average_rows["region"]
        )
    ]
    for column in ("forecast_p90", "forecast_p95", "forecast_p99"):
        average_rows[column] = np.nan
    average_rows["candidate_id"] = "historical_average::origin_mean"
    average_rows["model"] = "historical_average"
    average_rows["config_id"] = "origin_mean"
    average_rows["candidate_role"] = "traditional_baseline_forecast"
    average_rows["training_rows"] = np.nan
    forecasts = pd.concat([forecasts, average_rows], ignore_index=True)
    forecasts["origin_date"] = origin
    forecasts["horizon_step"] = (
        (pd.to_datetime(forecasts["week_start_date"]) - origin).dt.days // 7
    ).astype(int)
    if not forecasts["horizon_step"].between(
        1, int(FROZEN_PROTOCOL["evaluation"]["horizon_weeks"])
    ).all():
        raise ValueError("Forecast rows fall outside the frozen test horizon")
    forecasts = _attach_actuals(forecasts, target_rows)
    metadata, scales = _origin_metadata(history, target_rows)
    forecasts = forecasts.merge(
        metadata, on=["part_id", "region"], how="left", validate="many_to_one"
    )
    forecasts["mase_scale"] = [
        scales.get((str(part_id), str(region)), np.nan)
        for part_id, region in zip(forecasts["part_id"], forecasts["region"])
    ]
    lead_times = _lead_time_map(scheduling)
    forecasts = forecasts.merge(
        lead_times, on="part_id", how="left", validate="many_to_one"
    )
    forecasts["evaluation_split"] = "final_test"
    forecasts["ablation_group"] = "none"
    forecasts["runtime_seconds"] = np.nan
    return forecasts


def _window_band(step: int) -> str:
    return "weeks_1_14" if step <= 14 else "weeks_15_26"


def _metric_record(
    group: pd.DataFrame,
    *,
    level: str,
    horizon_band: str,
    group_dimension: str,
    group_value: str,
) -> dict[str, Any]:
    actual = pd.to_numeric(group["actual_qty"], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    forecast = pd.to_numeric(group["forecast_qty"], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    quantiles = {
        level: pd.to_numeric(group[column], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        for level, column in (
            (0.90, "forecast_p90"),
            (0.95, "forecast_p95"),
            (0.99, "forecast_p99"),
        )
        if group[column].notna().any()
    }
    metrics = forecast_metrics(
        actual,
        forecast,
        mase_scale=pd.to_numeric(group["mase_scale"], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        ),
        quantile_forecasts=quantiles,
    )
    first = group.iloc[0]
    record: dict[str, Any] = {
        "candidate_id": first["candidate_id"],
        "model": first["model"],
        "config_id": first["config_id"],
        "level": level,
        "horizon_band": horizon_band,
        "group_dimension": group_dimension,
        "group_value": group_value,
        "n_rows": int(len(group)),
        "n_scored": int(metrics["n_scored"]),
        "n_missing_actual": int(group["actual_qty"].isna().sum()),
        "n_zero_actual": int(group["actual_qty"].eq(0).sum()),
        "sample_status": (
            "insufficient_sample" if int(metrics["n_scored"]) == 0 else "scored"
        ),
    }
    record.update(metrics)
    for quantile, column in (
        (90, "forecast_p90"),
        (95, "forecast_p95"),
        (99, "forecast_p99"),
    ):
        record[f"p{quantile}_n"] = int(
            (
                group["actual_qty"].notna()
                & pd.to_numeric(group[column], errors="coerce").notna()
            ).sum()
        )
    return record


def _group_metric_rows(
    frame: pd.DataFrame, level: str, horizon_band: str, *, include_horizon_step: bool
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dimensions = [
        "part_id",
        "part_category",
        "region",
        "criticality",
        "sbc_class",
    ]
    if include_horizon_step:
        dimensions.append("horizon_step")
    for (candidate_id, _model), candidate in frame.groupby(
        ["candidate_id", "model"], sort=False, observed=True
    ):
        rows.append(
            _metric_record(
                candidate,
                level=level,
                horizon_band=horizon_band,
                group_dimension="overall",
                group_value="all",
            )
        )
        for dimension in dimensions:
            for value, group in candidate.groupby(
                dimension, dropna=False, sort=True, observed=True
            ):
                rows.append(
                    _metric_record(
                        group,
                        level=level,
                        horizon_band=horizon_band,
                        group_dimension=dimension,
                        group_value=str(value),
                    )
                )
    return rows


def _forecast_metric_table(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    forecasts = forecasts.copy()
    forecasts["horizon_band"] = forecasts["horizon_step"].map(_window_band)
    for horizon_band in ("all_26_weeks", "weeks_1_14", "weeks_15_26"):
        selected = (
            forecasts
            if horizon_band == "all_26_weeks"
            else forecasts.loc[forecasts["horizon_band"].eq(horizon_band)]
        )
        rows.extend(
            _group_metric_rows(
                selected, "weekly", horizon_band, include_horizon_step=True
            )
        )
    lead_rows = _lead_time_rows(forecasts, horizon=26)
    if not lead_rows.empty:
        rows.extend(
            _group_metric_rows(
                lead_rows,
                "lead_time_from_origin",
                "lead_time_from_origin",
                include_horizon_step=False,
            )
        )
    metrics = pd.DataFrame(rows)
    baseline = metrics.loc[
        metrics["model"].eq("historical_average"), [
            "level", "horizon_band", "group_dimension", "group_value",
            "wape_pct", "mae", "rmse", "smape_pct", "mase",
        ]
    ].rename(
        columns={
            name: f"baseline_{name}"
            for name in ("wape_pct", "mae", "rmse", "smape_pct", "mase")
        }
    )
    metrics = metrics.merge(
        baseline,
        on=["level", "horizon_band", "group_dimension", "group_value"],
        how="left",
        validate="many_to_one",
    )
    for metric in ("wape_pct", "mae", "rmse", "smape_pct", "mase"):
        metrics[f"worse_than_historical_average_{metric}"] = (
            metrics["model"].ne("historical_average")
            & metrics[metric].gt(metrics[f"baseline_{metric}"])
        )
    return metrics


def _schedule_paths(
    forecasts: pd.DataFrame, parts: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    champion_id = FROZEN_PROTOCOL["models"]["end_to_end_champion"]["candidate_id"]
    point = forecasts.loc[forecasts["candidate_id"].eq(champion_id)]
    average = forecasts.loc[
        forecasts["candidate_id"].eq("historical_average::origin_mean")
    ]
    joined = point[
        [
            "part_id", "region", "week_start_date", "horizon_step",
            "forecast_qty", "forecast_p90", "forecast_p95", "forecast_p99",
            "actual_qty",
        ]
    ].merge(
        average[
            ["part_id", "region", "week_start_date", "forecast_qty"]
        ].rename(columns={"forecast_qty": "historical_average_qty"}),
        on=["part_id", "region", "week_start_date"],
        how="left",
        validate="one_to_one",
    )

    def complete_sum(values: pd.Series) -> float:
        return float(values.sum(min_count=len(values)))

    grouped = (
        joined.groupby(
            ["part_id", "week_start_date", "horizon_step"],
            as_index=False,
            sort=True,
            observed=True,
        )
        .agg(
            forecast_qty=("forecast_qty", complete_sum),
            forecast_p90=("forecast_p90", complete_sum),
            forecast_p95=("forecast_p95", complete_sum),
            forecast_p99=("forecast_p99", complete_sum),
            historical_average_qty=("historical_average_qty", complete_sum),
            actual_qty=("actual_qty", complete_sum),
        )
        .rename(columns={"horizon_step": "week"})
    )
    grouped["origin_date"] = pd.Timestamp(
        FROZEN_PROTOCOL["evaluation"]["origin"]
    )
    paths: dict[str, pd.DataFrame] = {}
    exclusions: list[dict[str, str]] = []
    for part_id, group in grouped.groupby("part_id", sort=True, observed=True):
        key = str(part_id)
        reasons = []
        if key not in parts:
            reasons.append("missing_D2_scheduling_snapshot")
        if len(group) != 26 or set(group["week"].astype(int)) != set(range(1, 27)):
            reasons.append("incomplete_26_week_part_path")
        for column in (
            "actual_qty", "forecast_qty", "forecast_p90", "forecast_p95",
            "forecast_p99", "historical_average_qty",
        ):
            if group[column].isna().any() or not np.isfinite(
                pd.to_numeric(group[column], errors="coerce").to_numpy(
                    dtype=float, na_value=np.nan
                )
            ).all():
                reasons.append(f"incomplete_{column}")
        if reasons:
            exclusions.append(
                {"part_id": key, "exclusion_reason": "|".join(sorted(set(reasons)))}
            )
            continue
        paths[key] = group.sort_values("week", kind="stable").reset_index(drop=True)
    return paths, pd.DataFrame(exclusions, columns=["part_id", "exclusion_reason"])


def _run_phase7_schedulers(
    root: Path,
    forecasts: pd.DataFrame,
    project_config: dict[str, Any],
    scheduling: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    origin = pd.Timestamp(FROZEN_PROTOCOL["evaluation"]["origin"])
    parts = {
        str(row["part_id"]): row
        for row in scheduling.to_dict(orient="records")
    }
    paths, exclusions = _schedule_paths(forecasts, parts)
    histories = load_origin_visible_history(root, [origin])
    for part_id in list(paths):
        history = histories.get((origin.date().isoformat(), part_id))
        if history is None or int(history["complete_observed_weeks"]) < 8:
            exclusions.loc[len(exclusions)] = {
                "part_id": part_id,
                "exclusion_reason": "insufficient_origin_visible_history",
            }
            del paths[part_id]

    schedule_rows: list[dict[str, Any]] = []
    order_frames: list[pd.DataFrame] = []
    simulation_frames: list[pd.DataFrame] = []
    realized_frames: list[pd.DataFrame] = []
    policy_specs = (
        (
            FROZEN_PROTOCOL["models"]["end_to_end_champion"]["candidate_id"],
            FROZEN_PROTOCOL["scheduling"]["winning_policy"],
        ),
        (
            "historical_average::origin_mean",
            FROZEN_PROTOCOL["scheduling"]["traditional_baseline_policy"],
        ),
    )
    seed = int(project_config["scheduling"]["simulator_assumptions"]["seed"])
    for candidate_id, scheduler_id in policy_specs:
        capacity: dict[Any, float] = {}
        policy_order_frames: list[pd.DataFrame] = []
        policy_rows: list[dict[str, Any]] = []
        for part_id, path in paths.items():
            part = parts[part_id]
            plan_path = path.copy()
            if scheduler_id == "traditional_baseline":
                plan_path["forecast_qty"] = plan_path["historical_average_qty"]
            row, orders, simulation, checker, capacity = _run_plan(
                candidate_id,
                scheduler_id,
                PRIMARY_INITIALIZATION,
                plan_path,
                part,
                histories,
                project_config,
                capacity,
                horizon=26,
                run_id_prefix="phase7",
            )
            row["independent_checker_status"] = (
                "firm" if checker["feasible"] else "unresolved"
            )
            row["checker_failed_checks"] = "|".join(checker["failed_checks"])
            row["checker_hard_failure"] = bool(
                HARD_CONSTRAINTS.intersection(checker["failed_checks"])
            )
            row["global_checker_status"] = "pending"
            policy_rows.append(row)

            if not orders.empty:
                orders = orders.copy()
                orders["status"] = row["status"]
                orders["independent_checker_status"] = row[
                    "independent_checker_status"
                ]
                orders["independent_checker_failed_checks"] = row[
                    "checker_failed_checks"
                ]
                orders["scheduler_id"] = scheduler_id
                orders["candidate_id"] = candidate_id
                if row["status"] == "unresolved":
                    orders["failed_checks"] = row["failed_checks"]
                policy_order_frames.append(orders)

                snapshot = _initial_snapshot(
                    part, origin, histories, PRIMARY_INITIALIZATION
                )
                realized = realize_supplier_orders(
                    orders,
                    _supplier_snapshot(snapshot),
                    seed=seed,
                )
                realized["scheduler_id"] = scheduler_id
                realized["candidate_id"] = candidate_id
                realized_frames.append(realized)

            if not simulation.empty:
                sim = simulation.merge(
                    plan_path[
                        ["week", "week_start_date", "forecast_qty", "actual_qty"]
                    ],
                    on="week",
                    how="left",
                    validate="one_to_one",
                )
                sim["scheduler_id"] = scheduler_id
                sim["candidate_id"] = candidate_id
                sim["horizon_band"] = sim["week"].map(
                    lambda value: _window_band(int(value))
                )
                simulation_frames.append(sim)

        policy_orders = (
            pd.concat(policy_order_frames, ignore_index=True)
            if policy_order_frames
            else pd.DataFrame()
        )
        firm_orders = (
            policy_orders.loc[policy_orders["status"].eq("firm")]
            if not policy_orders.empty
            else policy_orders
        )
        global_checker = check_schedule(
            firm_orders,
            {
                part_id: _checker_parts(
                    part, scheduler_id == "risk_adjusted_mrp"
                )[part_id]
                for part_id, part in parts.items()
            },
            horizon_weeks=26,
            check_requirements=False,
        )
        global_failures = set(global_checker["failed_checks"])
        for row in policy_rows:
            row["global_checker_status"] = (
                "firm" if global_checker["feasible"] else "unresolved"
            )
            row["global_checker_failed_checks"] = "|".join(
                sorted(global_failures)
            )
            if row["status"] == "firm" and not global_checker["feasible"]:
                row["status"] = "unresolved"
                row["failed_checks"] = "|".join(
                    sorted(set(filter(None, str(row["failed_checks"]).split("|")))
                           | global_failures)
                )
        if not policy_orders.empty:
            status_by_run = {
                row["run_id"]: row["status"] for row in policy_rows
            }
            policy_orders["status"] = policy_orders["run_id"].map(status_by_run)
            if not global_checker["feasible"]:
                policy_orders.loc[
                    policy_orders["status"].eq("unresolved"), "failed_checks"
                ] = "|".join(sorted(global_failures))
            order_frames.append(policy_orders)
        schedule_rows.extend(policy_rows)

    checks = pd.DataFrame(schedule_rows)
    orders_out = (
        pd.concat(order_frames, ignore_index=True)
        if order_frames
        else pd.DataFrame()
    )
    simulation_out = (
        pd.concat(simulation_frames, ignore_index=True)
        if simulation_frames
        else pd.DataFrame()
    )
    realized_out = (
        pd.concat(realized_frames, ignore_index=True)
        if realized_frames
        else pd.DataFrame()
    )
    return checks, orders_out, simulation_out, exclusions, realized_out


SCHEDULER_KPI_DEFINITIONS: dict[str, tuple[str, str, str]] = {
    "forecast_accuracy_pct": (
        "100 - 100 * sum(abs(forecast - actual)) / sum(actual)",
        "percent",
        "matched complete part-weeks",
    ),
    "forecast_vs_actual_variance": (
        "sum(forecast - actual)",
        "units",
        "matched complete part-weeks",
    ),
    "forecast_bias_pct": (
        "100 * sum(forecast - actual) / sum(actual)",
        "percent",
        "actual demand units",
    ),
    "stock_out_pct": (
        "part-weeks with positive closing shortage / eligible part-weeks",
        "percent",
        "simulated part-weeks",
    ),
    "fill_rate_pct": (
        "current demand fulfilled / actual demand",
        "percent",
        "actual demand units",
    ),
    "schedule_adherence_pct": (
        "received units arriving by planned week / planned order units",
        "percent",
        "planned order units released in the horizon band",
    ),
    "emergency_procurement_reduction": (
        "traditional baseline emergency cost proxy - policy emergency cost proxy",
        "currency",
        "simulated emergency units x unit cost x configured multiplier",
    ),
    "excess_inventory_reduction": (
        "traditional baseline excess inventory units - policy excess units",
        "units",
        "simulated part-weeks",
    ),
    "carrying_cost_reduction": (
        "traditional baseline carrying cost proxy - policy carrying cost proxy",
        "currency",
        "simulated part-weeks",
    ),
    "inventory_turnover": (
        "period COGS / average weekly closing inventory value",
        "ratio",
        "simulated period inventory value",
    ),
}


def _primitive_scheduler_metrics(
    simulation: pd.DataFrame, realized_orders: pd.DataFrame, band: str
) -> dict[str, float]:
    selected = simulation.loc[simulation["horizon_band"].eq(band)]
    if selected.empty:
        return {}
    actual = pd.to_numeric(selected["demand_qty"], errors="coerce").to_numpy(
        dtype=float
    )
    predicted = pd.to_numeric(selected["forecast_qty"], errors="coerce").to_numpy(
        dtype=float
    )
    errors = predicted - actual
    actual_total = float(actual.sum())
    absolute_total = float(np.abs(errors).sum())
    base: dict[str, float] = {
        "forecast_accuracy_pct": (
            100.0 * (1.0 - absolute_total / actual_total)
            if actual_total > 0 else np.nan
        ),
        "forecast_vs_actual_variance": float(errors.sum()),
        "forecast_bias_pct": (
            100.0 * float(errors.sum()) / actual_total
            if actual_total > 0 else np.nan
        ),
        "stock_out_pct": float(selected["stockout_at_close_flag"].mean() * 100.0),
        "fill_rate_pct": (
            100.0
            * float(selected["current_demand_fulfilled_qty"].sum())
            / actual_total
            if actual_total > 0 else np.nan
        ),
        "emergency_order_count": float(selected["emergency_order_qty"].gt(0).sum()),
        "emergency_cost_proxy": float(selected["emergency_cost_proxy"].sum()),
        "excess_inventory_units": float(selected["excess_inventory_units"].sum()),
        "carrying_cost_proxy": float(selected["carrying_cost_proxy"].sum()),
    }
    weekly_inventory_value = (
        selected.assign(
            inventory_value=selected["closing_stock"] * selected["unit_cost"]
        )
        .groupby("week", observed=True)["inventory_value"]
        .sum()
    )
    cogs = float((selected["demand_qty"] * selected["unit_cost"]).sum())
    average_inventory = (
        float(weekly_inventory_value.mean())
        if not weekly_inventory_value.empty else 0.0
    )
    base["inventory_turnover"] = (
        cogs / average_inventory if average_inventory > 0 else np.nan
    )
    if realized_orders.empty:
        base["schedule_adherence_pct"] = np.nan
    else:
        selected_orders = (
            realized_orders
            if band == "all_26_weeks"
            else realized_orders.loc[
                realized_orders["requested_week"].map(
                    lambda value: _window_band(int(value))
                ).eq(band)
            ]
        )
        ordered = float(selected_orders["order_qty"].sum())
        on_time = float(
            selected_orders.loc[
                selected_orders["realized_arrival_week"].le(
                    selected_orders["arrival_week"]
                ),
                "received_qty",
            ].sum()
        )
        base["schedule_adherence_pct"] = (
            100.0 * on_time / ordered if ordered > 0 else np.nan
        )
    return base


def _scheduler_kpi_table(
    simulation: pd.DataFrame, realized_orders: pd.DataFrame
) -> pd.DataFrame:
    if simulation.empty:
        return pd.DataFrame()
    primitive: dict[tuple[str, str], dict[str, float]] = {}
    for (scheduler_id, _candidate), group in simulation.groupby(
        ["scheduler_id", "candidate_id"], observed=True
    ):
        for band in ("weeks_1_14", "weeks_15_26", "all_26_weeks"):
            if band == "all_26_weeks":
                group = group.assign(horizon_band="all_26_weeks")
            orders = (
                realized_orders.loc[
                    realized_orders["scheduler_id"].eq(scheduler_id)
                ]
                if not realized_orders.empty else pd.DataFrame()
            )
            primitive[(str(scheduler_id), band)] = _primitive_scheduler_metrics(
                group, orders, band
            )
    base_policy = "traditional_baseline"
    records: list[dict[str, Any]] = []
    for band in ("weeks_1_14", "weeks_15_26", "all_26_weeks"):
        baseline = primitive.get((base_policy, band), {})
        for scheduler_id in ("traditional_baseline", "risk_adjusted_mrp"):
            values = primitive.get((scheduler_id, band), {})
            for name in SCHEDULER_KPI_DEFINITIONS:
                formula, unit, basis = SCHEDULER_KPI_DEFINITIONS[name]
                if name.endswith("_reduction"):
                    source = {
                        "emergency_procurement_reduction": "emergency_cost_proxy",
                        "excess_inventory_reduction": "excess_inventory_units",
                        "carrying_cost_reduction": "carrying_cost_proxy",
                    }[name]
                    value = (
                        0.0
                        if scheduler_id == base_policy
                        else baseline.get(source, np.nan) - values.get(source, np.nan)
                    )
                else:
                    value = values.get(name, np.nan)
                baseline_value = (
                    0.0 if name.endswith("_reduction")
                    else baseline.get(name, np.nan)
                )
                records.append(
                    {
                        "candidate_id": (
                            "historical_average::origin_mean"
                            if scheduler_id == base_policy
                            else FROZEN_PROTOCOL["models"]["end_to_end_champion"][
                                "candidate_id"
                            ]
                        ),
                        "scheduler_id": scheduler_id,
                        "horizon_band": band,
                        "kpi": name,
                        "formula": formula,
                        "unit": unit,
                        "basis": basis,
                        "value": value,
                        "baseline_value": baseline_value,
                        "difference": value - baseline_value
                        if pd.notna(value) and pd.notna(baseline_value)
                        else np.nan,
                        "worse_than_baseline": (
                            bool(value < baseline_value)
                            if name.endswith("_reduction")
                            and pd.notna(value)
                            and pd.notna(baseline_value)
                            else bool(value > baseline_value)
                            if name in {
                                "stock_out_pct",
                                "emergency_order_count",
                                "emergency_cost_proxy",
                                "excess_inventory_units",
                                "carrying_cost_proxy",
                            }
                            and pd.notna(value)
                            and pd.notna(baseline_value)
                            else bool(value < baseline_value)
                            if name in {
                                "forecast_accuracy_pct",
                                "fill_rate_pct",
                                "schedule_adherence_pct",
                                "inventory_turnover",
                            }
                            and pd.notna(value)
                            and pd.notna(baseline_value)
                            else False
                        ),
                    }
                )
    return pd.DataFrame(records)


def _check_jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    raise TypeError(f"Value is not JSON serializable: {type(value).__name__}")


def _write_new_csv(root: Path, relative: str, frame: pd.DataFrame) -> None:
    path = root / relative
    with path.open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False)


def _write_new_json(root: Path, relative: str, value: Mapping[str, Any]) -> None:
    path = root / relative
    with path.open("x", encoding="utf-8", newline="") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=_check_jsonable)
        stream.write("\n")


def _claim_attempt(root: Path, metadata: Mapping[str, Any]) -> None:
    marker = root / ATTEMPT_MARKER
    attempt = {
        **metadata,
        "status": "attempt_started",
        "target_access_started": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with marker.open("x", encoding="utf-8", newline="") as stream:
        json.dump(attempt, stream, indent=2, sort_keys=True, default=_check_jsonable)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _update_attempt(root: Path, **updates: Any) -> None:
    marker = root / ATTEMPT_MARKER
    current = json.loads(marker.read_text(encoding="utf-8"))
    current.update(updates)
    temporary = marker.with_suffix(".tmp")
    with temporary.open("x", encoding="utf-8", newline="") as stream:
        json.dump(current, stream, indent=2, sort_keys=True, default=_check_jsonable)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, marker)


def _phase7_report(
    metadata: Mapping[str, Any],
    metrics: pd.DataFrame,
    checks: pd.DataFrame,
    exclusions: pd.DataFrame,
) -> str:
    firm = int(checks["status"].eq("firm").sum()) if not checks.empty else 0
    unresolved = int(checks["status"].eq("unresolved").sum()) if not checks.empty else 0
    hard_failures = (
        int(checks["checker_hard_failure"].fillna(False).sum())
        if not checks.empty else 0
    )
    return "\n".join(
        [
            "# Phase 7 final-test evaluation",
            "",
            "**Basis:** method evaluated on synthetic data; this is not evidence of factory savings.",
            "",
            f"- Protocol: `{metadata['protocol_id']}`.",
            f"- Origin/window: {FROZEN_PROTOCOL['evaluation']['origin']} / "
            f"{FROZEN_PROTOCOL['evaluation']['start']}–{FROZEN_PROTOCOL['evaluation']['end']} "
            f"({FROZEN_PROTOCOL['evaluation']['horizon_weeks']} weeks).",
            "- Reported forecast horizon bands: weeks 1–14 (primary) and 15–26 (extended diagnostic).",
            f"- End-to-end champion: `{metadata['champion_id']}` with "
            f"`{metadata['scheduler_id']}`.",
            f"- Track A point-forecast benchmark: `{metadata['point_benchmark_id']}`.",
            f"- Traditional baseline: `{metadata['baseline_id']}`.",
            f"- Schedule checks: {firm} firm paths, {unresolved} unresolved paths; "
            f"{hard_failures} checker hard-failure paths.",
            f"- Excluded schedule paths: {len(exclusions)}; see the versioned exclusions file.",
            "- Forecast metric rows preserve missing labels as missing; genuine zero demand remains scored.",
            "- Lead-time metrics are a single origin-anchored 26-week aggregate; the extended "
            "weeks 15–26 band is reported weekly and is not a second lead-time replenishment window.",
            "- No performance pass threshold is defined by the frozen protocol; results are reported, "
            "not used to reselect or tune a champion.",
            "- D2 is a single snapshot. Its date and use are recorded in run metadata; scheduling results "
            "are simulated policy comparisons, not historical operating-state evidence.",
            "",
            "Forecast metrics and segment-level comparisons are in "
            "`results/phase7_forecast_metrics_v1.csv`; scheduler KPIs with formula, unit, and basis are "
            "in `results/phase7_scheduler_kpis_v1.csv`.",
        ]
    )


def run_final_test_evaluation(
    root: Path = ROOT,
    *,
    authorized: bool = False,
    confirmation: str | None = None,
) -> dict[str, Any]:
    """Consume the final-test target at most once after explicit authorization."""
    if not authorized or confirmation != FINAL_TEST_CONFIRMATION:
        raise PermissionError(
            "Final-test evaluation requires explicit authorization and the exact confirmation phrase."
        )
    preflight = safe_preflight(root)
    if preflight["status"] != "safe_preflight_passed":
        raise RuntimeError(
            "Final-test evaluation blocked by safe preflight: "
            + "; ".join(preflight["blockers"])
        )
    hashes = {
        key: preflight[key]
        for key in (
            "protocol_id", "phase7_config_sha256", "project_config_sha256",
            "business_config_sha256", "phase6_report_sha256",
            "track_a_tuning_sha256", "split_manifest_sha256", "champion_id",
            "point_benchmark_id", "scheduler_id", "baseline_id",
        )
    }
    _claim_attempt(root, {**hashes, "run_id": FINAL_TEST_RUN_ID})
    try:
        _update_attempt(
            root,
            target_access_started=True,
            target_access_started_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        demand = _read_demand_for_authorized_operation(
            root,
            expected_split_manifest_sha256=preflight["split_manifest_sha256"],
        )
        origin = pd.Timestamp(FROZEN_PROTOCOL["evaluation"]["origin"])
        test_start = pd.Timestamp(FROZEN_PROTOCOL["evaluation"]["start"])
        test_end = pd.Timestamp(FROZEN_PROTOCOL["evaluation"]["end"])
        history = demand.loc[
            demand["split_role"].isin(["warmup", "train_val"])
            & demand["week_start_date"].le(origin)
        ].copy()
        target_rows = demand.loc[
            demand["split_role"].eq("final_test")
            & demand["week_start_date"].between(test_start, test_end)
        ].copy()
        if target_rows.empty:
            raise ValueError("No rows found in the frozen final-test window")
        scheduling = load_scheduling(root / "data" / "02_scheduling_inputs.csv")
        forecasts = _fit_final_test_forecasts(history, target_rows, scheduling)
        metrics = _forecast_metric_table(forecasts)
        project_config = _read_yaml(root / PROJECT_CONFIG_PATH)
        checks, orders, simulation, schedule_exclusions, realized = (
            _run_phase7_schedulers(root, forecasts, project_config, scheduling)
        )
        exclusions = pd.concat(
            [
                schedule_exclusions.assign(exclusion_scope="scheduling"),
                pd.DataFrame(
                    columns=["part_id", "exclusion_reason", "exclusion_scope"]
                ),
            ],
            ignore_index=True,
        )
        kpis = _scheduler_kpi_table(simulation, realized)
        run_metadata = {
            **hashes,
            "run_id": FINAL_TEST_RUN_ID,
            "started_at_utc": json.loads(
                (root / ATTEMPT_MARKER).read_text(encoding="utf-8")
            )["started_at_utc"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "completed",
            "target_access_started": True,
            "final_test_targets_consumed": True,
            "evaluation_origin": FROZEN_PROTOCOL["evaluation"]["origin"],
            "evaluation_start": FROZEN_PROTOCOL["evaluation"]["start"],
            "evaluation_end": FROZEN_PROTOCOL["evaluation"]["end"],
            "horizon_weeks": 26,
            "primary_horizon_weeks": 14,
            "extended_horizon_weeks": 12,
            "champion_id": FROZEN_PROTOCOL["models"]["end_to_end_champion"][
                "candidate_id"
            ],
            "point_benchmark_id": FROZEN_PROTOCOL["models"][
                "point_forecast_benchmark"
            ]["candidate_id"],
            "scheduler_id": FROZEN_PROTOCOL["scheduling"]["winning_policy"],
            "baseline_id": (
                "historical_average::traditional_baseline"
            ),
            "split_manifest": preflight["split_manifest"],
            "d2_snapshot_as_of_date": preflight["d2_snapshot_as_of_date"],
            "forecast_rows": int(len(forecasts)),
            "forecast_metric_rows": int(len(metrics)),
            "schedule_check_rows": int(len(checks)),
            "firm_schedule_paths": int(checks["status"].eq("firm").sum()),
            "unresolved_schedule_paths": int(
                checks["status"].eq("unresolved").sum()
            ),
            "independent_checker_hard_failure_paths": int(
                checks["checker_hard_failure"].fillna(False).sum()
            ),
            "schedule_exclusions": int(len(schedule_exclusions)),
        }
        report = _phase7_report(run_metadata, metrics, checks, exclusions)
        _write_new_csv(root, EVALUATION_OUTPUTS[0], forecasts)
        _write_new_csv(root, EVALUATION_OUTPUTS[1], metrics)
        _write_new_csv(root, EVALUATION_OUTPUTS[2], checks)
        _write_new_csv(root, EVALUATION_OUTPUTS[3], orders)
        _write_new_csv(root, EVALUATION_OUTPUTS[4], simulation)
        _write_new_csv(root, EVALUATION_OUTPUTS[5], kpis)
        _write_new_csv(root, EVALUATION_OUTPUTS[6], exclusions)
        report_path = root / EVALUATION_OUTPUTS[7]
        with report_path.open("x", encoding="utf-8", newline="") as stream:
            stream.write(report + "\n")
        _write_new_json(root, EVALUATION_OUTPUTS[8], run_metadata)
        _update_attempt(
            root,
            status="completed",
            completed_at_utc=run_metadata["completed_at_utc"],
            run_metadata_path=EVALUATION_OUTPUTS[8],
            final_test_targets_consumed=True,
        )
        return run_metadata
    except Exception as error:
        _update_attempt(
            root,
            status="failed_consumed",
            failed_at_utc=datetime.now(timezone.utc).isoformat(),
            error_type=type(error).__name__,
            error_message=str(error),
        )
        raise


def run_finalization(
    root: Path = ROOT,
    *,
    authorized: bool = False,
    confirmation: str | None = None,
) -> dict[str, Any]:
    """Retrain after the one-time evaluation and build the final 14-week schedule."""
    if not authorized or confirmation != FINALIZE_CONFIRMATION:
        raise PermissionError(
            "Finalization requires explicit authorization and the exact confirmation phrase."
        )
    expected_hashes = _validate_completed_evaluation(root)
    if not _phase5_is_closed(root):
        raise RuntimeError("Finalization is blocked by the Phase 5 status/evidence gate")
    existing = [
        path for path in FINALIZATION_OUTPUTS if (root / path).exists()
    ]
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite finalization outputs: {existing}"
        )
    demand = _read_demand_for_authorized_operation(
        root,
        expected_split_manifest_sha256=expected_hashes["split_manifest_sha256"],
    )
    origin = pd.Timestamp(FROZEN_PROTOCOL["future"]["start"]) - pd.Timedelta(weeks=1)
    end = pd.Timestamp(FROZEN_PROTOCOL["evaluation"]["end"])
    training = demand.loc[
        demand["split_role"].isin(["warmup", "train_val", "final_test"])
        & demand["week_start_date"].le(end)
    ].copy()
    future_rows = demand.loc[
        demand["split_role"].eq("future")
        & demand["week_start_date"].between(
            pd.Timestamp(FROZEN_PROTOCOL["future"]["start"]),
            pd.Timestamp(FROZEN_PROTOCOL["future"]["end"]),
        )
    ].copy()
    calendar = _target_calendar(future_rows)
    model = FROZEN_PROTOCOL["models"]["end_to_end_champion"]
    predictions, _, training_rows = fit_global_models(
        training,
        origin,
        calendar,
        model["parameters"],
        model["family"],
        int(model["seed"]),
        quantiles=tuple(model["quantiles"]),
        fit_split_roles=("warmup", "train_val", "final_test"),
    )
    predictions["model"] = model["family"]
    predictions["config_id"] = model["config_id"]
    predictions["candidate_id"] = model["candidate_id"]
    predictions["candidate_role"] = "end_to_end_champion"
    predictions["forecast_target_step"] = (
        (pd.to_datetime(predictions["week_start_date"]) - origin).dt.days // 7
    ).astype(int)
    if (
        len(predictions) != 14
        or pd.to_datetime(predictions["week_start_date"]).min()
        != pd.Timestamp(FROZEN_PROTOCOL["future"]["start"])
        or pd.to_datetime(predictions["week_start_date"]).max()
        != pd.Timestamp(FROZEN_PROTOCOL["future"]["end"])
    ):
        raise ValueError("Final future forecast does not match the frozen 14-week dates")
    scheduling = load_scheduling(root / "data" / "02_scheduling_inputs.csv")
    parts = {
        str(row["part_id"]): row
        for row in scheduling.to_dict(orient="records")
    }
    project_config = _read_yaml(root / PROJECT_CONFIG_PATH)
    schedule = _build_provisional_schedule(
        predictions,
        model["candidate_id"],
        FROZEN_PROTOCOL["scheduling"]["winning_policy"],
        parts,
        project_config,
        root,
    )
    schedule["forecast_status"] = "FINAL"
    checker = schedule.attrs.get("independent_checker", {})
    metadata = {
        "protocol_id": FROZEN_PROTOCOL["protocol_id"],
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model["candidate_id"],
        "scheduler_id": FROZEN_PROTOCOL["scheduling"]["winning_policy"],
        "training_rows": training_rows,
        "training_label_end": str(end.date()),
        "forecast_start": FROZEN_PROTOCOL["future"]["start"],
        "forecast_end": FROZEN_PROTOCOL["future"]["end"],
        "forecast_rows": int(len(predictions)),
        "schedule_rows": int(len(schedule)),
        "schedule_firm_rows": int(schedule["status"].eq("firm").sum()),
        "schedule_unresolved_rows": int(schedule["status"].eq("unresolved").sum()),
        "independent_checker": checker,
        "phase7_config_sha256": _sha256(root / PROTOCOL_PATH),
        "project_config_sha256": _sha256(root / PROJECT_CONFIG_PATH),
        "business_config_sha256": _sha256(root / BUSINESS_CONFIG_PATH),
    }
    forecast_output = predictions.copy()
    forecast_output["forecast_status"] = "FINAL"
    _write_new_csv(root, FINALIZATION_OUTPUTS[0], forecast_output)
    _write_new_csv(root, FINALIZATION_OUTPUTS[1], schedule)
    _write_new_json(root, FINALIZATION_OUTPUTS[2], metadata)
    return metadata


def _validate_completed_evaluation(root: Path) -> dict[str, Any]:
    marker_path = root / ATTEMPT_MARKER
    if not marker_path.is_file():
        raise RuntimeError("Finalization requires a completed final-test attempt")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("Final-test attempt marker is malformed") from error
    if not isinstance(marker, dict):
        raise ValueError("Final-test attempt marker must be a JSON object")

    if (
        marker.get("status") != "completed"
        or marker.get("target_access_started") is not True
        or marker.get("final_test_targets_consumed") is not True
        or marker.get("run_id") != FINAL_TEST_RUN_ID
        or marker.get("run_metadata_path") != EVALUATION_OUTPUTS[8]
    ):
        raise RuntimeError(
            "Finalization requires a fully consumed, completed Phase 7 evaluation."
        )

    metadata_path = root / EVALUATION_OUTPUTS[8]
    if not metadata_path.is_file():
        raise FileNotFoundError("Completed Phase 7 evaluation metadata is missing")
    try:
        evaluation = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("Completed Phase 7 evaluation metadata is malformed") from error
    if not isinstance(evaluation, dict):
        raise ValueError("Completed Phase 7 evaluation metadata must be a JSON object")
    if (
        evaluation.get("status") != "completed"
        or evaluation.get("target_access_started") is not True
        or evaluation.get("final_test_targets_consumed") is not True
    ):
        raise RuntimeError("Saved evaluation metadata does not record a completed test")

    current = validate_protocol(root)
    metadata = _read_split_metadata(root / "data" / "01_demand_forecasting.csv")
    split = _validate_split_metadata(metadata)
    expected = {
        **current,
        "split_manifest_sha256": split["split_manifest_sha256"],
        "run_id": FINAL_TEST_RUN_ID,
    }
    for key, value in expected.items():
        if marker.get(key) != value or evaluation.get(key) != value:
            raise ValueError(
                f"Completed evaluation {key} does not match the frozen current value"
            )
    return expected


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    commands.add_parser("validate", help="Validate the frozen protocol and handoff")
    commands.add_parser("preflight", help="Run target-free metadata and output preflight")
    evaluate = commands.add_parser("evaluate", help="Single-use final-test evaluation")
    evaluate.add_argument("--authorize-final-test", action="store_true")
    evaluate.add_argument("--confirmation")
    finalize = commands.add_parser("finalize", help="Post-evaluation future forecast")
    finalize.add_argument("--authorize-finalization", action="store_true")
    finalize.add_argument("--confirmation")
    args = parser.parse_args()
    if args.operation == "validate":
        result = validate_protocol()
    elif args.operation == "preflight":
        result = safe_preflight()
    elif args.operation == "evaluate":
        result = run_final_test_evaluation(
            authorized=args.authorize_final_test,
            confirmation=args.confirmation,
        )
    else:
        result = run_finalization(
            authorized=args.authorize_finalization,
            confirmation=args.confirmation,
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=_check_jsonable))


if __name__ == "__main__":
    _main()
