from __future__ import annotations

import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.phase7 import (
    ATTEMPT_MARKER,
    EVALUATION_OUTPUTS,
    FINALIZE_CONFIRMATION,
    FINAL_TEST_CONFIRMATION,
    FINAL_TEST_RUN_ID,
    FROZEN_PROTOCOL,
    PHASE5_GATE_EVIDENCE_PATH,
    PHASE5_REQUIRED_EVIDENCE,
    _attach_actuals,
    _claim_attempt,
    _phase5_gate_blockers,
    _read_demand_for_authorized_operation,
    _read_split_metadata,
    _target_calendar,
    _update_attempt,
    _validate_completed_evaluation,
    _phase7_report,
    run_final_test_evaluation,
    run_finalization,
)
from src.forecasting.global_models import build_fold_training_data
from src.scheduling.phase6 import _run_plan


def test_preflight_metadata_reader_requests_no_target_or_outcome_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    source = data_dir / "01_demand_forecasting.csv"
    source.write_text(
        "week_start_date,split_role,demand_qty,out_spike_flag\n"
        "2026-03-30,final_test,987654,1\n",
        encoding="utf-8",
    )
    original = pd.read_csv
    requested = []

    def recording_read_csv(*args, **kwargs):
        if "usecols" in kwargs:
            requested.append(set(kwargs["usecols"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", recording_read_csv)
    metadata = _read_split_metadata(source)

    assert metadata.columns.tolist() == ["week_start_date", "split_role"]
    assert requested == [{"week_start_date", "split_role"}]
    assert "987654" not in metadata.to_string()


def test_forecast_calendar_strips_labels_outcomes_and_unverified_covariates():
    rows = pd.DataFrame(
        {
            "part_id": ["P1"],
            "region": ["East"],
            "week_start_date": [pd.Timestamp("2026-03-30")],
            "year": [2026],
            "promotion_flag": [1],
            "demand_qty": [987654],
            "out_spike_flag": [1],
        }
    )

    calendar = _target_calendar(rows)

    assert calendar.columns.tolist() == [
        "part_id", "region", "week_start_date", "year"
    ]


def test_missing_targets_stay_missing_and_genuine_zero_is_retained():
    forecast = pd.DataFrame(
        {
            "part_id": ["P1", "P1"],
            "region": ["East", "West"],
            "week_start_date": pd.to_datetime(["2026-03-30", "2026-03-30"]),
            "forecast_qty": [1.0, 0.0],
        }
    )
    targets = pd.DataFrame(
        {
            "part_id": ["P1", "P1"],
            "region": ["East", "West"],
            "week_start_date": pd.to_datetime(["2026-03-30", "2026-03-30"]),
            "demand_qty": pd.Series([0, pd.NA], dtype="Int64"),
            "is_missing_week": pd.Series([0, 1], dtype="Int64"),
        }
    )

    scored = _attach_actuals(forecast, targets).sort_values("region")

    assert scored.iloc[0]["actual_qty"] == 0
    assert pd.isna(scored.iloc[1]["actual_qty"])


def test_final_test_attempt_marker_is_exclusive(tmp_path: Path):
    (tmp_path / "results").mkdir()

    _claim_attempt(tmp_path, {"protocol_id": "test-protocol"})

    with pytest.raises(FileExistsError):
        _claim_attempt(tmp_path, {"protocol_id": "test-protocol"})


def test_concurrent_attempt_claims_allow_exactly_one_winner(tmp_path: Path):
    (tmp_path / "results").mkdir()

    def claim():
        try:
            _claim_attempt(tmp_path, {"protocol_id": "test-protocol"})
            return "claimed"
        except FileExistsError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: claim(), range(2)))

    assert results.count("claimed") == 1
    assert results.count("rejected") == 1


@pytest.mark.parametrize("attempt_status", ["attempt_started", "failed_consumed"])
def test_interrupted_or_failed_attempt_cannot_be_retried(
    tmp_path: Path, attempt_status: str
):
    (tmp_path / "results").mkdir()
    _claim_attempt(tmp_path, {"protocol_id": "test-protocol"})
    if attempt_status == "failed_consumed":
        _update_attempt(tmp_path, status="failed_consumed")

    with pytest.raises(FileExistsError):
        _claim_attempt(tmp_path, {"protocol_id": "test-protocol"})


def test_evaluation_requires_explicit_confirmation_before_any_data_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def fail_if_read(_root):
        raise AssertionError("target-bearing data read before authorization")

    monkeypatch.setattr(
        "src.evaluation.phase7._read_demand_for_authorized_operation", fail_if_read
    )
    with pytest.raises(PermissionError):
        run_final_test_evaluation(
            tmp_path,
            authorized=False,
            confirmation=FINAL_TEST_CONFIRMATION,
        )


def _completion_payload():
    current = {
        "protocol_id": FROZEN_PROTOCOL["protocol_id"],
        "phase7_config_sha256": "phase7-hash",
        "project_config_sha256": "project-hash",
        "business_config_sha256": "business-hash",
        "phase6_report_sha256": "phase6-hash",
        "track_a_tuning_sha256": "tuning-hash",
        "champion_id": "lightgbm_quantile::random_1",
        "point_benchmark_id": "lightgbm_tweedie::random_1",
        "scheduler_id": "risk_adjusted_mrp",
        "baseline_id": "historical_average::traditional_baseline",
        "split_manifest_sha256": "split-hash",
    }
    marker = {
        **current,
        "run_id": FINAL_TEST_RUN_ID,
        "run_metadata_path": EVALUATION_OUTPUTS[8],
        "status": "completed",
        "target_access_started": True,
        "final_test_targets_consumed": True,
    }
    evaluation = {
        **current,
        "run_id": FINAL_TEST_RUN_ID,
        "status": "completed",
        "target_access_started": True,
        "final_test_targets_consumed": True,
    }
    return current, marker, evaluation


def _mock_finalization_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    current: dict,
    marker: dict,
    evaluation: dict,
):
    (tmp_path / "results").mkdir(exist_ok=True)
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / ATTEMPT_MARKER).write_text(
        json.dumps(marker), encoding="utf-8"
    )
    (tmp_path / EVALUATION_OUTPUTS[8]).write_text(
        json.dumps(evaluation), encoding="utf-8"
    )
    protocol_hashes = {
        key: value for key, value in current.items()
        if key != "split_manifest_sha256"
    }
    monkeypatch.setattr(
        "src.evaluation.phase7.validate_protocol",
        lambda _root: protocol_hashes,
    )
    monkeypatch.setattr(
        "src.evaluation.phase7._read_split_metadata",
        lambda _path: pd.DataFrame(
            {"week_start_date": [], "split_role": []}
        ),
    )
    monkeypatch.setattr(
        "src.evaluation.phase7._validate_split_metadata",
        lambda _metadata: {"split_manifest_sha256": current["split_manifest_sha256"]},
    )


def test_finalization_rejects_missing_consumption_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    current, marker, evaluation = _completion_payload()
    del marker["final_test_targets_consumed"]
    _mock_finalization_validation(tmp_path, monkeypatch, current, marker, evaluation)

    def fail_if_training_data_read(*_args, **_kwargs):
        raise AssertionError("training data read before finalization gate")

    monkeypatch.setattr(
        "src.evaluation.phase7._read_demand_for_authorized_operation",
        fail_if_training_data_read,
    )
    with pytest.raises(RuntimeError, match="fully consumed"):
        run_finalization(
            tmp_path,
            authorized=True,
            confirmation=FINALIZE_CONFIRMATION,
        )


@pytest.mark.parametrize(
    "field",
    ["protocol_id", "phase7_config_sha256", "split_manifest_sha256"],
)
def test_finalization_rejects_mismatched_frozen_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
):
    current, marker, evaluation = _completion_payload()
    marker[field] = "mismatch"
    evaluation[field] = "mismatch"
    _mock_finalization_validation(tmp_path, monkeypatch, current, marker, evaluation)

    with pytest.raises(ValueError, match=field):
        _validate_completed_evaluation(tmp_path)


def _synthetic_demand_csv(tmp_path: Path, future_target=None, missing_target=None):
    from src.evaluation.phase7 import DEMAND_READ_COLUMNS

    dates = pd.to_datetime(["2026-03-23", "2026-03-30", "2026-09-28"])
    roles = ["train_val", "final_test", "future"]
    targets = [0, 0, future_target]
    missing_flags = [0, 0, 0]
    if missing_target is not None:
        targets[0] = missing_target
        missing_flags[0] = 1
    metadata = pd.DataFrame(
        {"week_start_date": dates, "split_role": roles}
    )
    rows = []
    for date, role, target, missing_flag in zip(
        dates, roles, targets, missing_flags
    ):
        row = {column: "" for column in DEMAND_READ_COLUMNS}
        row.update(
            {
                "part_id": "P0001",
                "region": "East",
                "week_start_date": date.strftime("%Y-%m-%d"),
                "split_role": role,
                "demand_qty": target,
                "is_missing_week": missing_flag,
            }
        )
        rows.append(row)
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    source = data_dir / "01_demand_forecasting.csv"
    pd.DataFrame(rows).assign(out_spike_flag=1).to_csv(source, index=False)
    return source, metadata


def _patch_synthetic_split(
    monkeypatch: pytest.MonkeyPatch, metadata: pd.DataFrame
):
    monkeypatch.setattr(
        "src.evaluation.phase7._read_split_metadata",
        lambda _path: metadata.copy(),
    )
    monkeypatch.setattr(
        "src.evaluation.phase7._validate_split_metadata",
        lambda _metadata: {"split_manifest_sha256": "synthetic-split"},
    )


def test_target_reads_are_row_scoped_and_reject_future_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source, metadata = _synthetic_demand_csv(tmp_path, future_target=7)
    _patch_synthetic_split(monkeypatch, metadata)

    with pytest.raises(ValueError, match="Future split contains"):
        _read_demand_for_authorized_operation(
            tmp_path,
            expected_split_manifest_sha256="synthetic-split",
        )


def test_target_reader_rejects_populated_missing_week_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _source, metadata = _synthetic_demand_csv(tmp_path, missing_target=7)
    _patch_synthetic_split(monkeypatch, metadata)

    with pytest.raises(ValueError, match="Missing-week rows"):
        _read_demand_for_authorized_operation(
            tmp_path,
            expected_split_manifest_sha256="synthetic-split",
        )


def test_target_reader_keeps_zero_and_excludes_future_and_outcome_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source, metadata = _synthetic_demand_csv(tmp_path)
    _patch_synthetic_split(monkeypatch, metadata)

    demand = _read_demand_for_authorized_operation(
        tmp_path,
        expected_split_manifest_sha256="synthetic-split",
    )

    assert len(demand) == 3
    assert demand.iloc[:2]["demand_qty"].tolist() == [0.0, 0.0]
    assert pd.isna(demand.iloc[2]["demand_qty"])
    assert demand["split_role"].tolist() == [
        "train_val", "final_test", "future"
    ]
    assert not any(column.startswith("out_") for column in demand.columns)


def test_report_explains_single_26_week_lead_time_aggregate():
    report = _phase7_report(
        {
            "protocol_id": FINAL_TEST_RUN_ID,
            "champion_id": "lightgbm_quantile::random_1",
            "scheduler_id": "risk_adjusted_mrp",
            "point_benchmark_id": "lightgbm_tweedie::random_1",
            "baseline_id": "historical_average::traditional_baseline",
        },
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )

    assert "single origin-anchored 26-week aggregate" in report


def test_phase5_gate_requires_current_status_and_complete_evidence(tmp_path: Path):
    (tmp_path / "results").mkdir()
    state = tmp_path / "STATE.md"
    state.write_text("- **Phase 5: PARTIAL**\n", encoding="utf-8")
    assert _phase5_gate_blockers(tmp_path)

    state.write_text(
        "- **Phase 5: COMPLETE WITH APPROVED DATA-LIMITATION EXCEPTION**\n",
        encoding="utf-8",
    )
    assert _phase5_gate_blockers(tmp_path)
    evidence = {
        "phase": 5,
        "status": "COMPLETE WITH APPROVED DATA-LIMITATION EXCEPTION",
        "criteria": PHASE5_REQUIRED_EVIDENCE,
    }
    (tmp_path / PHASE5_GATE_EVIDENCE_PATH).write_text(
        json.dumps(evidence), encoding="utf-8"
    )
    assert _phase5_gate_blockers(tmp_path) == []


def test_training_rejects_populated_missing_week_target():
    history = pd.DataFrame(
        {
            "part_id": ["P1"],
            "region": ["East"],
            "week_start_date": [pd.Timestamp("2026-01-05")],
            "split_role": ["train_val"],
            "demand_qty": [3],
            "is_missing_week": [1],
        }
    )

    with pytest.raises(ValueError, match="Missing-week rows"):
        build_fold_training_data(history, pd.Timestamp("2026-01-05"))


def test_final_fit_and_phase6_scheduler_defaults_are_preserved():
    history = pd.DataFrame(
        {
            "part_id": ["P1", "P1", "P1"],
            "region": ["East", "East", "East"],
            "week_start_date": pd.date_range("2026-01-05", periods=3, freq="W-MON"),
            "split_role": ["warmup", "train_val", "final_test"],
            "demand_qty": pd.Series([2, 0, 4], dtype="Int64"),
            "is_missing_week": [0, 0, 0],
        }
    )
    origin = pd.Timestamp("2026-01-19")

    _, default_target = build_fold_training_data(history, origin)
    _, final_target = build_fold_training_data(
        history,
        origin,
        fit_split_roles=("warmup", "train_val", "final_test"),
    )

    assert default_target.tolist() == [0.0]
    assert final_target.tolist() == [2.0, 0.0, 4.0]
    scheduler_parameters = inspect.signature(_run_plan).parameters
    assert scheduler_parameters["horizon"].default == 14
    assert scheduler_parameters["run_id_prefix"].default == "phase6"
