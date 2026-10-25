from pathlib import Path

import pandas as pd
import pytest

from src.data.build_phase1_artifacts import (
    _feature_audit,
    _validate_phase1_integrity,
)
from src.data.loader import (
    DatasetSchemaError,
    build_modeling_table,
    build_scheduling_table,
    default_demand_feature_columns,
    eligible_training_rows,
    load_all,
    load_demand,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def source_frames():
    return load_all(ROOT / "data")


def test_explicit_dtypes_and_verified_shapes(source_frames):
    demand, scheduling, catalogue = source_frames
    assert demand.shape == (120_861, 62)
    assert scheduling.shape == (158, 61)
    assert catalogue.shape == (160, 29)
    assert pd.api.types.is_datetime64_any_dtype(demand["week_start_date"])
    assert pd.api.types.is_datetime64_any_dtype(scheduling["as_of_date"])
    assert str(demand["demand_qty"].dtype) == "Int64"
    assert str(demand["part_id"].dtype) == "string"
    assert str(scheduling["unit_cost"].dtype) == "Float64"


def test_phase1_output_artifacts_are_readable_parquet_and_documented():
    modeling_path = ROOT / "artifacts" / "modelling_table.parquet"
    scheduling_path = ROOT / "artifacts" / "scheduling_inputs.parquet"
    report_path = ROOT / "results" / "data_quality_report.md"
    assert modeling_path.is_file()
    assert scheduling_path.is_file()
    assert report_path.is_file()
    modeling = pd.read_parquet(modeling_path, columns=["part_id", "split_role"])
    scheduling = pd.read_parquet(scheduling_path, columns=["part_id"])
    report = report_path.read_text(encoding="utf-8")
    assert len(modeling) == 120_861
    assert len(scheduling) == 158
    assert "Eligible training rows after split/missing filters: 66,345" in report
    assert "1,403 observed historical rows differ" in report


def test_joins_preserve_rows_and_reject_shared_attribute_mismatch(source_frames):
    demand, scheduling, catalogue = source_frames
    modeling = build_modeling_table(demand, catalogue)
    scheduling_table = build_scheduling_table(scheduling, catalogue)
    assert len(modeling) == len(demand)
    assert len(scheduling_table) == len(scheduling)
    assert modeling["part_id"].nunique() == demand["part_id"].nunique()

    bad_catalogue = catalogue.copy()
    part_id = bad_catalogue.loc[0, "part_id"]
    bad_catalogue.loc[0, "part_name"] = "conflicting name"
    with pytest.raises(DatasetSchemaError, match="shared attribute"):
        build_modeling_table(
            demand.loc[demand["part_id"].eq(part_id)].head(1), bad_catalogue
        )


def test_source_integrity_and_feature_audit_have_no_hidden_default_leaks(source_frames):
    demand, scheduling, catalogue = source_frames
    integrity = _validate_phase1_integrity(demand, scheduling, catalogue)
    assert integrity["duplicate_keys"] == {
        "demand": 0,
        "scheduling": 0,
        "catalogue": 0,
    }
    assert integrity["weekly_gap_count"] == 0
    assert integrity["negative_numeric_values"] == 0
    modeling = build_modeling_table(demand, catalogue)
    audit = _feature_audit(demand, scheduling, catalogue, modeling)
    defaults = audit.loc[audit["default_forecast_feature"].eq("Yes")]
    assert defaults["available_at_forecast_origin"].eq("Yes").all()
    assert not defaults["column"].str.startswith("out_").any()
    assert audit.loc[
        audit["dataset"].eq("scheduling"), "default_forecast_feature"
    ].eq("No").all()


def test_default_feature_list_has_no_target_outcomes_or_answer_keys(source_frames):
    demand, _, _ = source_frames
    features = default_demand_feature_columns(demand)
    assert features
    assert all(not column.startswith("out_") for column in features)
    assert not {
        "demand_qty",
        "demand_qty_capped",
        "is_missing_week",
        "split_role",
        "intended_sbc_class",
        "demand_driver_tags",
        "lifecycle_pattern",
        "part_number",
        "part_description",
        "vehicle_parc",
        "avg_vehicle_age_years",
        "demand_class_trainval",
        "adi_trainval",
        "cv2_trainval",
        "abc_class",
        "part_type",
    }.intersection(features)


def test_training_selection_excludes_future_test_and_missing_targets(source_frames):
    demand, _, catalogue = source_frames
    modeling = build_modeling_table(demand, catalogue)
    training = eligible_training_rows(modeling)
    assert set(training["split_role"].dropna().unique()) == {"train_val"}
    assert training["demand_qty"].notna().all()
    assert training["is_missing_week"].eq(0).all()
    assert not training["week_start_date"].isna().any()


def test_missing_labels_are_not_zero_filled_and_observed_zeros_remain(source_frames):
    demand, _, _ = source_frames
    missing = demand.loc[demand["is_missing_week"].eq(1)]
    observed_zeros = demand.loc[
        demand["split_role"].ne("future")
        & demand["is_missing_week"].eq(0)
        & demand["demand_qty"].eq(0)
    ]
    future = demand.loc[demand["split_role"].eq("future")]
    assert len(missing) == 72
    assert missing["demand_qty"].isna().all()
    assert len(observed_zeros) > 0
    assert observed_zeros["demand_qty"].eq(0).all()
    assert future["demand_qty"].isna().all()


def test_schema_drift_is_rejected(tmp_path):
    source = ROOT / "data" / "01_demand_forecasting.csv"
    altered = tmp_path / source.name
    altered.write_text(
        source.read_text(encoding="utf-8-sig").replace(
            "week_start_date", "renamed_week", 1
        ),
        encoding="utf-8",
    )
    with pytest.raises(DatasetSchemaError, match="schema mismatch"):
        load_demand(altered)
