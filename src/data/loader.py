"""Explicitly typed, schema-validated readers and leak-safe table selectors.

Future-split rows are prediction-only; any displayed projection from them must
carry ``forecast_status="provisional_bridge"`` until it is operationally validated.
"""
from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd


ROOT: Final = Path(__file__).resolve().parents[2]
DATA_DIR: Final = ROOT / "data"

D1_STRING: Final = {
    "part_id", "region", "split_role", "quarter_fy", "fiscal_year",
    "festival_name", "festival_intensity", "part_name", "component",
    "vehicle_system", "part_category", "vehicle_model", "vehicle_type",
    "criticality", "abc_class", "part_type", "lifecycle_status", "make_or_buy",
    "fulfilment_mode", "demand_driver_tags", "lifecycle_pattern",
    "substitution_group", "intended_sbc_class", "demand_class_trainval",
}
D1_INT: Final = {
    "demand_qty", "demand_qty_capped", "is_missing_week", "year", "month",
    "week_of_year", "week_of_month", "festival_flag", "public_holiday_flag",
    "plant_shutdown_flag", "plant_working_days", "monsoon_flag",
    "summer_peak_flag", "fiscal_year_end_flag", "dealer_order_cycle_flag",
    "promotion_flag", "campaign_flag", "is_imported", "out_consumption_qty",
    "out_backorder_qty", "out_lost_sales_qty", "out_stockout_flag",
    "out_spike_flag", "out_unusual_demand_flag",
}
D1_FLOAT: Final = {
    "weeks_to_next_festival", "weeks_since_last_festival",
    "weeks_since_last_shutdown", "promotion_intensity", "campaign_intensity",
    "market_event_flag", "market_condition_index", "vehicle_parc",
    "avg_vehicle_age_years", "demand_lag52", "unit_cost", "adi_trainval",
    "cv2_trainval",
}
D1_DATE: Final = {"week_start_date"}
D1_HEADERS: Final = D1_STRING | D1_INT | D1_FLOAT | D1_DATE

D2_STRING: Final = {
    "part_id", "supply_mode", "lifecycle_status", "criticality",
    "planning_priority",
    *{
        f"sup{supplier}_{field}"
        for supplier in range(1, 4)
        for field in ("id", "type", "country")
    },
}
D2_INT: Final = {
    "on_hand_qty", "reserved_qty", "blocked_qty", "available_qty",
    "backorder_qty", "in_transit_qty", "open_po_qty",
    "open_po_due_this_month_qty", "open_po_due_later_qty",
    "emergency_po_open_flag", "current_safety_stock_qty",
    "current_reorder_point_qty",
}
D2_FLOAT: Final = {
    "service_level_target", "unit_cost", "carrying_cost_rate",
    *{
        f"sup{supplier}_{field}"
        for supplier in range(1, 4)
        for field in (
            "sob_pct", "lead_time_days", "moq", "order_multiple",
            "monthly_capacity", "capacity_committed_pct", "otd_pct",
            "avg_delay_days", "reject_pct", "reliability_pct",
        )
    },
}
D2_DATE: Final = {"as_of_date", "earliest_open_po_eta"}
D2_HEADERS: Final = D2_STRING | D2_INT | D2_FLOAT | D2_DATE

D3_STRING: Final = {
    "part_id", "part_number", "part_name", "component", "variant",
    "part_description", "vehicle_system", "part_category", "material",
    "part_type", "vehicle_model", "vehicle_type", "criticality",
    "planning_priority", "abc_class", "make_or_buy", "country_of_origin",
    "lifecycle_status", "unit_of_measure", "fulfilment_mode",
    "demand_driver_tags", "lifecycle_pattern", "substitution_group",
}
D3_INT: Final = {"is_imported", "standard_pack_qty"}
D3_FLOAT: Final = {"service_level_target", "unit_cost", "shelf_life_days"}
D3_DATE: Final = {"launch_week"}
D3_HEADERS: Final = D3_STRING | D3_INT | D3_FLOAT | D3_DATE

FORBIDDEN_FEATURE_COLUMNS: Final = {
    "out_consumption_qty", "out_backorder_qty", "out_lost_sales_qty",
    "out_stockout_flag", "out_spike_flag", "out_unusual_demand_flag",
    "intended_sbc_class", "demand_driver_tags", "lifecycle_pattern",
    "demand_class_trainval", "adi_trainval", "cv2_trainval", "abc_class",
    "part_type",
}
LABEL_AND_METADATA_COLUMNS: Final = {
    "demand_qty", "demand_qty_capped", "is_missing_week", "split_role",
    "week_start_date", "part_number", "part_description",
}
CONDITIONAL_FEATURE_COLUMNS: Final = {
    "demand_lag52", "market_event_flag", "market_condition_index",
    "plant_shutdown_flag", "plant_working_days", "weeks_since_last_shutdown",
    "vehicle_parc", "avg_vehicle_age_years", "lifecycle_status",
    "launch_week", "substitution_group",
}


class DatasetSchemaError(ValueError):
    """Raised when a source file does not match the verified dataset contract."""


def _dtype_map(
    strings: set[str], integers: set[str], floats: set[str]
) -> dict[str, str]:
    return {
        **{name: "string" for name in strings},
        **{name: "Int64" for name in integers},
        **{name: "Float64" for name in floats},
    }


def _read_csv(
    path: Path,
    expected_headers: set[str],
    string_columns: set[str],
    integer_columns: set[str],
    float_columns: set[str],
    date_columns: set[str],
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required source dataset not found: {path}")
    typed_headers = (
        string_columns | integer_columns | float_columns | date_columns
    )
    if typed_headers != expected_headers:
        raise RuntimeError(
            f"Internal dtype map does not cover the verified schema for {path.name}; "
            f"missing={sorted(expected_headers - typed_headers)}, "
            f"extra={sorted(typed_headers - expected_headers)}"
        )
    actual_headers = list(pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns)
    if len(actual_headers) != len(set(actual_headers)):
        raise DatasetSchemaError(f"{path.name} contains duplicate column names")
    if set(actual_headers) != expected_headers:
        raise DatasetSchemaError(
            f"{path.name} schema mismatch; "
            f"missing={sorted(expected_headers - set(actual_headers))}, "
            f"unexpected={sorted(set(actual_headers) - expected_headers)}"
        )
    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype=_dtype_map(string_columns, integer_columns, float_columns),
        keep_default_na=True,
        low_memory=False,
    )
    for column in date_columns:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def load_demand(path: Path | None = None) -> pd.DataFrame:
    """Load D1 with verified columns and explicit nullable dtypes.

    Future-split projections shown to users must be labeled
    ``forecast_status="provisional_bridge"``.
    """
    return _read_csv(
        path or DATA_DIR / "01_demand_forecasting.csv",
        D1_HEADERS, D1_STRING, D1_INT, D1_FLOAT, D1_DATE,
    )


def load_scheduling(path: Path | None = None) -> pd.DataFrame:
    """Load D2 with verified columns and explicit nullable dtypes."""
    return _read_csv(
        path or DATA_DIR / "02_scheduling_inputs.csv",
        D2_HEADERS, D2_STRING, D2_INT, D2_FLOAT, D2_DATE,
    )


def load_catalogue(path: Path | None = None) -> pd.DataFrame:
    """Load D3 with verified columns and explicit nullable dtypes."""
    return _read_csv(
        path or DATA_DIR / "03_spare_parts_catalogue.csv",
        D3_HEADERS, D3_STRING, D3_INT, D3_FLOAT, D3_DATE,
    )


def load_all(
    data_dir: Path = DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all original datasets through schema-checked readers."""
    return (
        load_demand(data_dir / "01_demand_forecasting.csv"),
        load_scheduling(data_dir / "02_scheduling_inputs.csv"),
        load_catalogue(data_dir / "03_spare_parts_catalogue.csv"),
    )


def build_modeling_table(
    demand: pd.DataFrame, catalogue: pd.DataFrame
) -> pd.DataFrame:
    """Enrich D1 with catalogue-only fields without changing D1 row count."""
    shared = (set(demand.columns) & set(catalogue.columns)) - {"part_id"}
    joined = demand.merge(
        catalogue,
        on="part_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "__catalogue"),
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        missing = joined.loc[joined["_merge"] != "both", "part_id"].unique()
        raise DatasetSchemaError(
            f"D1 parts absent from the catalogue: {sorted(missing.tolist())}"
        )
    for column in shared:
        catalogue_column = f"{column}__catalogue"
        equal = (
            joined[column].eq(joined[catalogue_column])
            & joined[column].notna()
            & joined[catalogue_column].notna()
        ) | (
            joined[column].isna() & joined[catalogue_column].isna()
        )
        if not bool(equal.all()):
            raise DatasetSchemaError(
                f"D1 and D3 disagree on shared attribute {column!r}"
            )
    return joined.drop(
        columns=[
            "_merge",
            *[f"{column}__catalogue" for column in shared],
        ]
    )


def build_scheduling_table(
    scheduling: pd.DataFrame, catalogue: pd.DataFrame
) -> pd.DataFrame:
    """Enrich D2 with catalogue-only fields using a validated one-to-one join."""
    shared = (set(scheduling.columns) & set(catalogue.columns)) - {"part_id"}
    joined = scheduling.merge(
        catalogue,
        on="part_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "__catalogue"),
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        missing = joined.loc[joined["_merge"] != "both", "part_id"].unique()
        raise DatasetSchemaError(
            f"Scheduling parts absent from the catalogue: {sorted(missing.tolist())}"
        )
    for column in shared:
        catalogue_column = f"{column}__catalogue"
        equal = (
            joined[column].eq(joined[catalogue_column])
            & joined[column].notna()
            & joined[catalogue_column].notna()
        ) | (
            joined[column].isna() & joined[catalogue_column].isna()
        )
        if not bool(equal.all()):
            raise DatasetSchemaError(
                f"D2 and D3 disagree on shared attribute {column!r}"
            )
    return joined.drop(
        columns=[
            "_merge",
            *[f"{column}__catalogue" for column in shared],
        ]
    )


def default_demand_feature_columns(
    frame: pd.DataFrame,
) -> list[str]:
    """Return only origin-safe candidates, excluding labels and audit keys."""
    return [
        column for column in frame.columns
        if column not in FORBIDDEN_FEATURE_COLUMNS
        and not column.startswith("out_")
        and column not in LABEL_AND_METADATA_COLUMNS
        and column not in CONDITIONAL_FEATURE_COLUMNS
        and not column.endswith("__catalogue")
    ]


def eligible_training_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Select train/validation labels only; exclude missing, blank, and future."""
    required = {"split_role", "is_missing_week", "demand_qty"}
    absent = required - set(frame.columns)
    if absent:
        raise DatasetSchemaError(
            f"Cannot select training rows; required columns absent: {sorted(absent)}"
        )
    return frame.loc[
        frame["split_role"].eq("train_val")
        & frame["is_missing_week"].eq(0)
        & frame["demand_qty"].notna()
    ].copy()
