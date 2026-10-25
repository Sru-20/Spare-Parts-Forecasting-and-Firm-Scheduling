"""Build Phase 1 derived tables and reproducible audit reports."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loader import (
    CONDITIONAL_FEATURE_COLUMNS,
    FORBIDDEN_FEATURE_COLUMNS,
    LABEL_AND_METADATA_COLUMNS,
    ROOT,
    build_modeling_table,
    build_scheduling_table,
    default_demand_feature_columns,
    eligible_training_rows,
    load_all,
)


DATASET_NAMES = {
    "demand": "01_demand_forecasting.csv",
    "scheduling": "02_scheduling_inputs.csv",
    "catalogue": "03_spare_parts_catalogue.csv",
}

DESCRIPTIONS = {
    "week_start_date": "Monday start date for the part-region weekly observation.",
    "part_id": "Stable spare-part identifier and join key.",
    "region": "Demand region.",
    "split_role": "Generator-defined temporal split; future rows have no demand label.",
    "demand_qty": "Raw non-negative integer demand; primary training and scoring target.",
    "demand_qty_capped": "Demand capped at a generator threshold estimated over warm-up and train-validation; not safe as-is for rolling-origin fitting.",
    "is_missing_week": "1 identifies a missing historical target; exclude from fitting/scoring.",
    "demand_lag52": "Generator-materialized 52-week lag; recompute from origin-available history before use.",
    "demand_class_trainval": "SBC class computed from train-validation labels; not an origin-safe default feature.",
    "adi_trainval": "ADI derived from train-validation labels; not an origin-safe default feature.",
    "cv2_trainval": "CV-squared derived from train-validation labels; not an origin-safe default feature.",
    "intended_sbc_class": "Generator answer key; exclude from default features.",
    "demand_driver_tags": "Generator-injected demand pattern answer keys; exclude from default features.",
    "lifecycle_pattern": "Generator-injected lifecycle pattern answer key; exclude unless separately proven known at origin.",
    "out_consumption_qty": "Synthetic outcome label; never a predictor.",
    "out_backorder_qty": "Synthetic outcome label; never a predictor.",
    "out_lost_sales_qty": "Synthetic outcome label; never a predictor.",
    "out_stockout_flag": "Synthetic outcome label; never a predictor.",
    "out_spike_flag": "Synthetic outcome label; never a predictor.",
    "out_unusual_demand_flag": "Synthetic outcome label; never a predictor.",
    "available_qty": "Available quantity, validated as on-hand minus reserved minus blocked.",
    "open_po_qty": "Open purchase order quantity; equals due-this-month plus due-later quantity.",
    "backorder_qty": "Snapshot backorder quantity; potential input to pending-requirement derivation.",
    "service_level_target": "Part service target associated with criticality.",
    "carrying_cost_rate": "Part annual inventory carrying-cost rate from the scheduling snapshot.",
    "unit_cost": "Synthetic unit cost in the source dataset's currency basis.",
    "market_event_flag": "Historical synthetic market-event indicator; future events are unknown.",
    "market_condition_index": "Synthetic market driver; future values are not presumed known at an origin.",
    "substitution_group": "Generator-assigned synthetic substitution group; treat as conditional pending origin-availability review.",
    "launch_week": "Catalogue launch week; use only when the launch was known at forecast origin.",
    "lifecycle_status": "Part lifecycle state; historical availability depends on whether state was known at the origin.",
}

KNOWN_AHEAD = {
    "year", "month", "quarter_fy", "fiscal_year", "week_of_year",
    "week_of_month", "festival_flag", "festival_name", "festival_intensity",
    "weeks_to_next_festival", "weeks_since_last_festival",
    "public_holiday_flag", "monsoon_flag", "summer_peak_flag",
    "fiscal_year_end_flag", "dealer_order_cycle_flag", "promotion_flag",
    "promotion_intensity", "campaign_flag", "campaign_intensity",
}
STATIC_CATALOGUE_FEATURES = {
    "part_name", "component", "vehicle_system", "part_category",
    "vehicle_model", "vehicle_type", "criticality", "service_level_target",
    "make_or_buy", "is_imported", "unit_cost", "fulfilment_mode",
    "part_number", "variant",
    "part_description", "material", "country_of_origin", "unit_of_measure",
    "standard_pack_qty", "shelf_life_days",
}
CATALOGUE_ANSWER_KEYS = {
    "demand_driver_tags", "lifecycle_pattern", "abc_class", "part_type",
}


def _feature_audit(
    demand: pd.DataFrame,
    scheduling: pd.DataFrame,
    catalogue: pd.DataFrame,
    modeling: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    default_features = set(default_demand_feature_columns(modeling))
    for dataset, frame in (
        ("demand", demand),
        ("scheduling", scheduling),
        ("catalogue", catalogue),
    ):
        for column in frame.columns:
            if column in FORBIDDEN_FEATURE_COLUMNS or column.startswith("out_"):
                status = "No"
                rationale = "Label, generator answer key, or synthetic outcome; barred from default predictors."
            elif dataset == "demand" and column in LABEL_AND_METADATA_COLUMNS:
                status = "No"
                rationale = "Target, split, missingness metadata, or date key; use only as label/filter/index."
            elif dataset == "scheduling":
                status = "Conditional"
                rationale = "Only the 2026-09-26 snapshot is available; unavailable at earlier rolling origins and not a default demand feature."
            elif column in CATALOGUE_ANSWER_KEYS:
                status = "No"
                rationale = "Generator answer key or train-validation-derived master field; not a default predictor."
            elif column in CONDITIONAL_FEATURE_COLUMNS:
                status = "Conditional"
                rationale = "Must be recomputed from origin-visible history or independently confirmed known at the origin."
            elif column in KNOWN_AHEAD:
                status = "Yes"
                rationale = "Calendar/planned driver is usable only when its value was published by the forecast origin."
            elif (
                dataset == "catalogue"
                and column in STATIC_CATALOGUE_FEATURES | {"part_id"}
            ) or (dataset == "demand" and column in STATIC_CATALOGUE_FEATURES | {"part_id", "region"}):
                status = "Yes"
                rationale = "Static part/region attribute; verify it was effective and known at the origin."
            else:
                status = "No"
                rationale = "Not approved until an origin-availability rule is documented."
            rows.append(
                {
                    "dataset": dataset,
                    "column": column,
                    "available_at_forecast_origin": status,
                    "default_forecast_feature": (
                        "Yes"
                        if dataset in {"demand", "catalogue"}
                        and status == "Yes"
                        and column in default_features
                        else "No"
                    ),
                    "rationale": rationale,
                    "description": DESCRIPTIONS.get(
                        column, f"Source field `{column}`; see source CSV and README."
                    ),
                }
            )
    return pd.DataFrame(rows)


def _validate_phase1_integrity(
    demand: pd.DataFrame,
    scheduling: pd.DataFrame,
    catalogue: pd.DataFrame,
) -> dict[str, object]:
    duplicate_counts = {
        "demand": int(
            demand.duplicated(["part_id", "region", "week_start_date"]).sum()
        ),
        "scheduling": int(scheduling.duplicated(["part_id"]).sum()),
        "catalogue": int(catalogue.duplicated(["part_id"]).sum()),
    }
    if any(duplicate_counts.values()):
        raise ValueError(f"Duplicate primary keys found: {duplicate_counts}")

    catalogue_parts = set(catalogue["part_id"].dropna())
    demand_parts = set(demand["part_id"].dropna())
    scheduling_parts = set(scheduling["part_id"].dropna())
    if not demand_parts <= catalogue_parts or not scheduling_parts <= catalogue_parts:
        raise ValueError("Demand or scheduling parts are missing from the catalogue")
    lifecycle = catalogue.set_index("part_id")["lifecycle_status"]
    missing_schedule_parts = catalogue_parts - scheduling_parts
    if any(lifecycle.loc[part] != "Obsolete" for part in missing_schedule_parts):
        raise ValueError(
            "Catalogue parts missing from D2 include non-obsolete parts: "
            f"{sorted(missing_schedule_parts)}"
        )

    if not bool(
        (demand["is_missing_week"].eq(1) <= demand["demand_qty"].isna()).all()
    ):
        raise ValueError("Rows flagged missing must have blank raw demand targets")
    future = demand["split_role"].eq("future")
    if not demand.loc[future, "demand_qty"].isna().all():
        raise ValueError("Future rows must not contain raw demand targets")
    observed = ~future & demand["is_missing_week"].eq(0)
    if not demand.loc[observed, "demand_qty"].notna().all():
        raise ValueError("Observed historical rows must have nonblank raw targets")
    if demand.loc[observed, "demand_qty"].lt(0).any():
        raise ValueError("Observed demand contains negative quantities")
    if demand.loc[observed, "demand_qty_capped"].gt(
        demand.loc[observed, "demand_qty"]
    ).any():
        raise ValueError("Capped demand exceeds raw demand")

    weekly_dates = demand.sort_values(
        ["part_id", "region", "week_start_date"]
    ).groupby(["part_id", "region"], observed=True)["week_start_date"]
    day_gaps = weekly_dates.diff().dt.days.dropna()
    if not day_gaps.eq(7).all():
        raise ValueError("Demand series contain missing or non-weekly date gaps")
    negative_counts = _negative_counts(
        {"demand": demand, "scheduling": scheduling, "catalogue": catalogue}
    )
    return {
        "duplicate_keys": duplicate_counts,
        "demand_parts": len(demand_parts),
        "scheduling_parts": len(scheduling_parts),
        "catalogue_parts": len(catalogue_parts),
        "missing_schedule_parts": sorted(missing_schedule_parts),
        "weekly_gap_count": int(day_gaps.ne(7).sum()),
        "negative_numeric_values": int(negative_counts["negative_rows"].sum())
        if not negative_counts.empty
        else 0,
    }


def _data_dictionary(
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for dataset, frame in frames.items():
        for column in frame.columns:
            values = frame[column].dropna()
            example = "" if values.empty else str(values.iloc[0])
            records.append(
                {
                    "dataset": dataset,
                    "column": column,
                    "pandas_dtype": str(frame[column].dtype),
                    "rows": len(frame),
                    "missing_count": int(frame[column].isna().sum()),
                    "missing_pct": round(float(frame[column].isna().mean() * 100), 4),
                    "example_nonmissing": example,
                    "description": DESCRIPTIONS.get(
                        column, f"Source field `{column}`; see source CSV and README."
                    ),
                }
            )
    return pd.DataFrame(records)


def _outlier_counts(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    excluded = {
        "year", "month", "week_of_year", "week_of_month",
        "is_missing_week", "is_imported",
    }
    for column in frame.select_dtypes(include="number").columns:
        if column in excluded or frame[column].nunique(dropna=True) <= 10:
            continue
        values = frame[column].dropna()
        if values.empty:
            continue
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((frame[column] < lower) | (frame[column] > upper)).sum())
        rows.append(
            {
                "column": column,
                "q1": q1,
                "q3": q3,
                "iqr_lower_fence": lower,
                "iqr_upper_fence": upper,
                "iqr_outlier_rows": count,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "iqr_outlier_rows", ascending=False
    )


def _negative_counts(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, frame in frames.items():
        for column in frame.select_dtypes(include="number").columns:
            count = int(frame[column].lt(0).sum())
            if count:
                rows.append(
                    {"dataset": dataset, "column": column, "negative_rows": count}
                )
    return pd.DataFrame(rows, columns=["dataset", "column", "negative_rows"])


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_None._"
    selected = frame[columns].fillna("").astype(str)
    widths = {
        column: max(len(column), *(len(value) for value in selected[column]))
        for column in columns
    }
    header = "| " + " | ".join(column.ljust(widths[column]) for column in columns) + " |"
    separator = "| " + " | ".join("-" * widths[column] for column in columns) + " |"
    rows = [
        "| " + " | ".join(row[column].ljust(widths[column]) for column in columns) + " |"
        for _, row in selected.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def build_phase1_artifacts(root: Path = ROOT) -> dict[str, object]:
    """Load, validate, join and persist Phase 1 derived tables and audit reports."""
    data_dir = root / "data"
    demand, scheduling, catalogue = load_all(data_dir)
    integrity = _validate_phase1_integrity(demand, scheduling, catalogue)
    original_counts = {
        "demand": len(demand),
        "scheduling": len(scheduling),
        "catalogue": len(catalogue),
    }
    modeling = build_modeling_table(demand, catalogue)
    scheduling_table = build_scheduling_table(scheduling, catalogue)
    if len(modeling) != original_counts["demand"]:
        raise RuntimeError("Demand/catalogue join changed the D1 row count")
    if len(scheduling_table) != original_counts["scheduling"]:
        raise RuntimeError("Scheduling/catalogue join changed the D2 row count")

    # The target choice is based on observed semantics and the generator's cap logic:
    # raw demand is the label; the precomputed cap is not origin-fitted.
    historical = demand.loc[demand["split_role"].ne("future")]
    observed = historical.loc[
        historical["is_missing_week"].eq(0) & historical["demand_qty"].notna()
    ]
    cap_changed = int(
        (observed["demand_qty"].ne(observed["demand_qty_capped"])).sum()
    )
    genuine_zeros = int(observed["demand_qty"].eq(0).sum())
    missing_targets = int(
        (historical["is_missing_week"].eq(1) & historical["demand_qty"].isna()).sum()
    )
    future_rows = demand.loc[demand["split_role"].eq("future")]
    future_nonblank = int(future_rows["demand_qty"].notna().sum())

    training = eligible_training_rows(modeling)
    feature_audit = _feature_audit(demand, scheduling, catalogue, modeling)
    dictionary = _data_dictionary(
        {"demand": demand, "scheduling": scheduling, "catalogue": catalogue}
    )
    outliers = _outlier_counts(demand)
    negatives = _negative_counts(
        {"demand": demand, "scheduling": scheduling, "catalogue": catalogue}
    )
    missing_by_column = (
        dictionary.loc[dictionary["missing_count"].gt(0)]
        .sort_values(["dataset", "missing_count"], ascending=[True, False])
    )
    key_duplicates = integrity["duplicate_keys"]

    artifacts_dir = root / "artifacts"
    results_dir = root / "results"
    reports_dir = root / "reports"
    for directory in (artifacts_dir, results_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)
    modeling.to_parquet(
        artifacts_dir / "modelling_table.parquet",
        index=False,
        engine="fastparquet",
    )
    scheduling_table.to_parquet(
        artifacts_dir / "scheduling_inputs.parquet",
        index=False,
        engine="fastparquet",
    )
    dictionary.to_csv(reports_dir / "data_dictionary.csv", index=False)
    feature_audit.to_csv(
        reports_dir / "feature_availability_audit.csv", index=False
    )
    outliers.to_csv(results_dir / "demand_iqr_outliers.csv", index=False)
    negatives.to_csv(results_dir / "negative_values.csv", index=False)

    split_summary = (
        demand.groupby("split_role", observed=True)
        .agg(
            rows=("part_id", "size"),
            start=("week_start_date", "min"),
            end=("week_start_date", "max"),
            missing_targets=("demand_qty", lambda values: int(values.isna().sum())),
        )
        .reset_index()
    )
    split_summary["start"] = split_summary["start"].dt.strftime("%Y-%m-%d")
    split_summary["end"] = split_summary["end"].dt.strftime("%Y-%m-%d")
    report = f"""# Phase 1 Data Quality Report

Generated from the original CSVs via schema-validated readers. Source files were not modified.

## Shapes and join checks

| Dataset/table | Rows | Columns |
|---|---:|---:|
| D1 demand source | {len(demand):,} | {demand.shape[1]} |
| D2 scheduling source | {len(scheduling):,} | {scheduling.shape[1]} |
| D3 catalogue source | {len(catalogue):,} | {catalogue.shape[1]} |
| Modeling table (D1 enriched by D3) | {len(modeling):,} | {modeling.shape[1]} |
| Scheduling table (D2 enriched by D3) | {len(scheduling_table):,} | {scheduling_table.shape[1]} |

Join row counts are preserved: modeling {original_counts['demand']:,}/{len(modeling):,}; scheduling {original_counts['scheduling']:,}/{len(scheduling_table):,}. D1→D3 uses `many_to_one`; D2→D3 uses `one_to_one`; shared overlapping attributes are compared and must agree.

## Temporal split and labels

{_markdown_table(split_summary, ['split_role', 'rows', 'start', 'end', 'missing_targets'])}

- Missing historical labels excluded: {missing_targets:,}.
- Future rows: {len(future_rows):,}; future rows with a nonblank target: {future_nonblank:,}.
- Genuine observed zero-demand labels: {genuine_zeros:,} of {len(observed):,} non-missing historical observations.
- Eligible training rows after split/missing filters: {len(training):,}; roles present: {sorted(training['split_role'].dropna().unique().tolist())}.
- Final-test rows are not selected for fitting; future rows are not selected for fitting.

## Target decision

**Use raw `demand_qty` as the training target and score against raw `demand_qty`.** It represents demand to fulfill and preserves high-demand observations. The generator computes `demand_qty_capped` using a per-series cap estimated over warm-up plus the entire train-validation span, then applies that cap to earlier observations as well. The supplied capped column therefore is not an origin-fitted target transform for rolling origins. {cap_changed:,} observed historical rows differ between raw and capped labels; capped fitting is not the default. Genuine zeros remain zeros; missing rows remain excluded, not imputed as zero.

## Keys, negatives and outliers

- Duplicate keys (must be zero): {key_duplicates}.
- Referential integrity: PASS; catalogue/D1/D2 part counts are {integrity['catalogue_parts']}/{integrity['demand_parts']}/{integrity['scheduling_parts']}; D3-only obsolete parts: {integrity['missing_schedule_parts']}.
- Weekly continuity: PASS; non-seven-day gaps found: {integrity['weekly_gap_count']}.
- Negative numeric values across source tables: {integrity['negative_numeric_values']}.
- IQR outliers are a diagnostic, not deletion criteria; sparse/lumpy demand and large legitimate orders can be flagged.

Top IQR counts:

{_markdown_table(outliers.head(10), ['column', 'iqr_lower_fence', 'iqr_upper_fence', 'iqr_outlier_rows'])}

Negative values by column:

{_markdown_table(negatives, ['dataset', 'column', 'negative_rows'])}

## Missing values

Missingness is preserved as nullable values; target filtering is explicit. Detailed per-column type/missing counts and examples are in `reports/data_dictionary.csv`.

{_markdown_table(missing_by_column.head(15), ['dataset', 'column', 'pandas_dtype', 'missing_count', 'missing_pct'])}

## Integrity and leakage

- Every source schema is checked against its verified column set before parsing with explicit nullable string/integer/float and datetime dtypes.
- Referential integrity and join row preservation passed; shared D1/D3 and D2/D3 attributes must match.
- No future or final-test rows enter `eligible_training_rows`; only nonmissing `train_val` labels are returned.
- `reports/feature_availability_audit.csv` lists each source column, origin-time status, whether it is an approved default forecast feature, and rationale.
- Feature-origin audit totals: {feature_audit['available_at_forecast_origin'].value_counts().to_dict()}; approved default-feature entries across source schemas: {int(feature_audit['default_forecast_feature'].eq('Yes').sum())}. Conditional drivers and fields not available at all historical origins are not default features.
- Generator answer keys, `out_*` outcomes, train-validation-derived class summaries, targets and split metadata are barred from default features. Origin-dependent lags and future drivers require origin-time recomputation or proof of advance availability.
- Missing target count, genuine zeros and future-label absence are checked in tests.

## Artifacts

- `artifacts/modelling_table.parquet`
- `artifacts/scheduling_inputs.parquet`
- `reports/data_dictionary.csv`
- `reports/feature_availability_audit.csv`
- `results/demand_iqr_outliers.csv`
- `results/negative_values.csv`
"""
    (results_dir / "data_quality_report.md").write_text(report, encoding="utf-8")
    return {
        "demand_rows": len(demand),
        "modeling_rows": len(modeling),
        "scheduling_rows": len(scheduling),
        "scheduling_table_rows": len(scheduling_table),
        "training_rows": len(training),
        "missing_historical_targets": missing_targets,
        "genuine_zero_targets": genuine_zeros,
        "capped_target_differences": cap_changed,
        "future_rows": len(future_rows),
        "future_nonblank_targets": future_nonblank,
        "duplicate_keys": key_duplicates,
        "negative_numeric_values": int(negatives["negative_rows"].sum()) if not negatives.empty else 0,
        "feature_audit_rows": len(feature_audit),
        "data_dictionary_rows": len(dictionary),
    }


if __name__ == "__main__":
    summary = build_phase1_artifacts()
    for key, value in summary.items():
        print(f"{key}: {value}")
