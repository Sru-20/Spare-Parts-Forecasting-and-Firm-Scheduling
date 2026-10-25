"""Pre-test development-window demand segmentation."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49
MIN_CLASSIFICATION_WEEKS = 26
MIN_POSITIVE_WEEKS_FOR_CV2 = 3
ABC_VALUE_THRESHOLDS = (0.80, 0.95)
XYZ_CV_THRESHOLDS = (0.50, 1.00)


def classify_sbc_series(
    demand: pd.Series,
    minimum_weeks: int = MIN_CLASSIFICATION_WEEKS,
) -> dict[str, object]:
    """Compute ADI/CV² class using the verified generator convention."""
    observed = demand.dropna().astype(float)
    n_weeks = len(observed)
    positives = observed.loc[observed.gt(0)]
    n_positive = len(positives)
    if n_weeks < minimum_weeks:
        return {
            "adi": np.nan,
            "cv2": np.nan,
            "sbc_class": pd.NA,
            "classification_status": "unclassified_short",
            "observed_weeks": n_weeks,
            "positive_weeks": n_positive,
        }

    adi = n_weeks / max(n_positive, 1)
    if n_positive < MIN_POSITIVE_WEEKS_FOR_CV2:
        cv2 = 1.0
        label = "Lumpy" if adi >= ADI_THRESHOLD else "Smooth"
    else:
        mean_positive = float(positives.mean())
        sample_variance = float(positives.var(ddof=1))
        cv2 = (np.sqrt(sample_variance) / mean_positive) ** 2 if mean_positive else 0.0
        if adi < ADI_THRESHOLD and cv2 < CV2_THRESHOLD:
            label = "Smooth"
        elif adi < ADI_THRESHOLD:
            label = "Erratic"
        elif cv2 < CV2_THRESHOLD:
            label = "Intermittent"
        else:
            label = "Lumpy"
    return {
        "adi": adi,
        "cv2": cv2,
        "sbc_class": label,
        "classification_status": (
            "classified_low_positive_count" if n_positive < MIN_POSITIVE_WEEKS_FOR_CV2
            else "classified"
        ),
        "observed_weeks": n_weeks,
        "positive_weeks": n_positive,
    }


def _classify_abc_xyz(
    trainval: pd.DataFrame,
    part_column: str,
    date_column: str,
    target_column: str,
    cost_column: str,
) -> pd.DataFrame:
    valid = trainval.loc[
        trainval[target_column].notna()
        & trainval[cost_column].notna()
    ].copy()
    valid["weekly_usage_value"] = (
        valid[target_column].astype(float) * valid[cost_column].astype(float)
    )
    per_week = (
        valid.groupby([part_column, date_column], observed=True)
        .agg(
            weekly_demand=(target_column, "sum"),
            weekly_usage_value=("weekly_usage_value", "sum"),
        )
        .reset_index()
    )
    part_summary = (
        per_week.groupby(part_column, observed=True)
        .agg(
            active_weeks=(date_column, "nunique"),
            mean_weekly_demand=("weekly_demand", "mean"),
            sd_weekly_demand=("weekly_demand", "std"),
            mean_weekly_usage_value=("weekly_usage_value", "mean"),
        )
        .reset_index()
    )
    part_summary["annualized_usage_value"] = (
        part_summary["mean_weekly_usage_value"] * 52.0
    )
    total_value = float(part_summary["annualized_usage_value"].sum())
    ranked = part_summary.sort_values(
        ["annualized_usage_value", part_column],
        ascending=[False, True],
        kind="stable",
    ).copy()
    ranked["cumulative_value_share"] = (
        ranked["annualized_usage_value"].cumsum() / total_value
        if total_value > 0
        else 0.0
    )
    ranked["value_share_before_part"] = (
        ranked["cumulative_value_share"]
        - ranked["annualized_usage_value"] / total_value
        if total_value > 0
        else 0.0
    )
    ranked["abc_class"] = np.select(
        [
            ranked["value_share_before_part"] < ABC_VALUE_THRESHOLDS[0],
            ranked["value_share_before_part"] < ABC_VALUE_THRESHOLDS[1],
        ],
        ["A", "B"],
        default="C",
    )
    cv = ranked["sd_weekly_demand"] / ranked["mean_weekly_demand"]
    cv_values = cv.to_numpy(dtype=float, na_value=np.nan)
    ranked["xyz_cv"] = cv_values
    ranked["xyz_class"] = np.select(
        [
            np.isfinite(cv_values) & (cv_values <= XYZ_CV_THRESHOLDS[0]),
            np.isfinite(cv_values)
            & (cv_values > XYZ_CV_THRESHOLDS[0])
            & (cv_values <= XYZ_CV_THRESHOLDS[1]),
        ],
        ["X", "Y"],
        default="Z",
    )
    ranked.loc[
        ~np.isfinite(cv_values) | ranked["mean_weekly_demand"].eq(0),
        "xyz_class",
    ] = "Unclassified"
    return ranked.drop(columns=["value_share_before_part"])


def build_segment_table(
    frame: pd.DataFrame,
    group_columns: Sequence[str] = ("part_id", "region"),
    date_column: str = "week_start_date",
    target_column: str = "demand_qty",
    part_column: str = "part_id",
    cost_column: str = "unit_cost",
    criticality_column: str = "criticality",
    component_column: str = "component",
    vehicle_type_column: str = "vehicle_type",
    split_column: str = "split_role",
    missing_column: str = "is_missing_week",
    development_roles: Sequence[str] = ("warmup", "train_val"),
) -> pd.DataFrame:
    """Create per-series SBC labels and per-part ABC/XYZ/criticality labels.

    Only observed warm-up and train-validation targets are used. Short or
    otherwise unclassified series receive an analogue group, not an SBC class.
    """
    required = {
        *group_columns, date_column, target_column, part_column, cost_column,
        criticality_column, component_column, vehicle_type_column, split_column,
        missing_column,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Segmentation input is missing columns: {sorted(missing)}")
    development = frame.loc[
        frame[split_column].isin(development_roles)
        & frame[missing_column].eq(0)
        & frame[target_column].notna()
    ].copy()
    if development.empty:
        raise ValueError("No observed development demand for segmentation")

    series_records: list[dict[str, object]] = []
    supplied_column = "demand_class_trainval"
    training_groups = {
        key if isinstance(key, tuple) else (key,): group
        for key, group in development.groupby(
            list(group_columns), sort=True, observed=True
        )
    }
    all_series = frame[list(group_columns)].drop_duplicates()
    for key_row in all_series.itertuples(index=False, name=None):
        key = tuple(key_row)
        group = training_groups.get(key)
        if group is None:
            classification = {
                "adi": np.nan,
                "cv2": np.nan,
                "sbc_class": pd.NA,
                "classification_status": "unclassified_no_development_observations",
                "observed_weeks": 0,
                "positive_weeks": 0,
            }
        else:
            classification = classify_sbc_series(group[target_column])
        record = dict(zip(group_columns, key))
        record.update(classification)
        if group is not None and supplied_column in group.columns:
            supplied = group[supplied_column].dropna()
            record["supplied_sbc_class"] = (
                supplied.iloc[0] if not supplied.empty else pd.NA
            )
        else:
            record["supplied_sbc_class"] = pd.NA
        record["supplied_class_agrees"] = (
            record["sbc_class"] == record["supplied_sbc_class"]
            if pd.notna(record["sbc_class"])
            and pd.notna(record["supplied_sbc_class"])
            else pd.NA
        )
        series_records.append(record)
    series_segments = pd.DataFrame(series_records)

    part_metadata = frame.sort_values(date_column).drop_duplicates(part_column)[
        [part_column, criticality_column, component_column, vehicle_type_column, cost_column]
    ].rename(
        columns={
            criticality_column: "criticality",
            component_column: "component",
            vehicle_type_column: "vehicle_type",
            cost_column: "unit_cost",
        }
    )
    if "abc_class" in frame:
        part_metadata = part_metadata.merge(
            frame[[part_column, "abc_class"]]
            .drop_duplicates(part_column)
            .rename(columns={"abc_class": "generator_abc_class"}),
            on=part_column,
            how="left",
            validate="one_to_one",
        )
    else:
        part_metadata["generator_abc_class"] = pd.NA
    abc_xyz = _classify_abc_xyz(
    development, part_column, date_column, target_column, cost_column
    )
    part_metadata = part_metadata.merge(
        abc_xyz, on=part_column, how="left", validate="one_to_one"
    )
    part_metadata["abc_class"] = part_metadata["abc_class"].fillna("Unclassified")
    part_metadata["xyz_class"] = part_metadata["xyz_class"].fillna("Unclassified")
    segments = series_segments.merge(
        part_metadata, on=part_column, how="left", validate="many_to_one"
    )
    segments["analogue_fallback"] = segments["sbc_class"].isna()
    segments["analogue_component"] = segments["component"].where(
        segments["analogue_fallback"]
    )
    segments["analogue_vehicle_type"] = segments["vehicle_type"].where(
        segments["analogue_fallback"]
    )
    return segments.sort_values(list(group_columns), kind="stable").reset_index(drop=True)
