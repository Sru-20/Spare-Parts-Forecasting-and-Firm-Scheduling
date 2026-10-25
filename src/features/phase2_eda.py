"""Train-validation-only EDA, segmentation, figures, and feature dictionary."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.loader import ROOT, load_all
from src.features.segmentation import (
    ABC_VALUE_THRESHOLDS,
    ADI_THRESHOLD,
    CV2_THRESHOLD,
    MIN_CLASSIFICATION_WEEKS,
    XYZ_CV_THRESHOLDS,
    build_segment_table,
)
from src.features.time_features import (
    CALENDAR_FEATURES,
    OPTIONAL_ORIGIN_KNOWN_COVARIATES,
    STATIC_FEATURES,
)


KNOWN_SUBSTITUTION_SHIFT = pd.Timestamp("2025-04-07")


def _relative_change(event: float, baseline: float) -> float:
    return (event / baseline - 1.0) * 100.0 if baseline else np.nan


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_None._"
    data = frame.fillna("").astype(str)
    columns = list(data.columns)
    widths = {
        column: max(len(column), *(len(value) for value in data[column]))
        for column in columns
    }
    lines = [
        "| " + " | ".join(column.ljust(widths[column]) for column in columns) + " |",
        "| " + " | ".join("-" * widths[column] for column in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(row[column].ljust(widths[column]) for column in columns) + " |"
        for _, row in data.iterrows()
    )
    return "\n".join(lines)


def _event_comparisons(train: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    events = {
        "festival": "festival_flag",
        "public_holiday": "public_holiday_flag",
        "plant_shutdown": "plant_shutdown_flag",
        "promotion": "promotion_flag",
        "campaign": "campaign_flag",
        "market_event": "market_event_flag",
    }
    for label, column in events.items():
        if column not in train:
            continue
        known = train.loc[train[column].notna() & train["demand_qty"].notna()]
        event = known.loc[known[column].eq(1), "demand_qty"].astype(float)
        baseline = known.loc[known[column].eq(0), "demand_qty"].astype(float)
        rows.append(
            {
                "comparison": label,
                "event_rows": len(event),
                "baseline_rows": len(baseline),
                "event_mean_demand": float(event.mean()) if len(event) else np.nan,
                "baseline_mean_demand": float(baseline.mean()) if len(baseline) else np.nan,
                "relative_change_pct": _relative_change(
                    float(event.mean()) if len(event) else np.nan,
                    float(baseline.mean()) if len(baseline) else np.nan,
                ),
                "interpretation_limit": "descriptive synthetic association; not causal",
            }
        )

    tagged = train.loc[
        train["demand_driver_tags"].fillna("").str.contains(
            "promo_sensitive", regex=False
        )
    ].copy()
    tagged = tagged.sort_values(["part_id", "region", "week_start_date"])
    tagged["prior_week_promotion"] = tagged.groupby(
        ["part_id", "region"], observed=True
    )["promotion_flag"].shift(1)
    post_promo = tagged.loc[
        tagged["prior_week_promotion"].eq(1) & tagged["demand_qty"].notna(),
        "demand_qty",
    ].astype(float)
    non_dip = tagged.loc[
        tagged["prior_week_promotion"].eq(0) & tagged["demand_qty"].notna(),
        "demand_qty",
    ].astype(float)
    rows.append(
        {
            "comparison": "post_promotion_week_promo_sensitive",
            "event_rows": len(post_promo),
            "baseline_rows": len(non_dip),
            "event_mean_demand": float(post_promo.mean()) if len(post_promo) else np.nan,
            "baseline_mean_demand": float(non_dip.mean()) if len(non_dip) else np.nan,
            "relative_change_pct": _relative_change(
                float(post_promo.mean()) if len(post_promo) else np.nan,
                float(non_dip.mean()) if len(non_dip) else np.nan,
            ),
            "interpretation_limit": "generator tag used for descriptive audit only; excluded from features",
        }
    )

    spike_rows = train.loc[
        train["demand_qty"].notna() & train["out_spike_flag"].notna()
    ]
    rows.append(
        {
            "comparison": "generator_spike_label_prevalence",
            "event_rows": int(spike_rows["out_spike_flag"].eq(1).sum()),
            "baseline_rows": len(spike_rows),
            "event_mean_demand": float(
                spike_rows.loc[spike_rows["out_spike_flag"].eq(1), "demand_qty"].mean()
            ) if spike_rows["out_spike_flag"].eq(1).any() else np.nan,
            "baseline_mean_demand": float(spike_rows["demand_qty"].mean()),
            "relative_change_pct": np.nan,
            "interpretation_limit": "out_spike_flag is a historical label only, never a feature",
        }
    )
    return pd.DataFrame(rows)


def _substitution_summary(train: pd.DataFrame) -> pd.DataFrame:
    data = train.loc[
        train["substitution_group"].notna()
        & train["demand_qty"].notna()
    ].copy()
    if data.empty:
        return pd.DataFrame(
            columns=[
                "substitution_group", "part_id", "before_mean", "after_mean",
                "relative_change_pct", "rows_before", "rows_after",
            ]
        )
    data["period"] = np.where(
        data["week_start_date"] < KNOWN_SUBSTITUTION_SHIFT, "before", "after"
    )
    summary = (
        data.groupby(["substitution_group", "part_id", "period"], observed=True)
        .agg(mean_demand=("demand_qty", "mean"), rows=("demand_qty", "size"))
        .reset_index()
        .pivot(
            index=["substitution_group", "part_id"],
            columns="period",
            values=["mean_demand", "rows"],
        )
    )
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    summary = summary.reset_index()
    for name in ("mean_demand_before", "mean_demand_after", "rows_before", "rows_after"):
        if name not in summary:
            summary[name] = np.nan
    summary["relative_change_pct"] = (
        summary["mean_demand_after"] / summary["mean_demand_before"] - 1.0
    ) * 100.0
    return summary.rename(
        columns={
            "mean_demand_before": "before_mean",
            "mean_demand_after": "after_mean",
        }
    )


def _feature_dictionary() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for lag in (1, 4, 13, 52):
        rows.append(
            {
                "feature": f"demand_lag_{lag}",
                "formula": f"demand_qty(t-{lag})",
                "availability": "prior observed/forecast demand only",
                "leakage_control": "grouped shift; values strictly precede target week",
            }
        )
    rows.append(
        {
            "feature": "demand_same_period_last_year",
            "formula": "demand_qty(t-52)",
            "availability": "prior observed/forecast demand only",
            "leakage_control": "same as 52-week lag",
        }
    )
    for window in (4, 13, 52):
        for statistic in ("mean", "median", "std"):
            rows.append(
                {
                    "feature": f"demand_rolling_{window}_{statistic}",
                    "formula": f"{statistic} of prior {window} rows, shifted one week; partial windows allowed",
                    "availability": "prior observed/forecast demand only",
                    "leakage_control": "rolling calculation runs on demand shifted by one",
                }
            )
    for feature, formula in (
        ("demand_zero_run_length", "consecutive observed zero-demand weeks before t"),
        ("weeks_since_last_demand", "elapsed weeks since last observed positive demand before t"),
        ("demand_month_sin", "sin(2π × month(t) / 12)"),
        ("demand_month_cos", "cos(2π × month(t) / 12)"),
    ):
        rows.append(
            {
                "feature": feature,
                "formula": formula,
                "availability": "past labels or target-date calendar",
                "leakage_control": "current target is not used",
            }
        )
    for feature in STATIC_FEATURES:
        rows.append(
            {
                "feature": feature,
                "formula": "verified static source/catalogue value or origin-fitted segment",
                "availability": "static/origin-derived; conditional attributes require explicit origin approval",
                "leakage_control": "generator answer keys are not accepted",
            }
        )
    for feature in CALENDAR_FEATURES:
        rows.append(
            {
                "feature": feature,
                "formula": "calendar/planned value for target week",
                "availability": "known in advance only if published at forecast origin",
                "leakage_control": "unknown market events and post-origin revisions are excluded",
            }
        )
    for feature in sorted(OPTIONAL_ORIGIN_KNOWN_COVARIATES):
        rows.append(
            {
                "feature": feature,
                "formula": "optional covariate passed by caller",
                "availability": "conditional; not passed by default",
                "leakage_control": "caller must affirm origin-time availability; no automatic source passthrough",
            }
        )
    return pd.DataFrame(rows).drop_duplicates("feature").sort_values("feature")


def build_phase2_outputs(root: Path = ROOT) -> dict[str, object]:
    """Compute train-validation EDA, segments, charts, and feature documentation."""
    demand, _, catalogue = load_all(root / "data")
    train = demand.loc[
        demand["split_role"].eq("train_val")
        & demand["is_missing_week"].eq(0)
        & demand["demand_qty"].notna()
    ].copy()
    if train.empty:
        raise ValueError("No observed train-validation data for Phase 2 EDA")
    if train["week_start_date"].max() > pd.Timestamp("2026-03-23"):
        raise ValueError("Phase 2 EDA must not include final-test observations")

    segments = build_segment_table(demand)
    part_catalogue = catalogue[
        ["part_id", "abc_class", "part_type", "criticality", "component", "vehicle_type"]
    ].drop_duplicates("part_id")
    segments = segments.merge(
        part_catalogue[["part_id", "abc_class", "part_type"]].rename(
            columns={
                "abc_class": "generator_abc_class_source",
                "part_type": "generator_part_type_source",
            }
        ),
        on="part_id",
        how="left",
        validate="many_to_one",
    )
    weekly_region = (
        train.groupby(["week_start_date", "region"], observed=True)
        .agg(demand_qty=("demand_qty", "sum"))
        .reset_index()
    )
    month_seasonality = (
        train.groupby("month", observed=True)
        .agg(
            mean_weekly_demand=("demand_qty", "mean"),
            median_weekly_demand=("demand_qty", "median"),
            observed_rows=("demand_qty", "size"),
        )
        .reset_index()
    )
    event_comparison = _event_comparisons(train)
    substitution = _substitution_summary(train)
    region_summary = (
        train.groupby("region", observed=True)
        .agg(
            demand_units=("demand_qty", "sum"),
            mean_weekly_part_region_demand=("demand_qty", "mean"),
            observed_rows=("demand_qty", "size"),
        )
        .reset_index()
    )
    criticality_summary = (
        train.groupby("criticality", observed=True)
        .agg(
            demand_units=("demand_qty", "sum"),
            mean_weekly_demand=("demand_qty", "mean"),
            stockout_label_rate=("out_stockout_flag", "mean"),
            observed_rows=("demand_qty", "size"),
        )
        .reset_index()
    )
    lifecycle_summary = (
        train.groupby("lifecycle_status", observed=True)
        .agg(
            demand_units=("demand_qty", "sum"),
            mean_weekly_demand=("demand_qty", "mean"),
            part_count=("part_id", "nunique"),
        )
        .reset_index()
    )
    part_driver = train.groupby("part_id", observed=True).agg(
        vehicle_parc=("vehicle_parc", "mean"),
        avg_vehicle_age_years=("avg_vehicle_age_years", "mean"),
        mean_demand=("demand_qty", "mean"),
        part_name=("part_name", "first"),
    )
    parc_correlation = float(
        part_driver[["vehicle_parc", "mean_demand"]].corr(method="spearman").iloc[0, 1]
    )
    age_correlation = float(
        part_driver[["avg_vehicle_age_years", "mean_demand"]]
        .corr(method="spearman")
        .iloc[0, 1]
    )
    part_tags = train.drop_duplicates("part_id")["demand_driver_tags"].fillna("")
    driver_counts = {
        "step_change_tagged_parts": int(
            part_tags.str.contains("step_change", regex=False).sum()
        ),
        "promo_sensitive_parts": int(
            part_tags.str.contains("promo_sensitive", regex=False).sum()
        ),
    }

    results_dir = root / "results"
    artifacts_dir = root / "artifacts"
    reports_dir = root / "reports"
    for directory in (results_dir, artifacts_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)
    chart_dir = results_dir / "phase2_charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    segments.to_parquet(
        artifacts_dir / "series_segments.parquet",
        index=False,
        engine="fastparquet",
    )
    weekly_region.to_csv(results_dir / "phase2_weekly_region.csv", index=False)
    month_seasonality.to_csv(
        results_dir / "phase2_month_seasonality.csv", index=False
    )
    event_comparison.to_csv(
        results_dir / "phase2_event_comparisons.csv", index=False
    )
    substitution.to_csv(
        results_dir / "phase2_substitution_comparison.csv", index=False
    )
    region_summary.to_csv(results_dir / "phase2_region_summary.csv", index=False)
    criticality_summary.to_csv(
        results_dir / "phase2_criticality_summary.csv", index=False
    )
    lifecycle_summary.to_csv(
        results_dir / "phase2_lifecycle_summary.csv", index=False
    )
    _feature_dictionary().to_csv(
        reports_dir / "feature_dictionary_phase2.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    for region, group in weekly_region.groupby("region", observed=True):
        ax.plot(group["week_start_date"], group["demand_qty"], label=region, linewidth=0.9)
    ax.set_title("Weekly demand by region — train/validation only")
    ax.set_ylabel("Demand units")
    ax.set_xlabel("Week")
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(chart_dir / "weekly_demand_by_region.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(month_seasonality["month"], month_seasonality["mean_weekly_demand"], marker="o")
    ax.set_xticks(range(1, 13))
    ax.set_title("Mean weekly demand by calendar month — train/validation only")
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean demand per part-region week")
    fig.tight_layout()
    fig.savefig(chart_dir / "month_of_year_seasonality.png", dpi=150)
    plt.close(fig)

    chart_events = event_comparison.loc[
        event_comparison["comparison"].isin(
            ["festival", "public_holiday", "plant_shutdown", "promotion", "campaign"]
        )
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(chart_events["comparison"], chart_events["relative_change_pct"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Descriptive demand difference during flagged event weeks")
    ax.set_ylabel("Difference vs non-event mean (%)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(chart_dir / "event_demand_differences.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    segment_counts = segments.groupby("sbc_class", dropna=False).size()
    segment_counts.plot(kind="bar", ax=ax)
    ax.set_title("Development-window SBC classes by part-region series")
    ax.set_ylabel("Series count")
    ax.set_xlabel("Computed class (blank = short/unclassified)")
    fig.tight_layout()
    fig.savefig(chart_dir / "sbc_segment_counts.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    train_sample = train.sample(
        n=min(15_000, len(train)), random_state=42
    )
    ax.scatter(
        train_sample["vehicle_parc"],
        train_sample["demand_qty"],
        alpha=0.15,
        s=8,
    )
    ax.set_title("Vehicle parc vs weekly demand — synthetic train/validation rows")
    ax.set_xlabel("Vehicle parc")
    ax.set_ylabel("Demand units")
    fig.tight_layout()
    fig.savefig(chart_dir / "vehicle_parc_vs_demand.png", dpi=150)
    plt.close(fig)

    sbc_counts = (
        segments["sbc_class"].fillna("Unclassified")
        .value_counts()
        .reindex(["Smooth", "Erratic", "Intermittent", "Lumpy", "Unclassified"], fill_value=0)
        .to_dict()
    )
    agreement = segments.loc[
        segments["supplied_class_agrees"].notna(), "supplied_class_agrees"
    ]
    agreement_rate = float(agreement.mean()) if len(agreement) else np.nan
    class_lookup = segments[["part_id", "abc_class", "xyz_class"]].drop_duplicates()
    abc_counts = class_lookup["abc_class"].value_counts().to_dict()
    xyz_counts = class_lookup["xyz_class"].value_counts().to_dict()
    report = f"""# Phase 2 EDA, Segmentation and Feature Engineering

## Scope and leakage boundary

Demand-label EDA, association summaries, and event diagnostics use only observed `train_val` rows dated through 2026-03-23. SBC/ABC/XYZ segments use observed `warmup` plus `train_val` labels through that same cutoff, matching the generator's verified `demand_class_trainval` calculation (which includes warm-up despite the README's train-validation-only description). `final_test` and `future` labels/rows are not used. Dataset `out_*` fields are read only as historical synthetic labels. Generator tags and lifecycle fields are used only for clearly labelled artifact diagnostics, never as model predictors. Descriptive associations are not causal estimates and do not validate real factory effects.

## What is supported by the supplied data

- Weekly regional demand, calendar month, festival/holiday, promotion/campaign, plant shutdown, market, vehicle parc/age, lifecycle, criticality, and the generator's synthetic outcome labels are present for historical train-validation dates.
- Per-part region series are segmented from observed warm-up and train-validation demand. The supplied `demand_class_trainval` is cross-checked only; recomputed SBC labels are used. Agreement for comparable classified series: {agreement_rate:.3%} ({len(agreement):,} series).
- Computed SBC series counts: {sbc_counts}. Series with fewer than {MIN_CLASSIFICATION_WEEKS} observed development weeks, or no observed development weeks, are left unclassified; they receive the same-component/vehicle-type analogue group.
- Part ABC uses annualized development-window usage value (average observed weekly `demand_qty × unit_cost × 52`) and cumulative-value cutoffs 80%/95%. Part XYZ uses CV of weekly part demand aggregated across regions: X ≤ {XYZ_CV_THRESHOLDS[0]:.2f}, Y ≤ {XYZ_CV_THRESHOLDS[1]:.2f}, Z > {XYZ_CV_THRESHOLDS[1]:.2f}; zero-mean series are Unclassified.
- Current computed part counts by ABC: {abc_counts}; XYZ: {xyz_counts}. Criticality remains the supplied part attribute and is attached to every part-region segment.
- Spearman correlation across parts, using train-validation mean demand, is {parc_correlation:.4f} for mean vehicle parc and {age_correlation:.4f} for mean vehicle age. These are synthetic descriptive associations, not causal or independently validated fleet effects.

## Event and known-pattern diagnostics

{_markdown_table(event_comparison)}

Substitution comparisons use the generator-documented 2025-04-07 change and train-validation rows only. The pairs are in `results/phase2_substitution_comparison.csv`; these effects are injected generator behavior, not discovered causal facts.

Generator tag diagnostics (distinct parts in train-validation rows): {driver_counts}. Tags are answer keys and are excluded from features. The synthetic `out_spike_flag` is used only to describe labelled historical spikes; see event table and never use it as a feature.

Regional and criticality summaries are in `results/phase2_region_summary.csv` and `results/phase2_criticality_summary.csv`. Lifecycle summaries are in `results/phase2_lifecycle_summary.csv`. The provided data cannot establish causal festival, promotion, vehicle-use, lifecycle, substitution, or shutdown effects in a real supply chain; real external covariates and operational outcomes are unavailable.

## Figures

- `results/phase2_charts/weekly_demand_by_region.png`
- `results/phase2_charts/month_of_year_seasonality.png`
- `results/phase2_charts/event_demand_differences.png`
- `results/phase2_charts/sbc_segment_counts.png`
- `results/phase2_charts/vehicle_parc_vs_demand.png`

## Origin-safe feature construction

Implemented in `src/features/time_features.py`. One-step features use only demand before the target week: lags 1/4/13/52; same-period-last-year; rolling mean/median/sample standard deviation for windows 4/13/52 using a one-row shift; zero-run length and elapsed weeks since last observed positive demand; calendar sin/cos. Target-date calendar fields are passed through only from the explicitly approved calendar set. Conditional future covariates are not passed by default and require explicit `origin_known_columns`.

For the 14-week future horizon the selected strategy is **recursive/direct one-step iteration**: pass the future calendar to `build_recursive_forecast_features`, which processes it chronologically and appends each model prediction as the prior value before constructing the next week's features. No actual future target or `out_*` value is accepted from the future calendar. Missing prior targets are never filled with zero; missing observations reset the consecutive-zero run.

`reports/feature_dictionary_phase2.csv` lists every generated lag, rolling, intermittent, static, calendar, and conditional feature with its formula and origin-time safeguard. `tests/test_leakage.py` verifies truncated-history invariance, future-label rejection, prior-only lag/rolling values, and recursive feedback.

## Feature and segmentation parameters

- SBC: ADI = observed weeks / positive-demand weeks; CV² = sample variance / mean² over positive demand. Smooth: ADI < {ADI_THRESHOLD} and CV² < {CV2_THRESHOLD}; Erratic: ADI < {ADI_THRESHOLD} and CV² ≥ {CV2_THRESHOLD}; Intermittent: ADI ≥ {ADI_THRESHOLD} and CV² < {CV2_THRESHOLD}; Lumpy: ADI ≥ {ADI_THRESHOLD} and CV² ≥ {CV2_THRESHOLD}. Minimum {MIN_CLASSIFICATION_WEEKS} observed weeks. Sparse (<3 positive weeks) cases follow the generator's CV²=1 convention and are flagged.
- ABC: annualized usage value with 80%/95% cumulative-value boundaries; explicitly a project segmentation convention.
- XYZ: CV of aggregated weekly part demand with 0.50 and 1.00 boundaries; explicitly a project segmentation convention.

## Artifacts

- `artifacts/series_segments.parquet`
- `reports/feature_dictionary_phase2.csv`
- `results/phase2_*` summary CSVs and `results/phase2_charts/*.png`
- Re-run: `python -m src.features.phase2_eda`
"""
    (results_dir / "phase2_eda_report.md").write_text(report, encoding="utf-8")
    return {
        "trainval_rows": len(train),
        "series_segments": len(segments),
        "sbc_counts": sbc_counts,
        "abc_part_counts": abc_counts,
        "xyz_part_counts": xyz_counts,
        "supplied_sbc_agreement": agreement_rate,
        "feature_dictionary_rows": len(_feature_dictionary()),
        "eda_final_test_max_date": str(train["week_start_date"].max().date()),
    }


if __name__ == "__main__":
    for key, value in build_phase2_outputs().items():
        print(f"{key}: {value}")
