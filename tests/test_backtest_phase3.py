from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import load_demand, load_scheduling
from src.forecasting.backtest import (
    ANALOGUE_FALLBACK,
    GLOBAL_MODEL_FAMILIES,
    SIMPLE_AVERAGE_ENSEMBLE,
    SIMPLE_AVERAGE_MEMBERS,
    SBC_ROUTER,
    _append_composite_candidates,
    _analogue_fallback_forecasts,
    _forecast_artifact_frame,
    _fold_data,
    _lead_time_map,
    _search_configurations,
)
from src.forecasting.global_models import (
    BASE_FEATURE_DROP_GROUPS,
    build_fold_training_data,
    select_feature_columns,
)
from src.features.time_features import (
    CALENDAR_FEATURES,
    STATIC_FEATURES,
    build_time_features,
    model_feature_columns,
)
from src.forecasting.global_models import _recursive_feature_frame


ROOT = Path(__file__).resolve().parents[1]


def test_phase3_folds_are_train_validation_only_and_cover_fourteen_weeks():
    demand = load_demand(ROOT / "data" / "01_demand_forecasting.csv")
    origin = pd.Timestamp("2025-11-03")
    history, targets, calendar = _fold_data(demand, origin, horizon=14)

    assert history["week_start_date"].max() == origin
    assert set(history["split_role"].unique()) <= {"warmup", "train_val"}
    assert set(targets["split_role"].unique()) == {"train_val"}
    assert targets["week_start_date"].min() == pd.Timestamp("2025-11-10")
    assert targets["week_start_date"].max() == pd.Timestamp("2026-02-09")
    assert calendar["week_start_date"].nunique() == 14
    assert "demand_qty" not in calendar
    assert not any(column.startswith("out_") for column in calendar)
    assert not {
        "promotion_flag",
        "promotion_intensity",
        "campaign_flag",
        "campaign_intensity",
        "market_event_flag",
        "market_condition_index",
        "plant_shutdown_flag",
        "plant_working_days",
    }.intersection(calendar.columns)


def test_unverified_future_covariates_are_excluded_from_model_features():
    demand = load_demand(ROOT / "data" / "01_demand_forecasting.csv")
    origin = pd.Timestamp("2025-11-03")
    history, _, _ = _fold_data(demand, origin, horizon=14)
    training_features, _ = build_fold_training_data(history, origin)

    selected = set(select_feature_columns(training_features))
    assert not {
        "promotion_flag",
        "promotion_intensity",
        "campaign_flag",
        "campaign_intensity",
        "market_event_flag",
        "market_condition_index",
        "plant_shutdown_flag",
        "plant_working_days",
        "weeks_since_last_shutdown",
    }.intersection(selected)


def test_lead_time_mapping_uses_sob_weighted_supplier_terms():
    scheduling = load_scheduling(ROOT / "data" / "02_scheduling_inputs.csv")
    lead_times = _lead_time_map(scheduling)

    assert len(lead_times) == len(scheduling) == 158
    assert lead_times["lead_time_days"].notna().all()
    assert lead_times["lead_time_weeks"].ge(1).all()
    assert lead_times["lead_time_weeks"].gt(14).sum() == 3


def test_fixed_random_search_is_seeded_and_equal_budget():
    search = {
        "configurations_per_objective": 3,
        "num_leaves": [15, 31, 63],
        "learning_rate": [0.03, 0.05, 0.1],
        "n_estimators": [100, 200, 300],
        "min_child_samples": [20, 50],
        "feature_fraction": [0.8, 1.0],
    }
    first = _search_configurations(search, seed=42)
    second = _search_configurations(search, seed=42)
    assert len(first) == 3
    assert first == second
    assert len({tuple(sorted(config.items())) for config in first}) == 3
    assert len(GLOBAL_MODEL_FAMILIES) == 3


def test_feature_ablation_excludes_only_named_approved_groups():
    frame = pd.DataFrame(
        columns=[
            "part_id",
            "region",
            "demand_lag_1",
            "month",
            "festival_flag",
            "promotion_flag",
            "campaign_flag",
            "promotion_intensity",
            "campaign_intensity",
            "out_stockout_flag",
        ]
    )
    features = select_feature_columns(frame, dropped_groups=("seasonality",))
    assert "demand_lag_1" in features
    assert "festival_flag" in features
    assert "promotion_flag" not in features
    assert "campaign_flag" not in features
    assert "promotion_intensity" not in features
    assert "campaign_intensity" not in features
    assert "month" not in features
    assert "out_stockout_flag" not in features
    assert "promotions" not in BASE_FEATURE_DROP_GROUPS


def test_recursive_model_feature_rows_match_shared_feature_builder():
    dates = pd.date_range("2024-01-01", periods=60, freq="W-MON")
    history = pd.DataFrame(
        {
            "part_id": ["P1"] * 60,
            "region": ["North"] * 60,
            "week_start_date": dates,
            "demand_qty": [0, 2, 0, 5, 1] * 12,
            "month": dates.month,
            "promotion_flag": [0] * 60,
        }
    )
    target_date = dates[-1] + pd.Timedelta(days=7)
    calendar = pd.DataFrame(
        {
            "part_id": ["P1"],
            "region": ["North"],
            "week_start_date": [target_date],
            "month": [target_date.month],
            "promotion_flag": [1],
        }
    )
    history_features = build_time_features(history)
    feature_columns = model_feature_columns(history_features)
    recent = history[["part_id", "region", "demand_qty"]].groupby(
        ["part_id", "region"], observed=True
    )
    histories = {
        (str(key[0]), str(key[1])): group["demand_qty"].astype(float).tolist()
        for key, group in recent
    }
    last_positive = {
        ("P1", "North"): dates[history["demand_qty"].gt(0).to_numpy().nonzero()[0][-1]]
    }

    actual = _recursive_feature_frame(
        calendar,
        histories,
        last_positive,
        feature_columns,
    ).iloc[0]
    augmented = pd.concat(
        [
            history,
            calendar.assign(demand_qty=pd.Series([pd.NA], dtype="Float64")),
        ],
        ignore_index=True,
    )
    expected = build_time_features(augmented).iloc[-1]
    for column in feature_columns:
        if pd.isna(expected[column]):
            assert pd.isna(actual[column])
        elif column in STATIC_FEATURES or column in CALENDAR_FEATURES:
            assert actual[column] == expected[column]
        else:
            assert actual[column] == pytest.approx(expected[column])


def test_analogue_fallback_uses_category_type_donors_and_prefers_region():
    dates = pd.date_range("2025-01-06", periods=13, freq="W-MON")
    history = pd.DataFrame(
        [
            {
                "part_id": part_id,
                "region": region,
                "week_start_date": date,
                "demand_qty": demand,
                "split_role": "train_val",
                "component": "Brake",
                "part_category": category,
                "vehicle_type": "Truck",
            }
            for part_id, demand, category, region in (
                ("P1", 100, "Brake", "North"),
                ("P2", 10, "Brake", "North"),
                ("P3", 20, "Brake", "North"),
                ("P4", 1000, "Engine", "North"),
                ("P5", 80, "Brake", "South"),
            )
            for date in dates
        ]
    )
    target_dates = pd.date_range(dates[-1] + pd.Timedelta(days=7), periods=2, freq="W-MON")
    calendar = pd.DataFrame(
        {
            "part_id": ["P1", "P1"],
            "region": ["North", "North"],
            "week_start_date": target_dates,
        }
    )
    metadata = pd.DataFrame(
        {
            "part_id": ["P1"],
            "region": ["North"],
            "sbc_class": ["Unclassified"],
            "component": ["Brake"],
            "part_category": ["Brake"],
            "vehicle_type": ["Truck"],
        }
    )

    forecasts = _analogue_fallback_forecasts(history, calendar, metadata)

    assert set(forecasts["forecast_qty"]) == {15.0}
    assert set(forecasts["week_start_date"]) == set(target_dates)
    assert forecasts["part_id"].eq("P1").all()
    assert ANALOGUE_FALLBACK == "analogue_category_vehicle_fallback"


def test_forecast_parquet_artifact_uses_portable_origin_date_values():
    forecasts = pd.DataFrame(
        {
            "week_start_date": pd.to_datetime(["2025-01-20"]),
            "origin_date": pd.to_datetime(["2025-01-13"]),
            "forecast_qty": [2.0],
        }
    )

    artifact = _forecast_artifact_frame(forecasts)

    assert artifact.loc[0, "origin_date"] == "2025-01-13"
    assert forecasts.loc[0, "origin_date"] == pd.Timestamp("2025-01-13")


def test_composite_candidates_are_simple_mean_and_fixed_sbc_routing():
    forecasts = pd.DataFrame(
        [
            {
                "part_id": "P1",
                "region": "North",
                "week_start_date": pd.Timestamp("2025-01-20"),
                "origin_date": pd.Timestamp("2025-01-13"),
                "evaluation_split": "evaluation",
                "horizon_step": 1,
                "actual_qty": 6.0,
                "sbc_class": "Intermittent",
                "lead_time_days": 7.0,
                "lead_time_weeks": 1,
                "mase_scale": 1.0,
                "model": model,
                "candidate_role": "forecast_candidate",
                "config_id": "test",
                "forecast_qty": float(index),
                "forecast_p90": 100.0 if model == "lightgbm_quantile" else float("nan"),
                "forecast_p95": 100.0 if model == "lightgbm_quantile" else float("nan"),
                "forecast_p99": 100.0 if model == "lightgbm_quantile" else float("nan"),
                "ablation_group": "none",
                "runtime_seconds": 0.1,
            }
            for index, model in enumerate(SIMPLE_AVERAGE_MEMBERS, start=1)
        ]
    )

    output = _append_composite_candidates(
        forecasts,
        {
            "Intermittent": "sba",
        },
    )
    ensemble = output.loc[output["model"].eq(SIMPLE_AVERAGE_ENSEMBLE)].iloc[0]
    router = output.loc[output["model"].eq(SBC_ROUTER)].iloc[0]

    assert ensemble["forecast_qty"] == pytest.approx(5.0)
    assert pd.isna(ensemble["forecast_p90"])
    assert router["forecast_qty"] == pytest.approx(5.0)
    assert router["config_id"] == "fixed_sbc_model_map"
