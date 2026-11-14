"""Fold-local global LightGBM fitting and recursive multi-step prediction."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.features.time_features import (
    CALENDAR_FEATURES,
    STATIC_FEATURES,
    build_time_features,
    model_feature_columns,
)


BASE_FEATURE_DROP_GROUPS: dict[str, tuple[str, ...]] = {
    "seasonality": (
        "year", "month", "quarter_fy", "fiscal_year", "week_of_year",
        "week_of_month", "demand_month_sin", "demand_month_cos",
        "demand_lag_52", "demand_same_period_last_year",
    ),
    "festivals": (
        "festival_flag", "festival_name", "festival_intensity",
        "weeks_to_next_festival", "weeks_since_last_festival",
    ),
}
UNVERIFIED_FUTURE_FEATURES = frozenset(
    {
        "promotion_flag",
        "promotion_intensity",
        "campaign_flag",
        "campaign_intensity",
    }
)
UNAVAILABLE_ABLATIONS = {
    "market": "not_run_origin_availability: future market events/index are not verified as known at forecast origin",
    "shutdowns": "not_run_origin_availability: future plant shutdown/working-day values are not verified as known at forecast origin",
    "promotions": "not_run_origin_availability: historical promotion/campaign calendar does not identify which plans were published by each backtest origin",
}


def select_feature_columns(
    feature_frame: pd.DataFrame,
    dropped_groups: Sequence[str] = (),
) -> list[str]:
    columns = model_feature_columns(feature_frame)
    dropped = {
        column
        for group in dropped_groups
        for column in BASE_FEATURE_DROP_GROUPS.get(group, ())
    }
    invalid_groups = set(dropped_groups) - set(BASE_FEATURE_DROP_GROUPS)
    if invalid_groups:
        raise ValueError(f"Unsupported feature ablation groups: {sorted(invalid_groups)}")
    return [
        column
        for column in columns
        if column not in dropped and column not in UNVERIFIED_FUTURE_FEATURES
    ]


def _categorical_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if (
            pd.api.types.is_string_dtype(frame[column].dtype)
            or pd.api.types.is_object_dtype(frame[column].dtype)
            or isinstance(frame[column].dtype, pd.CategoricalDtype)
        )
    ]


def _ratio_where_positive(
    numerator: pd.Series, denominator: pd.Series
) -> pd.Series:
    ratio = (numerator / denominator.where(denominator > 0)).replace(
        [np.inf, -np.inf], np.nan
    )
    return ratio.clip(lower=0.0, upper=10.0)


def _observed_mean(values: np.ndarray) -> float:
    observed = values[np.isfinite(values)]
    return float(np.mean(observed)) if len(observed) else float("nan")


def _add_series_trend_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add origin-safe trend and intermittency features within each series."""
    output = frame.copy()
    keys = ["part_id", "region"]
    output["demand_trend_4_13"] = _ratio_where_positive(
        output["demand_rolling_4_mean"], output["demand_rolling_13_mean"]
    )
    output["demand_trend_13_52"] = _ratio_where_positive(
        output["demand_rolling_13_mean"], output["demand_rolling_52_mean"]
    )
    output["demand_cv_13"] = _ratio_where_positive(
        output["demand_rolling_13_std"], output["demand_rolling_13_mean"]
    )
    grouped = output.groupby(keys, sort=False, observed=True)
    prior_four = grouped["demand_rolling_4_mean"].shift(4)
    output["demand_growth_4"] = _ratio_where_positive(
        output["demand_rolling_4_mean"], prior_four
    )

    audit = output.loc[:, [*keys, "demand_qty"]].copy()
    audit["observed"] = audit["demand_qty"].notna().astype(float)
    audit["zero_flag"] = (
        audit["demand_qty"].eq(0).astype(float).where(audit["demand_qty"].notna())
    )
    audit_grouped = audit.groupby(keys, sort=False, observed=True)
    audit["prior_zero"] = audit_grouped["zero_flag"].shift(1)
    audit["prior_observed"] = audit_grouped["observed"].shift(1)
    rolling_grouped = audit.groupby(keys, sort=False, observed=True)
    audit["demand_zero_rate_13"] = rolling_grouped["prior_zero"].transform(
        lambda values: values.rolling(13, min_periods=1).mean()
    )
    audit["demand_history_weeks"] = rolling_grouped["prior_observed"].transform(
        lambda values: values.cumsum()
    )
    output["demand_zero_rate_13"] = audit["demand_zero_rate_13"].reindex(
        output.index
    )
    output["demand_history_weeks"] = audit["demand_history_weeks"].reindex(
        output.index
    )
    return output


def _lightgbm_model(
    objective: str,
    parameters: Mapping[str, float | int],
    seed: int,
    quantile: float | None = None,
):
    try:
        from lightgbm import LGBMRegressor
    except ImportError as error:
        raise ImportError(
            "LightGBM is required for Track A Tier 1 global models; "
            "install the declared lightgbm dependency."
        ) from error

    model_parameters: dict[str, object] = {
        **parameters,
        "objective": objective,
        "n_jobs": 1,
        "random_state": seed,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
        "importance_type": "gain",
    }
    if objective == "tweedie":
        model_parameters["tweedie_variance_power"] = 1.5
    if objective == "quantile":
        if quantile is None:
            raise ValueError("Quantile objective requires a quantile level")
        model_parameters["alpha"] = quantile
    return LGBMRegressor(**model_parameters)


def _recursive_feature_frame(
    current_calendar: pd.DataFrame,
    histories: dict[tuple[str, str], list[float]],
    last_positive_dates: dict[tuple[str, str], pd.Timestamp | None],
    feature_columns: Sequence[str],
    extra_features: bool = False,
) -> pd.DataFrame:
    """Build only the current week's recursive features for each series."""
    records: list[dict[str, object]] = []
    static_calendar = set(STATIC_FEATURES) | set(CALENDAR_FEATURES)
    for row in current_calendar.itertuples(index=False, name=None):
        values = dict(zip(current_calendar.columns, row))
        key = (str(values["part_id"]), str(values["region"]))
        past = np.asarray(histories.get(key, []), dtype=float)
        date = pd.Timestamp(values["week_start_date"])
        features: dict[str, object] = {
            column: values[column]
            for column in current_calendar.columns
            if column in static_calendar
        }
        features.update(
            {
                "part_id": values["part_id"],
                "region": values["region"],
                "week_start_date": date,
                "demand_qty": np.nan,
            }
        )
        for lag in (1, 4, 13, 52):
            features[f"demand_lag_{lag}"] = (
                float(past[-lag]) if len(past) >= lag else np.nan
            )
        features["demand_same_period_last_year"] = (
            float(past[-52]) if len(past) >= 52 else np.nan
        )
        for window in (4, 13, 52):
            window_values = past[-window:]
            observed = window_values[np.isfinite(window_values)]
            features[f"demand_rolling_{window}_mean"] = (
                float(np.mean(observed)) if len(observed) else np.nan
            )
            features[f"demand_rolling_{window}_median"] = (
                float(np.median(observed)) if len(observed) else np.nan
            )
            features[f"demand_rolling_{window}_std"] = (
                float(np.std(observed, ddof=1)) if len(observed) > 1 else np.nan
            )
        zero_run = 0
        for value in past[::-1]:
            if not np.isfinite(value) or value > 0:
                break
            zero_run += 1
        features["demand_zero_run_length"] = float(zero_run)
        last_positive = last_positive_dates.get(key)
        features["weeks_since_last_demand"] = (
            float((date - last_positive).days // 7)
            if last_positive is not None
            else np.nan
        )
        if extra_features:
            def safe_ratio(numerator: float, denominator: float) -> float:
                if (
                    np.isfinite(numerator)
                    and np.isfinite(denominator)
                    and denominator > 0
                ):
                    return float(np.clip(numerator / denominator, 0.0, 10.0))
                return float("nan")

            mean_4 = features["demand_rolling_4_mean"]
            mean_13 = features["demand_rolling_13_mean"]
            mean_52 = features["demand_rolling_52_mean"]
            features["demand_trend_4_13"] = safe_ratio(mean_4, mean_13)
            features["demand_trend_13_52"] = safe_ratio(mean_13, mean_52)
            prior_four = _observed_mean(past[-8:-4])
            features["demand_growth_4"] = safe_ratio(mean_4, prior_four)
            features["demand_cv_13"] = safe_ratio(
                features["demand_rolling_13_std"], mean_13
            )
            observed_13 = past[-13:]
            observed_13 = observed_13[np.isfinite(observed_13)]
            features["demand_zero_rate_13"] = (
                float(np.mean(observed_13 == 0)) if len(observed_13) else np.nan
            )
            features["demand_history_weeks"] = float(np.isfinite(past).sum())
        angle = 2.0 * np.pi * date.month / 12.0
        features["demand_month_sin"] = float(np.sin(angle))
        features["demand_month_cos"] = float(np.cos(angle))
        records.append(features)
    output = pd.DataFrame(records)
    return output.loc[:, list(dict.fromkeys(
        [*feature_columns, "part_id", "region", "week_start_date", "demand_qty"]
    ))]


def build_fold_training_data(
    history: pd.DataFrame,
    origin: pd.Timestamp,
    extra_features: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the shared, origin-safe train design once per rolling fold."""
    history = history.loc[
        history["split_role"].isin(["warmup", "train_val"])
        & history["week_start_date"].le(origin)
    ].copy()
    history = history.sort_values(
        ["part_id", "region", "week_start_date"], kind="stable"
    ).reset_index(drop=True)
    required_columns = list(dict.fromkeys(
        column for column in (
            "part_id",
            "region",
            "week_start_date",
            "demand_qty",
            "split_role",
            *STATIC_FEATURES,
            *CALENDAR_FEATURES,
        )
        if column in history.columns
    ))
    history = history.loc[:, required_columns]
    history["demand_qty"] = pd.to_numeric(
        history["demand_qty"], errors="raise"
    ).astype("Float64")
    features = build_time_features(history)
    if extra_features:
        features = _add_series_trend_features(features)
    target_mask = (
        history["split_role"].eq("train_val")
        & history["demand_qty"].notna()
        & history["week_start_date"].le(origin)
    )
    train_features = features.loc[target_mask.to_numpy()].copy()
    train_target = history.loc[target_mask, "demand_qty"].astype(float)
    if train_features.empty:
        raise ValueError(f"No observed train_val fitting rows available by {origin.date()}")
    return train_features, train_target


def fit_global_models(
    history: pd.DataFrame,
    origin: pd.Timestamp,
    target_calendar: pd.DataFrame,
    parameters: Mapping[str, float | int | None],
    objective_family: str,
    seed: int,
    dropped_groups: Sequence[str] = (),
    quantiles: Sequence[float] = (0.50, 0.90, 0.95, 0.99),
    training_design: tuple[pd.DataFrame, pd.Series] | None = None,
    extra_features: bool = False,
) -> tuple[pd.DataFrame, dict[float, object], int]:
    """Fit on observed train_val targets, then recursively forecast the calendar."""
    if objective_family not in {
        "lightgbm_point",
        "lightgbm_tweedie",
        "lightgbm_poisson",
        "lightgbm_quantile",
        "random_forest",
        "xgboost",
    }:
        raise ValueError(f"Unsupported global model family: {objective_family}")
    history = history.loc[
        history["split_role"].isin(["warmup", "train_val"])
        & history["week_start_date"].le(origin)
    ].copy()
    history = history.sort_values(
        ["part_id", "region", "week_start_date"], kind="stable"
    ).reset_index(drop=True)
    required_history_columns = list(dict.fromkeys(
        column for column in (
            "part_id",
            "region",
            "week_start_date",
            "demand_qty",
            "split_role",
            *STATIC_FEATURES,
            *CALENDAR_FEATURES,
        )
        if column in history.columns
    ))
    history = history.loc[:, required_history_columns]
    history["demand_qty"] = pd.to_numeric(
        history["demand_qty"], errors="raise"
    ).astype("Float64")
    if training_design is None:
        train_features, train_target = build_fold_training_data(
            history, origin, extra_features=extra_features
        )
    else:
        train_features, train_target = training_design

    feature_columns = select_feature_columns(
        train_features, dropped_groups=dropped_groups
    )
    future_calendar = target_calendar.copy()
    available_calendar_columns = list(dict.fromkeys(
        column for column in (
            "part_id", "region", "week_start_date",
            *STATIC_FEATURES, *CALENDAR_FEATURES,
        )
        if column in future_calendar.columns
    ))
    future_calendar = future_calendar.loc[:, available_calendar_columns].copy()

    if future_calendar.empty:
        raise ValueError("Forecast calendar is empty")

    train_x_reference = train_features.loc[:, feature_columns]
    categorical = _categorical_columns(train_x_reference)
    categories_by_column: dict[str, list[str]] = {}
    prepared_train = train_x_reference.copy()
    if objective_family in {"random_forest", "xgboost"}:
        transformers = []
        if categorical:
            for column in categorical:
                prepared_train[column] = (
                    prepared_train[column].astype("string").fillna("__MISSING__")
                )
            transformers.append(
                (
                    "categorical",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                    categorical,
                )
            )
        numeric = [column for column in feature_columns if column not in categorical]
        for column in numeric:
            prepared_train[column] = pd.to_numeric(
                prepared_train[column], errors="raise"
            ).astype("float32")
        preprocessor = ColumnTransformer(
            transformers,
            remainder=SimpleImputer(
                strategy="median", keep_empty_features=True
            ),
            verbose_feature_names_out=False,
        )
    else:
        for column in feature_columns:
            if column in categorical:
                values = prepared_train[column].astype("string").fillna("__MISSING__")
                categories = sorted(values.unique().tolist())
                categories_by_column[column] = categories
                prepared_train[column] = pd.Categorical(values, categories=categories)
            else:
                prepared_train[column] = pd.to_numeric(
                    prepared_train[column], errors="raise"
                ).astype("float32")

    quantile_levels = (
        tuple(float(q) for q in quantiles)
        if objective_family == "lightgbm_quantile"
        else (None,)
    )
    models: dict[float, object] = {}
    for quantile in quantile_levels:
        if objective_family == "random_forest":
            model = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    (
                        "regressor",
                        RandomForestRegressor(
                            **parameters,
                            n_jobs=1,
                            random_state=seed,
                        ),
                    ),
                ]
            )
        elif objective_family == "xgboost":
            from xgboost import XGBRegressor

            model = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    (
                        "regressor",
                        XGBRegressor(
                            **parameters,
                            objective="reg:squarederror",
                            tree_method="hist",
                            n_jobs=1,
                            random_state=seed,
                        ),
                    ),
                ]
            )
        else:
            objective = (
                "regression"
                if objective_family == "lightgbm_point"
                else "tweedie"
                if objective_family == "lightgbm_tweedie"
                else "poisson"
                if objective_family == "lightgbm_poisson"
                else "quantile"
            )
            model = _lightgbm_model(objective, parameters, seed, quantile)
        model.fit(
            prepared_train,
            train_target.to_numpy(dtype=float),
            **(
                {}
                if objective_family in {"random_forest", "xgboost"}
                else {"categorical_feature": categorical}
            ),
        )
        models[float(quantile) if quantile is not None else 0.50] = model

    histories: dict[tuple[str, str], list[float]] = {}
    last_positive_dates: dict[tuple[str, str], pd.Timestamp | None] = {}
    for key, group in history.groupby(
        ["part_id", "region"], sort=False, observed=True
    ):
        normalized_key = (str(key[0]), str(key[1]))
        histories[normalized_key] = group["demand_qty"].to_numpy(
            dtype=float, na_value=np.nan
        ).tolist()
        positive = group.loc[
            group["demand_qty"].gt(0), "week_start_date"
        ]
        last_positive_dates[normalized_key] = (
            pd.Timestamp(positive.iloc[-1]) if not positive.empty else None
        )
    forecast_rows: list[dict[str, object]] = []
    dates = pd.DatetimeIndex(
        sorted(future_calendar["week_start_date"].dropna().unique())
    )
    for target_date in dates:
        current_calendar = future_calendar.loc[
            future_calendar["week_start_date"].eq(target_date)
        ].sort_values(["part_id", "region"], kind="stable").reset_index(drop=True)
        future_features = _recursive_feature_frame(
            current_calendar,
            histories,
            last_positive_dates,
            feature_columns,
            extra_features=extra_features,
        )
        predict_x = future_features.loc[:, feature_columns].copy()
        for column in feature_columns:
            if objective_family in {"random_forest", "xgboost"} and column in categorical:
                predict_x[column] = (
                    predict_x[column].astype("string").fillna("__MISSING__")
                )
            elif column in categories_by_column:
                values = predict_x[column].astype("string").fillna("__MISSING__")
                predict_x[column] = pd.Categorical(
                    values, categories=categories_by_column[column]
                )
            else:
                predict_x[column] = pd.to_numeric(
                    predict_x[column], errors="raise"
                ).astype("float32")
        quantile_values = {
            quantile: np.maximum(0.0, model.predict(predict_x))
            for quantile, model in models.items()
        }
        if objective_family == "lightgbm_quantile":
            quantile_matrix = np.column_stack(
                [quantile_values[float(q)] for q in sorted(quantile_values)]
            )
            quantile_matrix.sort(axis=1)
            quantile_values = {
                q: quantile_matrix[:, index]
                for index, q in enumerate(sorted(models))
            }
            point = quantile_values[0.50]
        else:
            point = quantile_values[0.50]
        for row_index, row in current_calendar.reset_index(drop=True).iterrows():
            record = {
                column: row[column]
                for column in ("part_id", "region", "week_start_date")
            }
            record["forecast_qty"] = float(point[row_index])
            for quantile in (0.90, 0.95, 0.99):
                record[f"forecast_p{int(quantile * 100)}"] = (
                    float(quantile_values[quantile][row_index])
                    if quantile in quantile_values
                    else np.nan
                )
            forecast_rows.append(record)
            key = (str(row["part_id"]), str(row["region"]))
            histories.setdefault(key, []).append(float(point[row_index]))
            if point[row_index] > 0:
                last_positive_dates[key] = pd.Timestamp(target_date)

    return pd.DataFrame(forecast_rows), models, len(train_target)
