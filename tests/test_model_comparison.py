from __future__ import annotations

import pandas as pd

from src.forecasting.model_comparison import (
    CHAMPION_LABEL,
    _decide,
    _total_runtime,
    _xgboost_configurations,
)


def test_xgboost_budget_matches_frozen_search_size():
    search = {
        "configurations_per_objective": 3,
        "num_leaves": [15, 31, 63],
        "learning_rate": [0.03, 0.05, 0.1],
        "n_estimators": [100, 200, 300],
        "min_child_samples": [20, 50],
        "feature_fraction": [0.8, 1.0],
    }
    configurations = _xgboost_configurations(search, 42)
    assert len(configurations) == 3
    for configuration in configurations:
        assert configuration["max_depth"] >= 2
        assert configuration["subsample"] == 1.0
        assert configuration["min_child_weight"] in {20.0, 50.0}
        assert 0.8 <= configuration["colsample_bytree"] <= 1.0


def _tuning_table(candidate_wape: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": CHAMPION_LABEL,
                "candidate_role": "champion",
                "evaluation_level": "lead_time",
                "wape_pct": 39.29,
            },
            {
                "candidate_id": "xgboost::random_1",
                "candidate_role": "new_candidate",
                "evaluation_level": "lead_time",
                "wape_pct": candidate_wape,
            },
        ]
    )


def test_tuning_gate_requires_half_point_lead_time_gain():
    miss = _decide(_tuning_table(38.85), 0.50)
    assert miss["tuning_gate_passed"] is False
    assert miss["held_evaluation_opened"] is False
    assert miss["decision"] == "champion_retained_tuning_gate_not_met"

    hit = _decide(_tuning_table(38.70), 0.50)
    assert hit["tuning_gate_passed"] is True
    assert hit["best_candidate"] == "xgboost::random_1"


def test_total_runtime_returns_nan_when_artifact_has_no_runtime():
    frame = pd.DataFrame(
        {
            "origin_date": pd.to_datetime(["2025-01-06", "2025-01-06"]),
            "runtime_seconds": [float("nan"), float("nan")],
        }
    )
    assert pd.isna(_total_runtime(frame))
