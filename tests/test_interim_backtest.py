from __future__ import annotations

import pandas as pd

from src.forecasting.interim_backtest import (
    HORIZON,
    ORIGINS,
    WINDOW_END,
    _aggregate_26w,
    _score,
)


def test_interim_origins_are_consecutive_and_end_at_sealed_boundary():
    assert ORIGINS[1] - ORIGINS[0] == pd.Timedelta(weeks=HORIZON)
    assert ORIGINS[1] + pd.Timedelta(weeks=HORIZON) == WINDOW_END
    assert WINDOW_END == pd.Timestamp("2026-03-23")


def test_aggregate_26w_counts_observed_and_forecast_weeks():
    rows = []
    for week in range(HORIZON):
        rows.append(
            {
                "origin_date": pd.Timestamp("2025-03-24"),
                "part_id": "P1",
                "region": "East",
                "actual_qty": 1.0,
                "forecast_qty": 1.0,
            }
        )
    rows.append(
        {
            "origin_date": pd.Timestamp("2025-03-24"),
            "part_id": "P2",
            "region": "East",
            "actual_qty": 2.0,
            "forecast_qty": None,
        }
    )
    grouped = _aggregate_26w(pd.DataFrame(rows))
    complete = grouped.loc[grouped["part_id"].eq("P1")].iloc[0]
    assert complete["observed_weeks"] == HORIZON
    assert complete["forecast_weeks"] == HORIZON
    assert complete["actual_qty"] == HORIZON
    assert pd.isna(complete["mase_scale"])


def test_score_counts_exclusions_separately():
    frame = pd.DataFrame(
        {
            "actual_qty": [1.0, 2.0, None, None],
            "forecast_qty": [1.0, None, 3.0, None],
            "mase_scale": [1.0, 1.0, 1.0, 1.0],
        }
    )
    row = _score("weekly", "pooled", "all", "m", frame)
    assert row["n_scored"] == 1
    assert row["excluded_missing_actual"] == 1
    assert row["excluded_missing_forecast"] == 1
    assert row["excluded_both_missing"] == 1
    assert row["wape_pct"] == 0.0
