"""Compare a proposed Track A score policy with the frozen Phase 4 baseline."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.data.loader import ROOT
from src.forecasting.phase4 import _sensitivity, _weighted_selection


PROPOSED_WEIGHTS = {
    "wape": 0.20,
    "absolute_bias": 0.15,
    "a_class_critical_parts": 0.20,
    "intermittent_lumpy": 0.20,
    "calibration": 0.08,
    "stability_across_origins": 0.07,
    "runtime": 0.03,
    "explainability": 0.07,
}
SCENARIOS = {
    "frozen_baseline": None,
    "proposed_business": PROPOSED_WEIGHTS,
}
RANKING_COLUMNS = [
    "selection_rank",
    "candidate_id",
    "model",
    "selection_score",
    "wape_mixed_pct",
    "absolute_bias_mixed_pct",
    "a_class_critical_parts_wape_pct",
    "intermittent_lumpy_wape_pct",
    "calibration_score",
    "origin_lead_time_wape_std_pct",
    "runtime_seconds",
    "explainability_tier",
]


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def run_weight_scenario_comparison(root: Path = ROOT) -> dict[str, object]:
    """Re-score saved Phase 4 validation metrics without refitting candidates."""
    config = _read_yaml(root / "configs" / "project.yaml")
    scoring = config["scoring"]["tracks"]["forecasting"]
    baseline_weights = {
        name: float(weight) for name, weight in scoring["weights"].items()
    }
    if not abs(sum(baseline_weights.values()) - 1.0) < 1e-9:
        raise ValueError("Frozen Track A weights must sum to 1.0")
    if not abs(sum(PROPOSED_WEIGHTS.values()) - 1.0) < 1e-9:
        raise ValueError("Proposed Track A weights must sum to 1.0")
    if baseline_weights.keys() != PROPOSED_WEIGHTS.keys():
        raise ValueError("Baseline and proposed components must match")

    metrics = pd.read_csv(root / "results" / "track_a_selection_matrix.csv")
    tiers = config["phase4"]["explainability_tier"]
    lead_share = float(scoring["lead_time_calibration_share"])
    weekly_share = float(scoring["weekly_calibration_share"])
    ranked_scenarios: dict[str, pd.DataFrame] = {}
    sensitivity_scenarios: dict[str, pd.DataFrame] = {}

    for scenario, configured_weights in SCENARIOS.items():
        weights = (
            baseline_weights
            if configured_weights is None
            else configured_weights
        )
        ranked = _weighted_selection(
            metrics,
            weights,
            tiers,
            lead_share,
            weekly_share,
        )
        sensitivity = _sensitivity(ranked, weights, tiers, 0.10)
        ranked_scenarios[scenario] = ranked
        sensitivity_scenarios[scenario] = sensitivity

    baseline_scores = (
        metrics.set_index("candidate_id")["selection_score"].sort_index()
    )
    rescored_baseline = (
        ranked_scenarios["frozen_baseline"]
        .set_index("candidate_id")["selection_score"]
        .sort_index()
    )
    if not baseline_scores.index.equals(rescored_baseline.index) or (
        (baseline_scores - rescored_baseline).abs().max() > 1e-8
    ):
        raise ValueError("Re-scoring failed to reproduce the frozen baseline")

    prior_sensitivity = pd.read_csv(
        root / "results" / "track_a_weight_sensitivity.csv"
    )
    prior_sensitivity = prior_sensitivity.sort_values(
        ["perturbed_component", "direction"], kind="stable"
    ).reset_index(drop=True)
    reproduced_sensitivity = (
        sensitivity_scenarios["frozen_baseline"]
        .sort_values(["perturbed_component", "direction"], kind="stable")
        .reset_index(drop=True)
    )
    if not prior_sensitivity["winner"].equals(reproduced_sensitivity["winner"]):
        raise ValueError("Re-scored baseline sensitivity differs from saved results")

    results_dir = root / "results"
    ranking_frames = []
    sensitivity_frames = []
    weight_rows = []
    for scenario, ranked in ranked_scenarios.items():
        weights = (
            baseline_weights
            if scenario == "frozen_baseline"
            else PROPOSED_WEIGHTS
        )
        frame = ranked.loc[:, RANKING_COLUMNS].copy()
        frame.insert(0, "scenario", scenario)
        frame["scenario_weights"] = ";".join(
            f"{name}={weights[name]:.2f}" for name in weights
        )
        ranking_frames.append(frame)

        sensitivity = sensitivity_scenarios[scenario].copy()
        sensitivity.insert(0, "scenario", scenario)
        sensitivity_frames.append(sensitivity)

    for component, baseline_weight in baseline_weights.items():
        proposed_weight = PROPOSED_WEIGHTS[component]
        weight_rows.append(
            {
                "component": component,
                "frozen_baseline_weight_pct": round(baseline_weight * 100, 6),
                "proposed_weight_pct": round(proposed_weight * 100, 6),
                "change_percentage_points": round(
                    (proposed_weight - baseline_weight) * 100,
                    6,
                ),
            }
        )

    rankings = pd.concat(ranking_frames, ignore_index=True)
    sensitivity = pd.concat(sensitivity_frames, ignore_index=True)
    rankings.to_csv(
        results_dir / "track_a_weight_scenario_rankings.csv", index=False
    )
    sensitivity.to_csv(
        results_dir / "track_a_weight_scenario_sensitivity.csv", index=False
    )
    pd.DataFrame(weight_rows).to_csv(
        results_dir / "track_a_weight_scenario_weights.csv", index=False
    )

    baseline = ranked_scenarios["frozen_baseline"].set_index("candidate_id")
    proposed = ranked_scenarios["proposed_business"].set_index("candidate_id")
    comparison = pd.DataFrame(
        {
            "candidate_id": baseline.index,
            "frozen_baseline_rank": baseline["selection_rank"].to_numpy(),
            "frozen_baseline_score": baseline["selection_score"].to_numpy(),
            "proposed_rank": proposed.loc[baseline.index][
                "selection_rank"
            ].to_numpy(),
            "proposed_score": proposed.loc[baseline.index][
                "selection_score"
            ].to_numpy(),
        }
    )
    comparison["rank_change_proposed_minus_baseline"] = (
        comparison["proposed_rank"] - comparison["frozen_baseline_rank"]
    )
    comparison["score_change_proposed_minus_baseline"] = (
        comparison["proposed_score"] - comparison["frozen_baseline_score"]
    )
    comparison.to_csv(
        results_dir / "track_a_weight_scenario_comparison.csv", index=False
    )

    return {
        "candidate_count": len(comparison),
        "baseline_winner": str(
            ranked_scenarios["frozen_baseline"].iloc[0]["candidate_id"]
        ),
        "proposed_winner": str(
            ranked_scenarios["proposed_business"].iloc[0]["candidate_id"]
        ),
        "baseline_winner_score": float(
            ranked_scenarios["frozen_baseline"].iloc[0]["selection_score"]
        ),
        "proposed_winner_score": float(
            ranked_scenarios["proposed_business"].iloc[0]["selection_score"]
        ),
        "baseline_sensitivity_changes": int(
            sensitivity_scenarios["frozen_baseline"]["winner_changed"].sum()
        ),
        "proposed_sensitivity_changes": int(
            sensitivity_scenarios["proposed_business"]["winner_changed"].sum()
        ),
    }


if __name__ == "__main__":
    print(run_weight_scenario_comparison())
