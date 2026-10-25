from pathlib import Path

import pandas as pd

from src.data.loader import load_demand
from src.features.segmentation import build_segment_table


ROOT = Path(__file__).resolve().parents[1]


def test_development_segments_match_generator_classes_and_route_fallbacks():
    demand = load_demand(ROOT / "data" / "01_demand_forecasting.csv")
    segments = build_segment_table(demand)

    assert len(segments) == 616
    assert (
        segments["sbc_class"].fillna("Unclassified").value_counts().to_dict()
        == {
            "Lumpy": 220,
            "Intermittent": 133,
            "Smooth": 119,
            "Erratic": 97,
            "Unclassified": 47,
        }
    )
    comparable = segments["supplied_class_agrees"].dropna()
    assert len(comparable) == 569
    assert comparable.all()
    fallbacks = segments.loc[segments["analogue_fallback"]]
    assert len(fallbacks) == 47
    assert fallbacks["analogue_component"].notna().all()
    assert fallbacks["analogue_vehicle_type"].notna().all()


def test_phase2_outputs_exist_and_report_respects_the_train_validation_cutoff():
    report_path = ROOT / "results" / "phase2_eda_report.md"
    segments_path = ROOT / "artifacts" / "series_segments.parquet"
    dictionary_path = ROOT / "reports" / "feature_dictionary_phase2.csv"
    chart_paths = [
        ROOT / "results" / "phase2_charts" / name
        for name in (
            "weekly_demand_by_region.png",
            "month_of_year_seasonality.png",
            "event_demand_differences.png",
            "sbc_segment_counts.png",
            "vehicle_parc_vs_demand.png",
        )
    ]

    assert report_path.is_file()
    assert segments_path.is_file()
    assert dictionary_path.is_file()
    assert all(path.is_file() for path in chart_paths)

    report = report_path.read_text(encoding="utf-8")
    assert "2026-03-23" in report
    assert "final_test" in report
    assert "includes warm-up" in report
    assert "100.000%" in report
    segments = pd.read_parquet(segments_path)
    feature_dictionary = pd.read_csv(dictionary_path)
    assert len(segments) == 616
    assert len(feature_dictionary) > 0
