from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = (ROOT / "SPEC.md").read_text(encoding="utf-8")


def test_phase0_contract_records_verified_inputs_and_holdout():
    assert "120,861 rows × 62 columns" in SPEC
    assert "47 series short/unclassified" in SPEC
    assert "final-test labels unopened until their one-time final evaluation" in SPEC
    assert "SPEC v4.1" in SPEC
    assert "32 PASS, 0 FAIL" in SPEC
    assert "reports/phase0_traceability.md" in (ROOT / "STATE.md").read_text(encoding="utf-8")
    assert "reports/phase0_environment.md" in (ROOT / "STATE.md").read_text(encoding="utf-8")
    assert (ROOT / "docs" / "Spec.md").is_file()
    assert (ROOT / "docs" / "SPEC.docx").is_file()


def test_phase0_contract_defines_metrics_and_track_disqualifiers():
    for text in (
        "Forecast Accuracy %",
        "Forecast-versus-actual variance",
        "Quantile coverage is `mean(actual <= quantile forecast)`",
        "Any hard-constraint violation disqualifies a scheduler.",
        "Incorrect auto-apply and invented-quantity rates must be zero.",
        "A champion is supported when it wins at least 75% of perturbations",
    ):
        assert text in SPEC


def test_phase0_traceability_and_runtime_configuration_exist():
    report = (ROOT / "reports" / "phase0_traceability.md").read_text(encoding="utf-8")
    assert "Sixteen influencing factors" in report
    assert "Ten original KPIs" in report
    assert "Updated assistant, controls and audit requirements" in report
    assert "Execution plan" in report
    assert (ROOT / "reports" / "phase0_environment.md").is_file()
    assert (ROOT / "configs" / "project.yaml").is_file()
    assert (ROOT / "requirements.txt").is_file()
    assert "Not measurable" in report
    assert report.count("| Updated §Key Factors") + report.count("| Updated §Key Factors;") >= 16


def test_phase0_cost_proxies_and_business_thresholds_are_explicit():
    business = yaml.safe_load(
        (ROOT / "configs" / "business.yaml").read_text(encoding="utf-8")
    )
    costs = business["scheduling"]["costs"]
    assert costs["emergency_procurement"]["unit_cost_multiplier"] == 1.5
    assert costs["emergency_procurement"]["sensitivity_range"] == [1.25, 2.0]
    assert costs["stockout_penalty"]["criticality_multipliers"] == {
        "High": 10.0,
        "Medium": 5.0,
        "Low": 2.0,
    }
    assert costs["downtime"]["basis"] == "not_measurable"


def test_application_architecture_uses_one_django_backend_and_imported_ml():
    config = yaml.safe_load((ROOT / "configs" / "project.yaml").read_text(encoding="utf-8"))
    application = config["application"]

    assert application["backend"] == "django_drf"
    assert application["database"] == "sqlite_development"
    assert application["backend_count"] == 1
    assert application["ml_execution"] == "imported_python_packages_in_process"
    assert application["ml_package_root"] == "src"
    assert application["separate_ml_http_service"] is False
    assert application["task_queue"] is False
    assert "Django + DRF" in SPEC
    assert "No second backend" in SPEC

    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "django==5.2.18" in requirements
    assert "djangorestframework==3.18.3" in requirements
    assert "djangorestframework-simplejwt==5.5.1" in requirements
    assert "mysqlclient" not in requirements
    assert "fastapi" not in requirements
    assert "uvicorn" not in requirements


def test_v41_forecast_weights_and_selection_gates_are_configured():
    config = yaml.safe_load((ROOT / "configs" / "project.yaml").read_text(encoding="utf-8"))
    tracks = config["scoring"]["tracks"]
    forecast = tracks["forecasting"]
    weights = forecast["weights"]
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights == {
        "overall_error": 0.20,
        "critical_a_class_error": 0.20,
        "intermittent_lumpy": 0.15,
        "asymmetric_absolute_bias": 0.15,
        "calibration": 0.15,
        "stability_across_origins": 0.10,
        "runtime_and_explainability": 0.05,
    }
    assert forecast["gates"]["enforce"]["baseline_beating"] is True
    assert config["forecast"]["backtest"]["over_horizon_extension_period_weeks"] == 4
    expected = {
        "forecasting": {
            "overall_error": 0.20,
            "critical_a_class_error": 0.20,
            "intermittent_lumpy": 0.15,
            "asymmetric_absolute_bias": 0.15,
            "calibration": 0.15,
            "stability_across_origins": 0.10,
            "runtime_and_explainability": 0.05,
        },
        "scheduling": {
            "critical_part_availability": 0.25,
            "overall_fill_rate_and_stockout": 0.15,
            "emergency_procurement": 0.15,
            "excess_inventory_and_carrying_cost": 0.20,
            "schedule_adherence_and_stability": 0.15,
            "inventory_turnover": 0.10,
        },
        "assistant": {
            "intent_accuracy": 0.25,
            "entity_accuracy": 0.25,
            "rule_validation_correctness": 0.20,
            "multilingual_consistency": 0.15,
            "audit_completeness": 0.10,
            "latency_cost": 0.05,
        },
    }
    for track_name, exact_weights in expected.items():
        actual_weights = tracks[track_name]["weights"]
        assert actual_weights == exact_weights
        assert sum(actual_weights.values()) == pytest.approx(1.0)

    assert config["application"]["database"] == "sqlite_development"


def test_configured_business_cost_proxies_and_languages_are_provisional():
    config = yaml.safe_load((ROOT / "configs" / "business.yaml").read_text(encoding="utf-8"))
    costs = config["scheduling"]["costs"]
    assert costs["emergency_procurement"]["unit_cost_multiplier"] == 1.5
    assert costs["emergency_procurement"]["sensitivity_range"] == [1.25, 2.0]
    assert costs["emergency_procurement"]["status"] == "provisional_editable"
    assert costs["stockout_penalty"]["criticality_multipliers"] == {
        "High": 10.0,
        "Medium": 5.0,
        "Low": 2.0,
    }
    assert costs["stockout_penalty"]["relative_sensitivity"] == 0.50
    assert costs["carrying_cost"]["annual_fallback_rate"] == 0.25
    assert costs["carrying_cost"]["fallback_condition"] == "only_when_source_rate_is_absent"
    assert config["assistant"]["languages"] == [
        "English", "Hindi", "Hinglish", "Marathi"
    ]
    assert config["assistant"]["optional_languages"] == ["Tamil"]
