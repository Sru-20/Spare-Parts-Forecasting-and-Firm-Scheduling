import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.classification.part_category import (
    GROUP_COLUMN,
    TARGET,
    TEXT_FIELDS,
    PartCategoryPredictor,
    _oof_grouped_predictions,
    _group_split,
    build_text,
)
from src.data.loader import ROOT, load_catalogue


def _synthetic_catalogue() -> pd.DataFrame:
    rows = []
    examples = {
        "Electrical": ("wire harness", "Copper"),
        "Mechanical": ("gear assembly", "Steel"),
        "Polymer": ("plastic housing", "Polypropylene"),
    }
    for category, (name, material) in examples.items():
        for component_number in range(4):
            component = f"{category} component {component_number}"
            for variant in range(2):
                rows.append(
                    {
                        "part_id": f"{category}-{component_number}-{variant}",
                        "part_name": f"{name} variant {variant}",
                        "component": component,
                        "material": material,
                        "part_category": category,
                        "part_description": f"{category} category target leakage",
                    }
                )
    return pd.DataFrame(rows)


def _predictor(threshold: float = 0.0) -> PartCategoryPredictor:
    catalogue = _synthetic_catalogue()
    text = build_text(catalogue)
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True
    )
    vectors = vectorizer.fit_transform(text)
    estimator = LogisticRegression(
        class_weight="balanced", max_iter=3000, random_state=42
    ).fit(vectors, catalogue[TARGET])
    return PartCategoryPredictor(
        vectorizer,
        estimator,
        threshold=threshold,
        auto_accept_enabled=True,
    )


def test_feature_allowlist_excludes_category_and_category_derived_description():
    catalogue = _synthetic_catalogue()
    original = build_text(catalogue)
    changed = catalogue.copy()
    changed[TARGET] = "Fabricated"
    changed["part_description"] = "completely different category description"

    assert TEXT_FIELDS == ("part_name", "component", "material")
    assert TARGET not in TEXT_FIELDS
    assert build_text(changed).equals(original)
    assert "part_description" not in build_text(catalogue).iloc[0]
    with pytest.raises(ValueError, match="reviewed"):
        build_text(catalogue, (*TEXT_FIELDS, TARGET))


def test_group_split_keeps_components_disjoint_and_is_repeatable():
    catalogue = load_catalogue(ROOT / "data" / "03_spare_parts_catalogue.csv")

    first = _group_split(catalogue, 0.25, 42)
    second = _group_split(catalogue, 0.25, 42)
    train_index, holdout_index = first
    train_groups = set(catalogue.iloc[train_index][GROUP_COLUMN])
    holdout_groups = set(catalogue.iloc[holdout_index][GROUP_COLUMN])

    assert np.array_equal(train_index, second[0])
    assert np.array_equal(holdout_index, second[1])
    assert train_groups.isdisjoint(holdout_groups)
    assert len(holdout_index) > 0


def test_grouped_cv_folds_keep_components_disjoint():
    catalogue = _synthetic_catalogue()
    config = {
        "grouped_cv_folds": 3,
        "seed": 42,
        "model": {
            "analyzer": "char_wb",
            "ngram_range": [2, 5],
            "min_df": 1,
            "sublinear_tf": True,
            "class_weight": "balanced",
            "max_iter": 3000,
        },
    }

    probabilities, seen, folds = _oof_grouped_predictions(
        build_text(catalogue),
        catalogue[TARGET],
        catalogue[GROUP_COLUMN],
        ("Electrical", "Mechanical", "Polymer"),
        config,
    )

    assert probabilities.shape == (len(catalogue), 3)
    assert np.all(seen == 1)
    assert folds["component_overlap"].tolist() == [0, 0, 0]


def test_predictions_are_deterministic_for_fixed_model_and_input():
    first = _predictor()
    second = _predictor()
    part = {
        "part_name": "wire harness repair",
        "component": "Electrical component new",
        "material": "Copper",
        "part_category": "not an input",
        "part_description": "category leakage must be ignored",
    }

    assert first.classify_part(part) == second.classify_part(part)


def test_predictor_is_exported_from_classification_package():
    from src.classification import PartCategoryPredictor as PublicPredictor

    assert PublicPredictor is PartCategoryPredictor


def test_missing_and_unknown_catalogue_categories_use_prediction_contract():
    predictor = _predictor()
    part = {
        "part_name": "wire harness repair",
        "component": "Electrical component new",
        "material": "Copper",
    }

    missing = predictor.classify_part(part, None)
    unknown = predictor.classify_part(part, "Unlisted")

    assert missing.catalogue_status == "missing"
    assert missing.resolved_category == missing.predicted_category
    assert unknown.catalogue_status == "unknown"
    assert unknown.resolved_category == unknown.predicted_category
    assert missing.confidence > 0
    assert len(missing.alternatives) == 2


def test_existing_catalogue_category_is_preserved_and_conflicts_are_flagged():
    predictor = _predictor()
    part = {
        "part_name": "wire harness repair",
        "component": "Electrical component new",
        "material": "Copper",
    }
    prediction = predictor.classify_part(part, "Electrical")
    conflict = predictor.classify_part(part, "Mechanical")

    assert prediction.catalogue_status == "known"
    assert prediction.resolved_category == "Electrical"
    assert conflict.catalogue_status == "conflicting"
    assert conflict.predicted_category == "Electrical"
    assert conflict.resolved_category == "Mechanical"
    assert conflict.requires_clarification


def test_low_confidence_requires_clarification_without_fabricated_resolution():
    predictor = _predictor(threshold=1.0)
    result = predictor.classify_part(
        {
            "part_name": "unseen fastener shape",
            "component": "novel assembly",
            "material": "Unlisted ceramic",
        },
        None,
    )

    assert result.confidence < 1.0
    assert result.catalogue_status == "missing"
    assert result.resolved_category is None
    assert result.requires_clarification


@pytest.mark.parametrize(
    "part_information",
    [{"part_name": None, "component": " ", "material": None}],
)
def test_empty_or_invalid_part_information_is_rejected(part_information):
    with pytest.raises(ValueError, match="populated"):
        build_text(part_information)


def test_empty_part_information_is_rejected():
    with pytest.raises(ValueError, match="at least one record"):
        build_text({})


def test_empty_training_catalogue_is_rejected():
    with pytest.raises(ValueError, match="at least one record"):
        build_text(pd.DataFrame(columns=TEXT_FIELDS))
