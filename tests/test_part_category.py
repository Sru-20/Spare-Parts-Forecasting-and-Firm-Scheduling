import pandas as pd

from src.classification.part_category import (
    TARGET,
    TEXT_FIELDS,
    _group_split,
    build_text,
    resolve_category,
)
from src.data.loader import ROOT, load_catalogue


def test_text_inputs_exclude_target_and_demand_columns():
    catalogue = load_catalogue(ROOT / "data" / "03_spare_parts_catalogue.csv")
    texts = build_text(catalogue)
    assert len(texts) == len(catalogue)
    assert TARGET not in TEXT_FIELDS
    assert "demand_qty" not in TEXT_FIELDS
    assert "on_hand_qty" not in TEXT_FIELDS
    assert texts.iloc[0]


def test_group_split_uses_disjoint_components():
    catalogue = load_catalogue(ROOT / "data" / "03_spare_parts_catalogue.csv")
    train_index, test_index = _group_split(catalogue, 0.25, 42)
    train_components = set(catalogue.iloc[train_index]["component"])
    test_components = set(catalogue.iloc[test_index]["component"])
    assert train_components.isdisjoint(test_components)
    assert len(test_index) > 0


def test_catalogue_category_is_never_overwritten():
    predictions = pd.DataFrame(
        {
            "part_id": ["P0001", "P9999"],
            "predicted_category": ["Mechanical", "Electrical"],
            "confidence": [0.95, 0.40],
        }
    )
    assert (
        resolve_category("P0001", "Body", predictions=predictions) == "Body"
    )
    assert (
        resolve_category("P0001", None, predictions=predictions) == "Mechanical"
    )
    assert (
        resolve_category("P9999", "", predictions=predictions) == "Electrical"
    )
