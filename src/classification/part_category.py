"""Catalogue part-category classification from text fields only."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.data.loader import ROOT, load_catalogue


TEXT_FIELDS = (
    "part_name",
    "part_description",
    "component",
    "material",
    "vehicle_system",
    "vehicle_model",
    "vehicle_type",
)
TARGET = "part_category"
MATERIAL_GROUP_STATUS = "not_available_in_catalogue"


def _config(root: Path) -> dict[str, Any]:
    config = yaml.safe_load(
        (root / "configs" / "project.yaml").read_text(encoding="utf-8")
    )
    if "classification" not in config:
        raise ValueError("Missing classification configuration")
    return config["classification"]


def build_text(catalogue: pd.DataFrame) -> pd.Series:
    missing = [field for field in TEXT_FIELDS if field not in catalogue]
    if missing:
        raise ValueError(f"Catalogue is missing text fields: {missing}")
    if TARGET in TEXT_FIELDS:
        raise ValueError("The target column cannot be a classifier input")
    return (
        catalogue.loc[:, list(TEXT_FIELDS)]
        .fillna("")
        .astype(str)
        .agg(" | ".join, axis=1)
    )


def _logistic_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def _svm_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                CalibratedClassifierCV(
                    LinearSVC(random_state=42, max_iter=10000),
                    method="sigmoid",
                    cv=3,
                ),
            ),
        ]
    )


MODELS = {
    "tfidf_logistic_regression": _logistic_pipeline,
    "tfidf_linear_svm": _svm_pipeline,
}


def _group_split(catalogue: pd.DataFrame, fraction: float, seed: int):
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=fraction, random_state=seed
    )
    train_index, test_index = next(
        splitter.split(
            catalogue,
            y=catalogue[TARGET],
            groups=catalogue["component"],
        )
    )
    train_components = set(catalogue.iloc[train_index]["component"])
    test_components = set(catalogue.iloc[test_index]["component"])
    if train_components & test_components:
        raise RuntimeError("Component groups overlap the hold-out split")
    return train_index, test_index


def _metric_rows(
    model_name: str,
    evaluation: str,
    truth: pd.Series,
    prediction: np.ndarray,
    labels: list[str],
    train_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=labels, zero_division=0
    )
    summary = [
        {
            "model": model_name,
            "evaluation": evaluation,
            "macro_f1": float(f1_score(truth, prediction, labels=labels, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(truth, prediction, labels=labels, average="weighted", zero_division=0)),
            "accuracy": float(accuracy_score(truth, prediction)),
            "n_train": int(train_size),
            "n_test": int(len(truth)),
            "labels_present_in_test": int(np.isin(labels, truth.unique()).sum()),
            "labels_total": len(labels),
        }
    ]
    per_class = [
        {
            "model": model_name,
            "evaluation": evaluation,
            "label": label,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    ]
    return summary, per_class


def run_part_category(root: Path = ROOT) -> dict[str, Any]:
    config = _config(root)
    seed = int(config["seed"])
    threshold = float(config["confidence_threshold"])
    catalogue = load_catalogue(root / "data" / "03_spare_parts_catalogue.csv")
    if catalogue["part_id"].duplicated().any():
        raise ValueError("Catalogue part_id must be unique")
    texts = build_text(catalogue)
    labels = sorted(catalogue[TARGET].dropna().astype(str).unique().tolist())
    truth = catalogue[TARGET].astype(str)

    summary_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    holdout_predictions: dict[str, np.ndarray] = {}
    train_index, test_index = _group_split(
        catalogue, float(config["group_holdout_test_fraction"]), seed
    )
    folds = StratifiedKFold(
        n_splits=int(config["cv_folds"]), shuffle=True, random_state=seed
    )
    for model_name, factory in MODELS.items():
        cv_scores = cross_val_score(
            factory(),
            texts,
            truth,
            cv=folds,
            scoring="f1_macro",
        )
        summary_rows.append(
            {
                "model": model_name,
                "evaluation": "stratified_5fold_cv",
                "macro_f1": float(np.mean(cv_scores)),
                "weighted_f1": np.nan,
                "accuracy": np.nan,
                "n_train": int(len(catalogue)),
                "n_test": int(len(catalogue)),
                "labels_present_in_test": len(labels),
                "labels_total": len(labels),
            }
        )
        holdout_model = factory().fit(texts.iloc[train_index], truth.iloc[train_index])
        holdout_prediction = holdout_model.predict(texts.iloc[test_index])
        holdout_predictions[model_name] = holdout_prediction
        summary, per_class = _metric_rows(
            model_name,
            "component_group_holdout",
            truth.iloc[test_index],
            holdout_prediction,
            labels,
            len(train_index),
        )
        summary_rows.extend(summary)
        per_class_rows.extend(per_class)

    metrics = pd.DataFrame(summary_rows)
    per_class_frame = pd.DataFrame(per_class_rows)
    holdout_scores = metrics.loc[
        metrics["evaluation"].eq("component_group_holdout")
    ]
    best_model = str(
        holdout_scores.sort_values("macro_f1", ascending=False).iloc[0]["model"]
    )
    final_model = MODELS[best_model]().fit(texts, truth)
    probability = final_model.predict_proba(texts)
    predicted = final_model.classes_[probability.argmax(axis=1)]
    confidence = probability.max(axis=1)
    predictions = pd.DataFrame(
        {
            "part_id": catalogue["part_id"].astype(str),
            "component": catalogue["component"],
            "material": catalogue["material"],
            "catalogue_category": catalogue[TARGET],
            "predicted_category": predicted,
            "confidence": confidence,
        }
    )
    predictions["needs_review"] = predictions["confidence"].lt(threshold) | predictions[
        "confidence"
    ].isna()
    predictions["prediction_model"] = best_model

    confusion = confusion_matrix(
        truth.iloc[test_index],
        holdout_predictions[best_model],
        labels=labels,
    )
    confusion_frame = pd.DataFrame(confusion, index=labels, columns=labels)
    component_overlap = set(catalogue.iloc[train_index]["component"]) & set(
        catalogue.iloc[test_index]["component"]
    )
    material_mapping = catalogue.groupby("material")[TARGET].nunique()
    catalogue_by_category = catalogue[TARGET].value_counts().sort_index()

    results = root / "results"
    results.mkdir(exist_ok=True)
    predictions.to_csv(results / "part_category_predictions.csv", index=False)
    metrics.to_csv(results / "part_category_metrics.csv", index=False)
    per_class_frame.to_csv(results / "part_category_per_class.csv", index=False)
    confusion_frame.to_csv(results / "part_category_confusion_matrix.csv")

    holdout = metrics.loc[
        metrics["evaluation"].eq("component_group_holdout")
    ].sort_values("macro_f1", ascending=False)
    cv = metrics.loc[metrics["evaluation"].eq("stratified_5fold_cv")]
    table_rows = []
    for row in metrics.itertuples():
        weighted = "" if pd.isna(row.weighted_f1) else f"{row.weighted_f1:.4f}"
        accuracy = "" if pd.isna(row.accuracy) else f"{row.accuracy:.4f}"
        table_rows.append(
            f"| {row.model} | {row.evaluation} | {row.macro_f1:.4f} | "
            f"{weighted} | {accuracy} | {row.n_test} |"
        )
    table = "\n".join(table_rows)
    report = f"""# Part-category classification — measured result

**Basis:** synthetic catalogue (160 parts), supplied `part_category` label.
**Target:** `part_category` (8 classes). Material group is **not available** in
the catalogue; it is not inferred or invented (`{MATERIAL_GROUP_STATUS}`).

## Method

Inputs are D3 text and specification fields only: {", ".join(TEXT_FIELDS)}.
Demand, schedule, outcome, lifecycle and ABC/type fields are not inputs.
Two models use character 2–5-gram TF-IDF with seed 42:

1. Logistic regression, balanced class weights.
2. Linear SVM with sigmoid calibration for confidence.

Evaluation uses (a) stratified 5-fold cross-validation and (b) a group hold-out
by `component` so the test components never appear in training.

## Results

| Model | Evaluation | Macro-F1 | Weighted-F1 | Accuracy | Test rows |
|---|---|---:|---:|---:|---:|
{table}

Best component hold-out model: **{best_model}**. The hold-out has
**{int(len(test_index))}** parts and **{len(set(catalogue.iloc[test_index]['component']))}**
unseen components; train/test component overlap is **{len(component_overlap)}**.

Per-class precision, recall and support are in
`results/part_category_per_class.csv`; the hold-out confusion matrix is in
`results/part_category_confusion_matrix.csv`.

## Optimism warning

The generator assigns one category and one material per component, and
`component`, `material` and the component name inside `part_name` are inputs.
Training and CV rows therefore share component/material with their test rows,
so stratified CV is optimistic. Only the component group hold-out is a
non-memorisation estimate. Material alone is also partly determinative:
{int(material_mapping.eq(1).sum())} of {len(material_mapping)} materials map to
exactly one category. This is a data-generating property, not a rule to trust in
production.

## Use in Ask AI

`results/part_category_predictions.csv` contains `predicted_category`,
`confidence` and `needs_review` (confidence below {threshold:.2f};
config status `provisional_editable`). The catalogue category is never
overwritten. `src/classification/part_category.py:resolve_category` returns the
catalogue category when present and falls back to the prediction only when it
is missing. Ask AI category requests should use that lookup. Ask AI itself is
Phase 8 and does not exist yet, so this is the integration contract, not a
tested assistant path.

## Reproduction

```bash
python -m src.classification.part_category
python -m pytest -q tests/test_part_category.py
```
"""
    (results / "part_category_report.md").write_text(report, encoding="utf-8")
    summary = {
        "target": TARGET,
        "material_group_status": MATERIAL_GROUP_STATUS,
        "confidence_threshold": threshold,
        "models": list(MODELS),
        "best_component_holdout_model": best_model,
        "holdout_macro_f1": float(
            holdout.iloc[0]["macro_f1"]
        ),
        "cv_macro_f1": {
            row.model: float(row.macro_f1) for row in cv.itertuples()
        },
        "holdout_parts": int(len(test_index)),
        "holdout_components": int(
            catalogue.iloc[test_index]["component"].nunique()
        ),
        "component_overlap": len(component_overlap),
        "needs_review_parts": int(predictions["needs_review"].sum()),
        "catalogue_categories": {
            str(key): int(value) for key, value in catalogue_by_category.items()
        },
    }
    (results / "part_category_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary


def load_predictions(root: Path = ROOT) -> pd.DataFrame:
    path = root / "results" / "part_category_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing classification output: {path}")
    return pd.read_csv(path)


def resolve_category(
    part_id: str,
    catalogue_category: str | None,
    predictions: pd.DataFrame | None = None,
    root: Path = ROOT,
) -> str:
    """Use the catalogue label; fall back to the prediction only when absent."""
    if catalogue_category is not None and pd.notna(catalogue_category) and str(
        catalogue_category
    ).strip():
        return str(catalogue_category)
    frame = predictions if predictions is not None else load_predictions(root)
    match = frame.loc[frame["part_id"].astype(str).eq(str(part_id))]
    if match.empty:
        raise KeyError(f"No category or trained prediction for part {part_id}")
    return str(match.iloc[0]["predicted_category"])


if __name__ == "__main__":
    print(run_part_category())
