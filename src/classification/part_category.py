"""Component-grouped part-category classification and safe resolution."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from src.data.loader import ROOT, load_catalogue


TARGET = "part_category"
TEXT_FIELDS = ("part_name", "component", "material")
GROUP_COLUMN = "component"
MATERIAL_GROUP_STATUS = "not_available_in_catalogue"
OUTPUTS = {
    "metrics": "results/part_category_phase8a_metrics_v1.csv",
    "per_class": "results/part_category_phase8a_per_class_v1.csv",
    "confusion": "results/part_category_phase8a_confusion_matrix_v1.csv",
    "folds": "results/part_category_phase8a_grouped_cv_folds_v1.csv",
    "thresholds": "results/part_category_phase8a_threshold_tuning_v1.csv",
    "holdout_predictions": "results/part_category_phase8a_holdout_predictions_v1.csv",
    "metadata": "results/part_category_phase8a_model_metadata_v1.json",
    "report": "reports/part_category_phase8a_v1.md",
    "model": "artifacts/models/part_category_phase8a_v1.joblib",
    "preprocessor": "artifacts/models/part_category_preprocessor_phase8a_v1.joblib",
    "class_mapping": "artifacts/models/part_category_class_mapping_phase8a_v1.json",
}


@dataclass(frozen=True)
class CategoryCandidate:
    category: str
    confidence: float


@dataclass(frozen=True)
class CategoryResolution:
    predicted_category: str
    confidence: float
    alternatives: tuple[CategoryCandidate, ...]
    catalogue_status: str
    catalogue_category: str | None
    resolved_category: str | None
    requires_clarification: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(root: Path) -> tuple[dict[str, Any], Path]:
    path = root / "configs" / "part_category.yaml"
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Part-category configuration must be a YAML mapping")
    if config.get("target_column") != TARGET:
        raise ValueError(f"Classification target must be {TARGET!r}")
    if tuple(config.get("feature_allowlist", ())) != TEXT_FIELDS:
        raise ValueError("Feature allowlist must exactly match the reviewed fields")
    if config.get("group_column") != GROUP_COLUMN:
        raise ValueError("Grouped evaluation must use component as its group")
    if config.get("model", {}).get("name") != "tfidf_char_logistic_regression":
        raise ValueError("Only the SPEC baseline TF-IDF/logistic model is configured")
    confidence = config.get("confidence", {})
    candidates = confidence.get("candidate_thresholds")
    if (
        not isinstance(candidates, list)
        or not candidates
        or any(not 0 <= float(value) <= 1 for value in candidates)
    ):
        raise ValueError("Confidence thresholds must be a non-empty list in [0, 1]")
    if not 0 <= float(confidence.get("minimum_auto_accept_precision", -1)) <= 1:
        raise ValueError("Minimum auto-accept precision must be in [0, 1]")
    return config, path


def build_text(
    part_information: pd.DataFrame | Mapping[str, Any],
    feature_allowlist: Sequence[str] = TEXT_FIELDS,
) -> pd.Series:
    """Build deterministic text from the explicit, non-target feature allowlist."""
    fields = tuple(feature_allowlist)
    if fields != TEXT_FIELDS or TARGET in fields:
        raise ValueError("Classifier inputs must use the reviewed name/component/material fields")
    frame = (
        pd.DataFrame([part_information])
        if isinstance(part_information, Mapping)
        else part_information
    )
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Part information must contain at least one record")
    selected = frame.reindex(columns=list(fields)).copy()
    if isinstance(part_information, Mapping):
        selected = selected.fillna("")
    text_columns = []
    for field in fields:
        values = selected[field].fillna("").astype(str).str.strip()
        text_columns.append(values.map(lambda value: f"{field} {value}" if value else ""))
    text = pd.concat(text_columns, axis=1).agg(" ".join, axis=1).str.strip()
    if text.eq("").any():
        raise ValueError("At least one allowed part-description field must be populated")
    return text


def _make_vectorizer(config: Mapping[str, Any]) -> TfidfVectorizer:
    model_config = config["model"]
    ngram_values = model_config["ngram_range"]
    if len(ngram_values) != 2:
        raise ValueError("Character n-gram range must have exactly two values")
    return TfidfVectorizer(
        analyzer=model_config["analyzer"],
        ngram_range=(int(ngram_values[0]), int(ngram_values[1])),
        min_df=int(model_config["min_df"]),
        sublinear_tf=bool(model_config["sublinear_tf"]),
        lowercase=True,
    )


def _make_classifier(config: Mapping[str, Any]) -> LogisticRegression:
    model_config = config["model"]
    return LogisticRegression(
        class_weight=model_config["class_weight"],
        max_iter=int(model_config["max_iter"]),
        random_state=int(config["seed"]),
        solver="lbfgs",
    )


def _valid_label_mask(labels: pd.Series, config: Mapping[str, Any]) -> pd.Series:
    unknown = {
        str(value).strip().casefold()
        for value in config["unknown_category_values"]
    }
    return labels.notna() & ~labels.astype("string").str.strip().str.casefold().isin(
        unknown
    )


def _group_split(
    catalogue: pd.DataFrame, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if GROUP_COLUMN not in catalogue or TARGET not in catalogue:
        raise ValueError("Grouped split requires component and part_category")
    groups = catalogue[GROUP_COLUMN]
    if groups.isna().any() or groups.astype("string").str.strip().eq("").any():
        raise ValueError("Component group values must be present")
    if groups.nunique() < 2:
        raise ValueError("At least two component groups are required")
    if not 0 < fraction < 1:
        raise ValueError("Held-out component fraction must be in (0, 1)")
    splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    train_index, holdout_index = next(
        splitter.split(catalogue, catalogue[TARGET], groups=groups)
    )
    if set(groups.iloc[train_index]) & set(groups.iloc[holdout_index]):
        raise RuntimeError("Component groups overlap the held-out split")
    if catalogue.iloc[train_index][TARGET].nunique() < 2:
        raise ValueError("Development split has fewer than two target classes")
    return train_index, holdout_index


def _probability_matrix(
    estimator: LogisticRegression,
    vectors: Any,
    classes: Sequence[str],
) -> np.ndarray:
    probabilities = estimator.predict_proba(vectors)
    expanded = np.zeros((probabilities.shape[0], len(classes)), dtype=float)
    class_positions = {label: index for index, label in enumerate(classes)}
    for source_index, label in enumerate(estimator.classes_.tolist()):
        expanded[:, class_positions[str(label)]] = probabilities[:, source_index]
    return expanded


def _metrics(
    truth: Iterable[str], prediction: Iterable[str], labels: Sequence[str]
) -> dict[str, Any]:
    truth_values = list(truth)
    prediction_values = list(prediction)
    return {
        "accuracy": float(accuracy_score(truth_values, prediction_values)),
        "macro_f1": float(
            f1_score(
                truth_values,
                prediction_values,
                labels=list(labels),
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                truth_values,
                prediction_values,
                labels=list(labels),
                average="weighted",
                zero_division=0,
            )
        ),
        "n_samples": int(len(truth_values)),
        "labels_present": int(pd.Series(truth_values).nunique()),
        "labels_total": int(len(labels)),
    }


def _per_class_rows(
    evaluation: str,
    truth: Iterable[str],
    prediction: Iterable[str],
    labels: Sequence[str],
) -> list[dict[str, Any]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        list(truth), list(prediction), labels=list(labels), zero_division=0
    )
    return [
        {
            "evaluation": evaluation,
            "category": label,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    ]


def _confusion_rows(
    evaluation: str,
    truth: Iterable[str],
    prediction: Iterable[str],
    labels: Sequence[str],
) -> list[dict[str, Any]]:
    matrix = confusion_matrix(list(truth), list(prediction), labels=list(labels))
    return [
        {
            "evaluation": evaluation,
            "actual_category": labels[row],
            "predicted_category": labels[column],
            "count": int(matrix[row, column]),
        }
        for row in range(len(labels))
        for column in range(len(labels))
    ]


def _threshold_table(
    truth: Iterable[str],
    probabilities: np.ndarray,
    labels: Sequence[str],
    candidates: Sequence[float],
    minimum_precision: float,
) -> tuple[pd.DataFrame, float, bool]:
    truth_values = np.asarray(list(truth), dtype=object)
    predicted_indices = probabilities.argmax(axis=1)
    predicted = np.asarray(labels, dtype=object)[predicted_indices]
    confidence = probabilities.max(axis=1)
    rows = []
    for threshold in sorted({float(value) for value in candidates}):
        accepted = confidence >= threshold
        count = int(accepted.sum())
        precision = (
            float(np.mean(predicted[accepted] == truth_values[accepted]))
            if count else np.nan
        )
        rows.append(
            {
                "threshold": threshold,
                "auto_accepted": count,
                "coverage_pct": 100.0 * count / len(truth) if len(truth) else np.nan,
                "selective_accuracy": precision,
                "minimum_precision": minimum_precision,
                "meets_minimum_precision": bool(
                    count > 0 and precision >= minimum_precision
                ),
                "selected": False,
            }
        )
    table = pd.DataFrame(rows)
    eligible = table.loc[table["meets_minimum_precision"]]
    if eligible.empty:
        return table, 1.0, False
    selected_index = eligible.sort_values(
        ["auto_accepted", "selective_accuracy", "threshold"],
        ascending=[False, False, True],
        kind="stable",
    ).index[0]
    threshold = float(table.loc[selected_index, "threshold"])
    table.loc[selected_index, "selected"] = True
    return table, threshold, True


def _oof_grouped_predictions(
    texts: pd.Series,
    labels: pd.Series,
    components: pd.Series,
    classes: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    folds = int(config["grouped_cv_folds"])
    if components.nunique() < folds:
        raise ValueError("Not enough distinct components for configured grouped CV")
    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=int(config["seed"])
    )
    oof = np.zeros((len(labels), len(classes)), dtype=float)
    seen = np.zeros(len(labels), dtype=np.int8)
    fold_rows = []
    for fold_number, (train_index, validation_index) in enumerate(
        splitter.split(
            texts.to_numpy(dtype=str),
            labels.to_numpy(dtype=str),
            groups=components.to_numpy(dtype=str),
        ),
        start=1,
    ):
        train_groups = set(components.iloc[train_index])
        validation_groups = set(components.iloc[validation_index])
        if train_groups & validation_groups:
            raise RuntimeError(f"Component leakage in grouped CV fold {fold_number}")
        vectorizer = _make_vectorizer(config)
        train_vectors = vectorizer.fit_transform(texts.iloc[train_index].tolist())
        validation_vectors = vectorizer.transform(
            texts.iloc[validation_index].tolist()
        )
        estimator = _make_classifier(config).fit(
            train_vectors, labels.iloc[train_index].tolist()
        )
        probabilities = _probability_matrix(estimator, validation_vectors, classes)
        oof[validation_index] = probabilities
        seen[validation_index] += 1
        predictions = np.asarray(classes, dtype=object)[probabilities.argmax(axis=1)]
        fold_metrics = _metrics(
            labels.iloc[validation_index].tolist(), predictions.tolist(), classes
        )
        fold_rows.append(
            {
                "fold": fold_number,
                **fold_metrics,
                "train_components": int(len(train_groups)),
                "validation_components": int(len(validation_groups)),
                "component_overlap": int(len(train_groups & validation_groups)),
                "train_labels": int(labels.iloc[train_index].nunique()),
                "validation_labels": int(labels.iloc[validation_index].nunique()),
            }
        )
    if not np.all(seen == 1):
        raise RuntimeError("Grouped CV must produce exactly one OOF prediction per row")
    return oof, seen, pd.DataFrame(fold_rows)


def _catalogue_records(
    catalogue: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if catalogue.empty:
        raise ValueError("Catalogue is empty")
    if "part_id" not in catalogue or TARGET not in catalogue:
        raise ValueError("Catalogue requires unique part_id and part_category")
    if catalogue["part_id"].isna().any() or catalogue["part_id"].duplicated().any():
        raise ValueError("Catalogue part_id values must be present and unique")
    texts = build_text(catalogue, config["feature_allowlist"])
    valid = _valid_label_mask(catalogue[TARGET], config)
    records = catalogue.loc[valid].copy().reset_index(drop=True)
    if records.empty or records[TARGET].nunique() < 2:
        raise ValueError("At least two known category labels are required for training")
    targets = records[TARGET].astype(str)
    return records, texts.loc[valid].reset_index(drop=True), targets


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _check_outputs(root: Path) -> None:
    existing = [relative for relative in OUTPUTS.values() if (root / relative).exists()]
    if existing:
        raise FileExistsError(
            "Phase 8A outputs already exist; refusing to overwrite: "
            + ", ".join(existing)
        )


def run_part_category(root: Path = ROOT) -> dict[str, Any]:
    """Evaluate grouped folds/held-out components, then save serving artifacts."""
    config, config_path = _load_config(root)
    _check_outputs(root)
    data_path = root / "data" / "03_spare_parts_catalogue.csv"
    catalogue = load_catalogue(data_path)
    records, texts, labels = _catalogue_records(catalogue, config)
    classes = sorted(labels.unique().tolist())

    train_index, holdout_index = _group_split(
        records,
        float(config["heldout_component_fraction"]),
        int(config["seed"]),
    )
    development = records.iloc[train_index].reset_index(drop=True)
    development_texts = texts.iloc[train_index].reset_index(drop=True)
    development_labels = labels.iloc[train_index].reset_index(drop=True)
    oof_probabilities, _, fold_metrics = _oof_grouped_predictions(
        development_texts,
        development_labels,
        development[GROUP_COLUMN].reset_index(drop=True),
        classes,
        config,
    )
    oof_predictions = np.asarray(classes, dtype=object)[
        oof_probabilities.argmax(axis=1)
    ]
    cv_summary = _metrics(development_labels.tolist(), oof_predictions.tolist(), classes)
    threshold_table, threshold, auto_accept_enabled = _threshold_table(
        development_labels.tolist(),
        oof_probabilities,
        classes,
        config["confidence"]["candidate_thresholds"],
        float(config["confidence"]["minimum_auto_accept_precision"]),
    )

    holdout = records.iloc[holdout_index].reset_index(drop=True)
    train_vectorizer = _make_vectorizer(config)
    train_matrix = train_vectorizer.fit_transform(development_texts.tolist())
    holdout_estimator = _make_classifier(config).fit(
        train_matrix, development_labels.tolist()
    )
    holdout_probabilities = _probability_matrix(
        holdout_estimator,
        train_vectorizer.transform(texts.iloc[holdout_index].tolist()),
        classes,
    )
    holdout_predictions = np.asarray(classes, dtype=object)[
        holdout_probabilities.argmax(axis=1)
    ]
    holdout_confidence = holdout_probabilities.max(axis=1)
    holdout_truth = holdout[TARGET].astype(str).tolist()
    holdout_summary = _metrics(holdout_truth, holdout_predictions.tolist(), classes)
    accepted = holdout_confidence >= threshold if auto_accept_enabled else np.zeros(
        len(holdout), dtype=bool
    )
    holdout_summary.update(
        {
            "auto_accepted": int(accepted.sum()),
            "auto_accept_coverage_pct": (
                float(100.0 * accepted.mean()) if len(accepted) else None
            ),
            "auto_accept_precision": (
                float(np.mean(holdout_predictions[accepted] == np.asarray(holdout_truth)[accepted]))
                if accepted.any() else None
            ),
            "confidence_threshold": threshold,
        }
    )

    metric_rows = [
        {"evaluation": "grouped_5fold_oof", **cv_summary},
        {"evaluation": "heldout_components", **holdout_summary},
    ]
    holdout_truth = holdout[TARGET].astype(str).tolist()
    per_class = [
        *_per_class_rows(
            "grouped_5fold_oof",
            development_labels.tolist(),
            oof_predictions.tolist(),
            classes,
        ),
        *_per_class_rows(
            "heldout_components", holdout_truth, holdout_predictions.tolist(), classes
        ),
    ]
    confusion = [
        *_confusion_rows(
            "grouped_5fold_oof",
            development_labels.tolist(),
            oof_predictions.tolist(),
            classes,
        ),
        *_confusion_rows(
            "heldout_components", holdout_truth, holdout_predictions.tolist(), classes
        ),
    ]
    holdout_prediction_frame = pd.DataFrame(
        {
            "part_id": holdout["part_id"].astype(str),
            "component": holdout[GROUP_COLUMN].astype(str),
            "catalogue_category": holdout[TARGET].astype(str),
            "predicted_category": holdout_predictions,
            "confidence": holdout_confidence,
            "auto_accepted_by_cv_threshold": accepted,
            "requires_clarification": ~accepted,
        }
    )

    final_vectorizer = _make_vectorizer(config)
    final_matrix = final_vectorizer.fit_transform(texts)
    final_estimator = _make_classifier(config).fit(final_matrix, labels)
    model_path = root / OUTPUTS["model"]
    preprocessor_path = root / OUTPUTS["preprocessor"]
    class_mapping_path = root / OUTPUTS["class_mapping"]
    output_dirs = {model_path.parent}
    output_dirs.update((root / relative).parent for relative in OUTPUTS.values())
    for output_dir in output_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_estimator, model_path, compress=3)
    joblib.dump(final_vectorizer, preprocessor_path, compress=3)
    class_mapping = {label: index for index, label in enumerate(final_estimator.classes_)}
    _write_json(
        class_mapping_path,
        {
            "target_column": TARGET,
            "class_to_index": class_mapping,
            "index_to_class": list(final_estimator.classes_),
        },
    )

    metrics_frame = pd.DataFrame(metric_rows)
    per_class_frame = pd.DataFrame(per_class)
    confusion_frame = pd.DataFrame(confusion)
    threshold_table.to_csv(root / OUTPUTS["thresholds"], index=False)
    metrics_frame.to_csv(root / OUTPUTS["metrics"], index=False)
    per_class_frame.to_csv(root / OUTPUTS["per_class"], index=False)
    confusion_frame.to_csv(root / OUTPUTS["confusion"], index=False)
    fold_metrics.to_csv(root / OUTPUTS["folds"], index=False)
    holdout_prediction_frame.to_csv(
        root / OUTPUTS["holdout_predictions"], index=False
    )

    class_counts = labels.value_counts().sort_index()
    group_counts = records.groupby(TARGET, observed=True)[GROUP_COLUMN].nunique()
    missing_label_count = int((~_valid_label_mask(catalogue[TARGET], config)).sum())
    metadata = {
        "protocol_id": config["protocol_id"],
        "target_column": TARGET,
        "feature_allowlist": list(TEXT_FIELDS),
        "excluded_columns": [
            TARGET,
            "part_id",
            "part_number",
            "part_description",
            "part_type",
            "lifecycle_pattern",
            "demand_driver_tags",
            "abc_class",
            "vehicle_system",
            "vehicle_model",
            "vehicle_type",
            "criticality",
            "unit_cost",
            "outcome_fields",
        ],
        "seed": int(config["seed"]),
        "model": config["model"],
        "grouped_cv": {
            "group": GROUP_COLUMN,
            "folds": int(config["grouped_cv_folds"]),
            "development_rows": int(len(development)),
            "fold_results": fold_metrics.to_dict(orient="records"),
            **cv_summary,
        },
        "heldout_components": {
            "rows": int(len(holdout)),
            "components": int(holdout[GROUP_COLUMN].nunique()),
            "train_components": int(development[GROUP_COLUMN].nunique()),
            "component_overlap": int(
                len(set(development[GROUP_COLUMN]) & set(holdout[GROUP_COLUMN]))
            ),
            "classes_present": int(holdout[TARGET].nunique()),
            "metrics": holdout_summary,
        },
        "confidence": {
            "threshold": threshold,
            "auto_accept_enabled": auto_accept_enabled,
            "threshold_tuned_on": "grouped cross-validation OOF predictions from development components only",
            "minimum_auto_accept_precision": float(
                config["confidence"]["minimum_auto_accept_precision"]
            ),
        },
        "catalogue": {
            "rows": int(len(catalogue)),
            "training_rows": int(len(records)),
            "missing_or_unknown_labels_not_used_for_training": missing_label_count,
            "class_counts": {str(key): int(value) for key, value in class_counts.items()},
            "components_by_class": {str(key): int(value) for key, value in group_counts.items()},
            "material_group_status": MATERIAL_GROUP_STATUS,
        },
        "files": {
            "catalogue_sha256": _sha256(data_path),
            "config_sha256": _sha256(config_path),
            "model_path": OUTPUTS["model"],
            "model_sha256": _sha256(model_path),
            "preprocessor_path": OUTPUTS["preprocessor"],
            "preprocessor_sha256": _sha256(preprocessor_path),
            "class_mapping_path": OUTPUTS["class_mapping"],
        },
        "runtime": {
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "limitations": [
            "Synthetic D3 labels are assigned one category and one material per component; grouped holdout, not row-wise CV, is the generalization estimate.",
            "Rare classes have fewer components than grouped CV folds; some validation folds contain classes absent from their training fold.",
            "The held-out component split does not contain every category; absent-class metrics are reported with zero support.",
            "No catalogue category is overwritten. Existing categories remain authoritative; disagreements are flagged.",
            "No category labels were missing or unknown in the supplied catalogue, so fallback behavior is validated with synthetic interface tests only.",
        ],
    }
    _write_json(root / OUTPUTS["metadata"], metadata)
    _write_report(root / OUTPUTS["report"], metadata, per_class_frame, confusion_frame)
    return metadata


def _write_report(
    path: Path,
    metadata: Mapping[str, Any],
    per_class: pd.DataFrame,
    confusion: pd.DataFrame,
) -> None:
    cv = metadata["grouped_cv"]
    holdout = metadata["heldout_components"]["metrics"]
    class_counts = metadata["catalogue"]["class_counts"]
    distribution = ", ".join(f"{key}: {value}" for key, value in class_counts.items())
    holdout_class = per_class.loc[
        per_class["evaluation"].eq("heldout_components")
    ]
    confusion_pivot = confusion.loc[
        confusion["evaluation"].eq("heldout_components")
    ].pivot(index="actual_category", columns="predicted_category", values="count")
    confusion_table = _markdown_table(confusion_pivot)
    fold_rows = "\n".join(
        f"| {row['fold']} | {row['n_samples']} | {row['train_components']} | "
        f"{row['validation_components']} | {row['component_overlap']} | "
        f"{row['accuracy']:.4f} | {row['macro_f1']:.4f} |"
        for row in metadata["grouped_cv"]["fold_results"]
    )
    per_class_rows = "\n".join(
        f"| {row.category} | {int(row.support)} | {row.precision:.4f} | "
        f"{row.recall:.4f} | {row.f1:.4f} |"
        for row in holdout_class.itertuples()
    )
    report = f"""# Phase 8A — Part-category classification

**Basis:** method validation on the synthetic D3 catalogue; not evidence of
production classification performance.

## Data and features

- D3 catalogue: {metadata['catalogue']['rows']} rows, {len(class_counts)} known
  categories, {metadata['grouped_cv']['development_rows'] +
  metadata['heldout_components']['rows']} labeled training/evaluation rows,
  {metadata['catalogue']['missing_or_unknown_labels_not_used_for_training']}
  missing/unknown category labels.
- Class counts: {distribution}.
- Explicit feature allowlist: `{", ".join(metadata['feature_allowlist'])}`.
  `part_description` is excluded because the generator places the category
  name directly in its text. IDs, target/category-derived fields, outcomes,
  and unrelated catalogue/scheduling attributes are not inputs.
- No source catalogue data is modified. Missing/unknown labels are not
  invented or used as training targets.

## Evaluation protocol and results

The fixed baseline is character 2–5-gram TF-IDF plus balanced logistic
regression (seed {metadata['seed']}). The component-group holdout was split
before grouped cross-validation. Grouped CV uses
{metadata['grouped_cv']['folds']} `StratifiedGroupKFold` folds on development
components only; every row receives exactly one out-of-fold prediction. The
heldout components are never used for threshold tuning.

| Evaluation | Rows | Accuracy | Macro-F1 | Weighted-F1 | Labels present / total |
|---|---:|---:|---:|---:|---:|
| Grouped CV OOF | {cv['n_samples']} | {cv['accuracy']:.4f} | {cv['macro_f1']:.4f} | {cv['weighted_f1']:.4f} | {cv['labels_present']} / {cv['labels_total']} |
| Held-out components | {holdout['n_samples']} | {holdout['accuracy']:.4f} | {holdout['macro_f1']:.4f} | {holdout['weighted_f1']:.4f} | {holdout['labels_present']} / {holdout['labels_total']} |

| Fold | Validation rows | Train components | Validation components | Overlap | Accuracy | Macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|
{fold_rows}

### Held-out per-class metrics

| Category | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
{per_class_rows}

### Held-out confusion matrix

{confusion_table}

Full grouped-CV and held-out per-class/confusion rows are saved in versioned
CSV outputs alongside this report.

## Confidence and integration contract

The clarification threshold is {metadata['confidence']['threshold']:.2f};
automatic fill is {'enabled' if metadata['confidence']['auto_accept_enabled'] else 'disabled because no threshold met the configured precision target'}.
Threshold candidates were chosen only using development-set grouped-CV OOF
predictions, targeting minimum selective accuracy
{metadata['confidence']['minimum_auto_accept_precision']:.2f}. The held-out
auto-accept precision and coverage are reported in model metadata and were
not used to tune the threshold.

`PartCategoryPredictor.classify_part()` returns predicted category,
probability, alternative candidates, catalogue status (`missing`, `unknown`,
`known`, or `conflicting`), resolved category, and clarification flag. A
known catalogue category is preserved even when the prediction conflicts;
conflicts require clarification and are never silently written back.

## Limitations

The generator assigns one category and one material per component, making
component/material association unusually deterministic. Component-group
evaluation prevents the same component from appearing in train and validation,
but the 160-row/42-component synthetic sample is small. Rare categories have
fewer groups than CV folds, so some folds cannot train every class. The
held-out split covers only a subset of categories; zero-support classes remain
visible in per-class and confusion outputs. The supplied catalogue has no
missing/unknown category rows, so real missing-label fill performance is not
measurable here.

The original problem-statement PDF is absent from the workspace; the updated
problem statement is present. `docs/SPEC.md` v4.2 Section 10 governs this phase.

## Reproduction

```powershell
py -3.10 -m src.classification.part_category
py -3.10 -m pytest -q tests/test_part_category.py
```
"""
    with path.open("x", encoding="utf-8", newline="") as stream:
        stream.write(report)


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = ["actual / predicted", *[str(column) for column in frame.columns]]
    integer_frame = frame.astype("int64")
    rows = [
        [str(index), *[str(value) for value in row]]
        for index, row in zip(
            integer_frame.index, integer_frame.itertuples(index=False, name=None)
        )
    ]
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in rows))
        for column in range(len(headers))
    ]
    format_row = lambda row: "| " + " | ".join(
        value.ljust(widths[index]) for index, value in enumerate(row)
    ) + " |"
    return "\n".join(
        [
            format_row(headers),
            "| " + " | ".join("-" * width for width in widths) + " |",
            *(format_row(row) for row in rows),
        ]
    )


class PartCategoryPredictor:
    """Read-only inference interface; never mutates catalogue records."""

    def __init__(
        self,
        vectorizer: TfidfVectorizer,
        estimator: LogisticRegression,
        *,
        threshold: float,
        auto_accept_enabled: bool,
        alternative_count: int = 3,
        feature_allowlist: Sequence[str] = TEXT_FIELDS,
        unknown_values: Sequence[str] = ("", "unknown", "unclassified", "n/a", "none"),
    ) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("Confidence threshold must be in [0, 1]")
        if alternative_count < 1:
            raise ValueError("At least one alternative candidate must be requested")
        if tuple(feature_allowlist) != TEXT_FIELDS:
            raise ValueError("Unsupported part-category feature allowlist")
        self.vectorizer = vectorizer
        self.estimator = estimator
        self.threshold = float(threshold)
        self.auto_accept_enabled = bool(auto_accept_enabled)
        self.alternative_count = int(alternative_count)
        self.feature_allowlist = tuple(feature_allowlist)
        self.unknown_values = {
            str(value).strip().casefold() for value in unknown_values
        }

    @classmethod
    def load(cls, root: Path = ROOT) -> PartCategoryPredictor:
        metadata_path = root / OUTPUTS["metadata"]
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing classification metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        config, _ = _load_config(root)
        vectorizer = joblib.load(root / OUTPUTS["preprocessor"])
        estimator = joblib.load(root / OUTPUTS["model"])
        return cls(
            vectorizer,
            estimator,
            threshold=float(metadata["confidence"]["threshold"]),
            auto_accept_enabled=bool(metadata["confidence"]["auto_accept_enabled"]),
            alternative_count=int(config["alternative_candidates"]),
            feature_allowlist=config["feature_allowlist"],
            unknown_values=config["unknown_category_values"],
        )

    def classify_part(
        self,
        part_information: Mapping[str, Any],
        catalogue_category: str | None = None,
    ) -> CategoryResolution:
        text = build_text(part_information, self.feature_allowlist)
        probabilities = self.estimator.predict_proba(
            self.vectorizer.transform(text)
        )[0]
        order = np.argsort(probabilities)[::-1]
        labels = [str(label) for label in self.estimator.classes_]
        predicted = labels[int(order[0])]
        confidence = float(probabilities[int(order[0])])
        alternatives = tuple(
            CategoryCandidate(labels[int(index)], float(probabilities[int(index)]))
            for index in order[1 : 1 + self.alternative_count]
        )
        normalized = (
            str(catalogue_category).strip()
            if catalogue_category is not None
            else ""
        )
        category_key = normalized.casefold()
        if category_key in self.unknown_values:
            status = "missing" if not normalized else "unknown"
            saved_category = None
            resolved = (
                predicted
                if self.auto_accept_enabled and confidence >= self.threshold
                else None
            )
        elif normalized not in labels:
            status = "unknown"
            saved_category = normalized
            resolved = (
                predicted
                if self.auto_accept_enabled and confidence >= self.threshold
                else None
            )
        elif normalized != predicted:
            status = "conflicting"
            saved_category = normalized
            resolved = normalized
        else:
            status = "known"
            saved_category = normalized
            resolved = normalized
        low_confidence = not self.auto_accept_enabled or confidence < self.threshold
        clarification = low_confidence or status in {"missing", "unknown", "conflicting"} and resolved is None
        if status == "conflicting":
            clarification = True
        return CategoryResolution(
            predicted_category=predicted,
            confidence=confidence,
            alternatives=alternatives,
            catalogue_status=status,
            catalogue_category=saved_category,
            resolved_category=resolved,
            requires_clarification=clarification,
        )


def load_predictions(root: Path = ROOT) -> pd.DataFrame:
    """Load saved held-out predictions for audit/reporting."""
    path = root / OUTPUTS["holdout_predictions"]
    if not path.is_file():
        raise FileNotFoundError(f"Missing classification output: {path}")
    return pd.read_csv(path)


if __name__ == "__main__":
    print(json.dumps(run_part_category(), indent=2, sort_keys=True))
