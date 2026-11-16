# Part-category classification — measured result

**Basis:** synthetic catalogue (160 parts), supplied `part_category` label.
**Target:** `part_category` (8 classes). Material group is **not available** in
the catalogue; it is not inferred or invented (`not_available_in_catalogue`).

## Method

Inputs are D3 text and specification fields only: part_name, part_description, component, material, vehicle_system, vehicle_model, vehicle_type.
Demand, schedule, outcome, lifecycle and ABC/type fields are not inputs.
Two models use character 2–5-gram TF-IDF with seed 42:

1. Logistic regression, balanced class weights.
2. Linear SVM with sigmoid calibration for confidence.

Evaluation uses (a) stratified 5-fold cross-validation and (b) a group hold-out
by `component` so the test components never appear in training.

## Results

| Model | Evaluation | Macro-F1 | Weighted-F1 | Accuracy | Test rows |
|---|---|---:|---:|---:|---:|
| tfidf_logistic_regression | stratified_5fold_cv | 1.0000 |  |  | 160 |
| tfidf_logistic_regression | component_group_holdout | 0.1541 | 0.4351 | 0.4878 | 41 |
| tfidf_linear_svm | stratified_5fold_cv | 1.0000 |  |  | 160 |
| tfidf_linear_svm | component_group_holdout | 0.2103 | 0.4692 | 0.5610 | 41 |

Best component hold-out model: **tfidf_linear_svm**. The hold-out has
**41** parts and **11**
unseen components; train/test component overlap is **0**.

Per-class precision, recall and support are in
`results/part_category_per_class.csv`; the hold-out confusion matrix is in
`results/part_category_confusion_matrix.csv`.

## Optimism warning

The generator assigns one category and one material per component, and
`component`, `material` and the component name inside `part_name` are inputs.
Training and CV rows therefore share component/material with their test rows,
so stratified CV is optimistic. Only the component group hold-out is a
non-memorisation estimate. Material alone is also partly determinative:
8 of 13 materials map to
exactly one category. This is a data-generating property, not a rule to trust in
production.

## Use in Ask AI

`results/part_category_predictions.csv` contains `predicted_category`,
`confidence` and `needs_review` (confidence below 0.60;
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
