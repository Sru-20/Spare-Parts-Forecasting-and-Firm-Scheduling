# Phase 8A — Part-category classification

**Basis:** method validation on the synthetic D3 catalogue; not evidence of
production classification performance.

## Data and features

- D3 catalogue: 160 rows, 8 known
  categories, 160 labeled training/evaluation rows,
  0
  missing/unknown category labels.
- Class counts: Body: 10, Electrical: 28, Electronics: 11, Ferrous: 32, Mechanical: 40, Non-ferrous: 16, Polymer: 11, Rubber: 12.
- Explicit feature allowlist: `part_name, component, material`.
  `part_description` is excluded because the generator places the category
  name directly in its text. IDs, target/category-derived fields, outcomes,
  and unrelated catalogue/scheduling attributes are not inputs.
- No source catalogue data is modified. Missing/unknown labels are not
  invented or used as training targets.

## Evaluation protocol and results

The fixed baseline is character 2–5-gram TF-IDF plus balanced logistic
regression (seed 42). The component-group holdout was split
before grouped cross-validation. Grouped CV uses
5 `StratifiedGroupKFold` folds on development
components only; every row receives exactly one out-of-fold prediction. The
heldout components are never used for threshold tuning.

| Evaluation | Rows | Accuracy | Macro-F1 | Weighted-F1 | Labels present / total |
|---|---:|---:|---:|---:|---:|
| Grouped CV OOF | 119 | 0.2773 | 0.2106 | 0.2648 | 8 / 8 |
| Held-out components | 41 | 0.2439 | 0.0694 | 0.2168 | 5 / 8 |

| Fold | Validation rows | Train components | Validation components | Overlap | Accuracy | Macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 24 | 25 | 6 | 0 | 0.1667 | 0.0417 |
| 2 | 23 | 25 | 6 | 0 | 0.2609 | 0.1458 |
| 3 | 23 | 25 | 6 | 0 | 0.1739 | 0.0833 |
| 4 | 27 | 24 | 7 | 0 | 0.2593 | 0.0774 |
| 5 | 22 | 25 | 6 | 0 | 0.5455 | 0.2631 |

### Held-out per-class metrics

| Category | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Body | 3 | 0.0000 | 0.0000 | 0.0000 |
| Electrical | 3 | 0.0000 | 0.0000 | 0.0000 |
| Electronics | 7 | 0.0000 | 0.0000 | 0.0000 |
| Ferrous | 16 | 0.5000 | 0.6250 | 0.5556 |
| Mechanical | 12 | 0.0000 | 0.0000 | 0.0000 |
| Non-ferrous | 0 | 0.0000 | 0.0000 | 0.0000 |
| Polymer | 0 | 0.0000 | 0.0000 | 0.0000 |
| Rubber | 0 | 0.0000 | 0.0000 | 0.0000 |

### Held-out confusion matrix

| actual / predicted | Body | Electrical | Electronics | Ferrous | Mechanical | Non-ferrous | Polymer | Rubber |
| ------------------ | ---- | ---------- | ----------- | ------- | ---------- | ----------- | ------- | ------ |
| Body               | 0    | 0          | 0           | 2       | 1          | 0           | 0       | 0      |
| Electrical         | 1    | 0          | 0           | 0       | 0          | 0           | 2       | 0      |
| Electronics        | 0    | 7          | 0           | 0       | 0          | 0           | 0       | 0      |
| Ferrous            | 0    | 0          | 0           | 10      | 6          | 0           | 0       | 0      |
| Mechanical         | 0    | 0          | 0           | 8       | 0          | 0           | 4       | 0      |
| Non-ferrous        | 0    | 0          | 0           | 0       | 0          | 0           | 0       | 0      |
| Polymer            | 0    | 0          | 0           | 0       | 0          | 0           | 0       | 0      |
| Rubber             | 0    | 0          | 0           | 0       | 0          | 0           | 0       | 0      |

Full grouped-CV and held-out per-class/confusion rows are saved in versioned
CSV outputs alongside this report.

## Confidence and integration contract

The clarification threshold is 1.00;
automatic fill is disabled because no threshold met the configured precision target.
Threshold candidates were chosen only using development-set grouped-CV OOF
predictions, targeting minimum selective accuracy
0.80. The held-out
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
