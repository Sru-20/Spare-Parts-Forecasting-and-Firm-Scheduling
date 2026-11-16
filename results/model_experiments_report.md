# Structural forecasting experiments — measured result

**Status:** closed. No candidate is admitted. The frozen champion
`lightgbm_tweedie::random_1` remains in place. The final-test window stays
sealed.

**Question:** can a different model form or a blend improve on the champion's
held lead-time WAPE of 34.90% without breaking the bias screen?

**Answer:** four candidates were tested. Three were worse on tuning evidence
and stopped there. One blend improved held lead-time WAPE by 0.76 percentage
points, but it missed the tuning advancement rule, its confidence interval
includes zero, and it worsens bias. It is recorded as exploratory, not adopted.

## Rules used

- Advancement was decided on the two tuning origins only: a candidate needed
  to improve pooled tuning lead-time WAPE by at least 0.5 percentage points.
- Candidates that advanced were scored once on the four held origins against
  the same gates as before: 0.5 pp lead-time WAPE gain, |lead-time bias| ≤ 10%,
  and no P95 coverage loss where quantiles exist.
- Tuning bias was reported but not used to reject candidates, because the
  frozen champion itself has −16.93% tuning lead-time bias yet −5.06% on held
  origins. Rejecting on tuning bias would reject the champion.
- The champion emits no quantiles, so the P95 rule is not applicable.

## Candidates and tuning result

| Candidate | Change | Tuning weekly WAPE | Tuning weekly bias | Tuning lead WAPE | Tuning lead bias | Advanced |
|---|---|---:|---:|---:|---:|---|
| Frozen champion | none | 63.11 | −18.02 | **39.29** | −16.93 | baseline |
| Blend 75% Tweedie + 25% quantile P50 | ensemble | 61.64 | −21.68 | 38.90 | −20.23 | no: gain 0.40 pp |
| Tweedie + trend features | features | 63.59 | −17.74 | 40.03 | −17.35 | no: −0.74 pp |
| Poisson loss | objective | 65.73 | −11.93 | 41.28 | −12.01 | no: −1.99 pp |
| Poisson + trend features | objective + features | 67.16 | −10.43 | 42.31 | −10.29 | no: −3.01 pp |

The trend features were origin-safe and computed only from prior observations:
4/13-week trend ratio, 13/52-week trend ratio, 4-week growth against the prior
four weeks, 13-week coefficient of variation, 13-week zero rate, and observed
history length. All ratios were capped at 10. They did not improve tuning WAPE.

Poisson moved tuning bias toward zero (+5 pp) but increased WAPE by 2 pp. That
is not an accuracy improvement.

## Held result for the blend (exploratory)

The blend was measured on held origins because it was already fitted for
tuning. It is **not admissible** under the stated rule.

| Candidate | Weekly WAPE | Weekly bias | Lead WAPE | Lead bias | Paired lead WAPE delta vs champion (95% CI) |
|---|---:|---:|---:|---:|---:|
| Frozen champion | 63.25 | −15.11 | **34.90** | **−5.06** | — |
| Blend 75/25 | **61.25** | −20.03 | **34.15** | −9.91 | −0.76 pp (−2.17, +0.46) |

Per-origin lead-time WAPE:

| Origin | Champion | Blend |
|---|---:|---:|
| 2025-01-13 | 36.15 | 35.86 |
| 2025-04-21 | 38.32 | 37.24 |
| 2025-07-28 | 33.80 | 31.55 |
| 2025-11-03 | 31.31 | 31.87 |

The blend improves three of four origins and worsens one by 0.56 pp. The
average gain is 0.76 pp, but the paired bootstrap interval includes zero, so
the gain is not distinguishable from noise at this sample size. It also moves
lead-time bias to −9.91%, just inside the 10% screen, and worsens weekly bias
from −15.11% to −20.03%. This is an accuracy-for-bias trade, not a uniform
improvement.

## Why the other candidates were stopped

- `tweedie_augmented`: tuning lead WAPE 40.03% versus 39.29%, a 0.74 pp loss.
- `poisson`: 41.28%, a 1.99 pp loss.
- `poisson_augmented`: 42.31%, a 3.01 pp loss.

Scoring them on held origins after they failed tuning would add comparisons
without a selection basis, so they were not evaluated there.

## Where the error sits

The champion's held lead-time absolute error decomposes as follows:

| Segment | Share of actual demand | WAPE | Share of absolute error |
|---|---:|---:|---:|
| Smooth | 64.5% | 19.77% | 36.5% |
| Erratic | 23.7% | 51.31% | 34.9% |
| Lumpy | 7.1% | 102.18% | 20.8% |
| Intermittent | 4.7% | 57.87% | 7.8% |
| ABC A | 73.4% | 31.43% | 66.1% |

Two individual Lumpy series (P0032 West and P0098 West) contribute 6.9% of all
absolute error through single one-off spikes of 2,617 and 1,576 units; the
model forecasts near zero for them in those origins. No history-based model
can see those spikes in advance. This is a data-generating property, not a
tuning defect.

## Decision

- Keep the frozen champion. No Phase 4 or Phase 5 artifact changes.
- Do not adopt the blend. It fails the pre-declared tuning advancement rule,
  its held gain is within noise, and it worsens both weekly and lead-time bias.
- The blend is the only candidate worth revisiting, and only with a new
  independent period (real data or the future window) and full Track A
  scoring, not lead-time WAPE alone. Even then, its bias needs to improve, not
  just its WAPE.
- Further variants scored on the same four held origins would be exploratory.
  The final test remains sealed until Phase 7.

## Limitations

- Synthetic data; method evidence only, not factory performance.
- Two tuning origins and four held origins; the blend weight was selected on
  the same tuning sample, and the held interval reflects only four origins.
- The blend needs both models at inference; that adds runtime and reduces
  explainability relative to a single model.
- No direct multi-horizon or hurdle model was implemented. The failed SBA/TSB
  router (lead WAPE 39.26%) and the results above give no evidence that those
  forms would clear the gates, but they remain untested.

## Outputs and reproduction

| File | SHA-256 |
|---|---|
| `results/model_experiments_tuning.csv` | `5f6d7c42acb99b1324efef83afa05c8b4b6368d3ca5cf9cf0e3660b2a3902f31` |
| `results/model_experiments_held.csv` | `b8ad7e68523d56b421f79c716ae701a02e54c0ce69ddbe5942373fc769cafbde` |
| `results/model_experiments_summary.json` | runtime field changes each run; metrics are deterministic |

```bash
python -m src.forecasting.model_experiments
python -m pytest -q tests/test_model_experiments.py
```

Five new tests cover ratio capping, origin safety of the trend features,
training/recursive feature agreement, blend weight selection, and blend
application without input mutation. The affected test set (experiments, Phase
4 v4.1, bias calibration) passes 22 tests. The full suite later unblocked after
the Phase 0 contract test and config were consolidated on `docs/SPEC.md` (SPEC
v4.2); see `reports/project_report_current.md` section 1.
