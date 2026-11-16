# Tuning-only bias calibration — measured result

**Status:** closed. No calibration variant is admitted. The frozen champion stays
`lightgbm_tweedie::random_1` **uncorrected**. The final-test window remains sealed.

**Question:** can a multiplicative correction, fitted only on the two tuning
origins, remove the champion's measured under-forecast without making held
accuracy worse?

**Answer:** no. Every tested correction reduces the aggregate bias by
over-shooting, but raises held lead-time WAPE by 4.0–7.8 percentage points.
The bias is not a stable model property that a constant or segment multiplier
can remove.

## Acceptance rules used

A candidate had to satisfy all applicable rules on the four held origins:

1. Improve held lead-time WAPE by at least 0.5 percentage points (0.005 ratio)
   over the frozen champion.
2. Keep |held lead-time bias| at or below 10%.
3. Not reduce P95 coverage by more than 1 percentage point.

Rule 3 is not applicable here: the champion is a point Tweedie model and emits
no quantiles, so P95 coverage is undefined on every row. The 0.5-point margin
is this experiment's convention. No outcome depends on it: all variants are at
least 4.0 points worse.

## Method

- Refitted the frozen champion at the two tuning origins (2024-07-01,
  2024-10-07) with the frozen parameters; 15,922 observed weekly tuning rows
  were produced and stored.
- Built lead-time windows with the same 14-week horizon and 4-week extension
  used in v4.1.
- Fitted factors as `Σactual / Σforecast` on tuning rows only, clamped to
  [0.5, 1.5], requiring at least 30 rows and a positive actual total.
- Applied each factor unchanged to the saved held artifact (four origins,
  32,032 weekly rows, 2,248 lead-time windows over 565 series; 48 windows use
  the over-14-week extension).
- Held labels were used only for scoring.

The four scalar variants (`lead_pooled`, `weekly_pooled`, `recent_origin`,
`lead_criticality`) were specified before this run. After the scalar variants
failed, SBC class, ABC class and tuning-only volume quartile were added as
exploratory transfer diagnostics. They are not confirmatory.

## Measured results on held validation

WAPE and bias are percentages for display; none of these are savings or
business outcomes. Gain is baseline minus candidate lead-time WAPE; negative
means worse.

| Variant | Factor(s) | Weekly WAPE | Weekly bias | Lead WAPE | Lead bias | High-criticality lead bias | WAPE gain (pp) | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Frozen champion (no correction) | 1.000 | 63.25 | −15.11 | **34.90** | **−5.06** | −7.04 | — | baseline |
| lead_pooled | 1.2038 | 68.22 | +2.19 | 41.83 | +14.29 | +11.90 | −6.93 | fail |
| weekly_pooled | 1.2198 | 68.78 | +3.56 | 42.73 | +15.81 | +13.39 | −7.83 | fail |
| recent_origin | 1.1611 | 66.87 | −1.43 | 39.64 | +10.24 | +7.93 | −4.74 | fail |
| lead_criticality | H 1.3114, M 1.1517, L 1.0680 | 67.95 | +0.91 | 42.12 | +14.16 | +21.90 | −7.22 | fail |
| lead_sbc | 1.0385–1.50 | 68.04 | −2.92 | 39.02 | +7.47 | +7.60 | −4.12 | fail |
| lead_abc | 1.0271–1.50 | 67.31 | −1.69 | 38.95 | +9.05 | +7.06 | −4.05 | fail |
| lead_volume_quartile | 0.50–1.2206 | 67.58 | +1.05 | 42.32 | +14.13 | +12.33 | −7.42 | fail |

`lead_sbc` and `lead_abc` land inside the ±10% bias screen, but both still fail
the primary accuracy gate. The corrections trade a modest bias reduction for a
substantially larger absolute error.

Per-origin lead-time WAPE (baseline → best variant, `lead_abc`; change is
candidate minus baseline, so negative means better):

| Origin | Baseline | lead_abc | Change |
|---|---:|---:|---:|
| 2025-01-13 | 36.15 | 34.85 | −1.30 |
| 2025-04-21 | 38.32 | 45.04 | +6.72 |
| 2025-07-28 | 33.80 | 42.92 | +9.12 |
| 2025-11-03 | 31.31 | 33.68 | +2.37 |

Every variant worsens at least three of the four origins. The one improvement
is confined to a single origin.

## Why the correction does not transfer

Bias measured on tuning does not predict bias measured on held origins, and the
sign can change:

| Segment | Tuning lead bias | Held lead bias | Tuning factor |
|---|---:|---:|---:|
| SBC Smooth | −3.71 | +0.27 | 1.039 |
| SBC Erratic | −23.89 | −1.52 | 1.314 |
| SBC Intermittent | −23.17 | −17.97 | 1.302 |
| SBC Lumpy | −77.78 | −56.71 | 1.500 (clamped; raw 4.50) |
| ABC A | −13.30 | −4.38 | 1.153 |
| ABC B | −2.63 | +5.45 | 1.027 |
| ABC C | −61.44 | −33.39 | 1.500 (clamped; raw 2.59) |
| Criticality High | −23.74 | −7.04 | 1.311 |
| Criticality Medium | −13.18 | −0.99 | 1.152 |
| Criticality Low | −6.37 | −11.79 | 1.068 |
| Volume Q1 (lowest) | +348.90 | −74.59 | 0.500 (clamped; raw 0.223) |
| Volume Q2 | +0.48 | −18.58 | 0.995 |
| Volume Q3 | −17.52 | −1.68 | 1.212 |
| Volume Q4 | −18.07 | −0.73 | 1.221 |

The lowest-volume quartile flips from large over-forecast on tuning to large
under-forecast on held evidence. ABC B also changes sign. A factor learned in
one period therefore has no reliable direction in another.

Weekly bias is also origin-specific. Aggregated over horizons 1–14, the held
weekly bias is −15.11%, while the lead-time aggregate is −5.06%. The two
aggregations use different series and window compositions, so a single
multiplier cannot satisfy both. Bias by horizon is negative overall (from
−6.96% at step 2 to −38.62% at step 6), but origin-level cells change sign:
2025-04-21 reaches +25.2% at step 4 and −60.4% at step 6. This is a
period-and-series effect, not a fixed offset.

## Decision

- Keep `lightgbm_tweedie::random_1` uncorrected. No Phase 4 or Phase 5 result
  changes.
- Record the bias as an open, measured weakness: weekly −15.11%, lead-time
  −5.06%, High criticality −7.04%. It is not treated as corrected.
- Do not adopt a constant or segment multiplier. The transfer evidence above
  rejects it under the stated gates.
- A structural change is the plausible next lever, for example segment routing
  or a hurdle/two-part form for intermittent and lumpy series, or re-admitting
  only covariates with documented origin availability. Those are proposals,
  not measured results, and require separate approval and validation.
- The four held origins have now been used for many candidate comparisons. Any
  further variant scored on them is exploratory. Confirmatory evidence needs a
  new independent period with real data; the final test stays sealed until the
  Phase 7 evaluation.

## Limitations

- Synthetic data: this validates a method, not factory performance.
- Two tuning origins and four held origins; inference remains limited.
- Only multiplicative corrections were tested. Additive, per-horizon, and
  model-form changes were not tested here.
- Segment factors were clamped; unclamped Lumpy/C factors would be larger and
  more damaging.
- The 4-week over-horizon extension is the frozen v4.1 project assumption and
  is unchanged.

## Outputs and reproduction

| File | SHA-256 |
|---|---|
| `results/bias_calibration_candidates.csv` | `efd33bd2bbb9ef0ed989e632816ddef1413143f31e4c67e530069014126f5c46` |
| `results/bias_calibration_segment_transfer.csv` | `607554cad19d097fdccbe7b765e71aedd20d444877aec7afdffa4a9b56cc898e` |
| `artifacts/bias_calibration_tuning_forecasts.csv.gz` | `d69b8697d103c6bda3fed3d699cda2ae67bc856a9920f97d38809d9f5ebed8dc` |
| `results/bias_calibration_summary.json` | runtime field changes each run; all metrics are deterministic |

```bash
python -m src.forecasting.bias_calibration
python -m pytest -q tests/test_bias_calibration.py
```

The test file has 10 passing tests covering factor fitting and clamping, thin
evidence fallback, group application, gate arithmetic, quantile non-scaling,
volume-quartile index alignment, and reproduction of the published factor from
tuning-only rows with no held origin present. The full suite later unblocked after
the Phase 0 contract test and config were consolidated on `docs/SPEC.md` (SPEC v4.2);
see `reports/project_report_current.md` section 1.
