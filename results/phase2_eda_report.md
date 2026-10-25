# Phase 2 EDA, Segmentation and Feature Engineering

## Scope and leakage boundary

Demand-label EDA, association summaries, and event diagnostics use only observed `train_val` rows dated through 2026-03-23. SBC/ABC/XYZ segments use observed `warmup` plus `train_val` labels through that same cutoff, matching the generator's verified `demand_class_trainval` calculation (which includes warm-up despite the README's train-validation-only description). `final_test` and `future` labels/rows are not used. Dataset `out_*` fields are read only as historical synthetic labels. Generator tags and lifecycle fields are used only for clearly labelled artifact diagnostics, never as model predictors. Descriptive associations are not causal estimates and do not validate real factory effects.

## What is supported by the supplied data

- Weekly regional demand, calendar month, festival/holiday, promotion/campaign, plant shutdown, market, vehicle parc/age, lifecycle, criticality, and the generator's synthetic outcome labels are present for historical train-validation dates.
- Per-part region series are segmented from observed warm-up and train-validation demand. The supplied `demand_class_trainval` is cross-checked only; recomputed SBC labels are used. Agreement for comparable classified series: 100.000% (569 series).
- Computed SBC series counts: {'Smooth': 119, 'Erratic': 97, 'Intermittent': 133, 'Lumpy': 220, 'Unclassified': 47}. Series with fewer than 26 observed development weeks, or no observed development weeks, are left unclassified; they receive the same-component/vehicle-type analogue group.
- Part ABC uses annualized development-window usage value (average observed weekly `demand_qty × unit_cost × 52`) and cumulative-value cutoffs 80%/95%. Part XYZ uses CV of weekly part demand aggregated across regions: X ≤ 0.50, Y ≤ 1.00, Z > 1.00; zero-mean series are Unclassified.
- Current computed part counts by ABC: {'A': 66, 'B': 47, 'C': 46, 'Unclassified': 1}; XYZ: {'Z': 70, 'Y': 59, 'X': 28, 'Unclassified': 3}. Criticality remains the supplied part attribute and is attached to every part-region segment.
- Spearman correlation across parts, using train-validation mean demand, is 0.1537 for mean vehicle parc and -0.0063 for mean vehicle age. These are synthetic descriptive associations, not causal or independently validated fleet effects.

## Event and known-pattern diagnostics

| comparison                          | event_rows | baseline_rows | event_mean_demand  | baseline_mean_demand | relative_change_pct | interpretation_limit                                                  |
| ----------------------------------- | ---------- | ------------- | ------------------ | -------------------- | ------------------- | --------------------------------------------------------------------- |
| festival                            | 14315      | 52030         | 21.360530911631155 | 19.036344416682685   | 12.209205948762136  | descriptive synthetic association; not causal                         |
| public_holiday                      | 4000       | 62345         | 20.618             | 19.468521934397305   | 5.90429036922302    | descriptive synthetic association; not causal                         |
| plant_shutdown                      | 2275       | 64070         | 14.27076923076923  | 19.724847822693928   | -27.650801876654498 | descriptive synthetic association; not causal                         |
| promotion                           | 7996       | 58349         | 22.100550275137568 | 19.186635589298874   | 15.187210244738768  | descriptive synthetic association; not causal                         |
| campaign                            | 11465      | 54880         | 25.876580898386393 | 18.21359329446064    | 42.072903902253735  | descriptive synthetic association; not causal                         |
| market_event                        | 2872       | 63473         | 21.965877437325904 | 19.427961495439007   | 13.063212743564012  | descriptive synthetic association; not causal                         |
| post_promotion_week_promo_sensitive | 1282       | 9253          | 11.033541341653667 | 20.142656435750567   | -45.22300781504379  | generator tag used for descriptive audit only; excluded from features |
| generator_spike_label_prevalence    | 665        | 66345         | 246.87819548872181 | 19.537825005652273   |                     | out_spike_flag is a historical label only, never a feature            |

Substitution comparisons use the generator-documented 2025-04-07 change and train-validation rows only. The pairs are in `results/phase2_substitution_comparison.csv`; these effects are injected generator behavior, not discovered causal facts.

Generator tag diagnostics (distinct parts in train-validation rows): {'step_change_tagged_parts': 12, 'promo_sensitive_parts': 27}. Tags are answer keys and are excluded from features. The synthetic `out_spike_flag` is used only to describe labelled historical spikes; see event table and never use it as a feature.

Regional and criticality summaries are in `results/phase2_region_summary.csv` and `results/phase2_criticality_summary.csv`. Lifecycle summaries are in `results/phase2_lifecycle_summary.csv`. The provided data cannot establish causal festival, promotion, vehicle-use, lifecycle, substitution, or shutdown effects in a real supply chain; real external covariates and operational outcomes are unavailable.

## Figures

- `results/phase2_charts/weekly_demand_by_region.png`
- `results/phase2_charts/month_of_year_seasonality.png`
- `results/phase2_charts/event_demand_differences.png`
- `results/phase2_charts/sbc_segment_counts.png`
- `results/phase2_charts/vehicle_parc_vs_demand.png`

## Origin-safe feature construction

Implemented in `src/features/time_features.py`. One-step features use only demand before the target week: lags 1/4/13/52; same-period-last-year; rolling mean/median/sample standard deviation for windows 4/13/52 using a one-row shift; zero-run length and elapsed weeks since last observed positive demand; calendar sin/cos. Target-date calendar fields are passed through only from the explicitly approved calendar set. Conditional future covariates are not passed by default and require explicit `origin_known_columns`.

For the 14-week future horizon the selected strategy is **recursive/direct one-step iteration**: pass the future calendar to `build_recursive_forecast_features`, which processes it chronologically and appends each model prediction as the prior value before constructing the next week's features. No actual future target or `out_*` value is accepted from the future calendar. Missing prior targets are never filled with zero; missing observations reset the consecutive-zero run.

`reports/feature_dictionary_phase2.csv` lists every generated lag, rolling, intermittent, static, calendar, and conditional feature with its formula and origin-time safeguard. `tests/test_leakage.py` verifies truncated-history invariance, future-label rejection, prior-only lag/rolling values, and recursive feedback.

## Feature and segmentation parameters

- SBC: ADI = observed weeks / positive-demand weeks; CV² = sample variance / mean² over positive demand. Smooth: ADI < 1.32 and CV² < 0.49; Erratic: ADI < 1.32 and CV² ≥ 0.49; Intermittent: ADI ≥ 1.32 and CV² < 0.49; Lumpy: ADI ≥ 1.32 and CV² ≥ 0.49. Minimum 26 observed weeks. Sparse (<3 positive weeks) cases follow the generator's CV²=1 convention and are flagged.
- ABC: annualized usage value with 80%/95% cumulative-value boundaries; explicitly a project segmentation convention.
- XYZ: CV of aggregated weekly part demand with 0.50 and 1.00 boundaries; explicitly a project segmentation convention.

## Artifacts

- `artifacts/series_segments.parquet`
- `reports/feature_dictionary_phase2.csv`
- `results/phase2_*` summary CSVs and `results/phase2_charts/*.png`
- Re-run: `python -m src.features.phase2_eda`
