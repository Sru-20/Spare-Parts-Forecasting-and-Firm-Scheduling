# Spare Parts Demand Forecasting & Firm Scheduling

Datasets follow `docs/Spare_Parts_Forecasting_Problem_Statement.pdf`. Synthetic, seed **42**.

Inventory snapshot date: **2026-09-26**. History: **2023-01-02 to 2026-09-21**. Forecast horizon: **14 weeks** (2026-09-28 to 2026-12-28).

| # | File | Grain | Rows × cols | Purpose |
|---|---|---|---|---|
| 1 | `data/01_demand_forecasting.csv` | part × region × week | 120,861 × 62 | Train and backtest demand (history + 14 future weeks with blank target) |
| 2 | `data/02_scheduling_inputs.csv` | one row per part | 158 × 61 | Supplier, lead time, capacity, inventory for the firm schedule |
| 3 | `data/03_spare_parts_catalogue.csv` | one row per part | 160 × 29 | Master data for 160 spare parts |

Join key: `part_id`. Dataset 1 also carries part attributes so it can be used alone.

Regenerate: `python data/generate_datasets.py`  
Validate: `python data/validate_datasets.py data`

## Application architecture

- Frontend: React + TypeScript.
- Backend: one Django application exposing APIs through Django REST Framework, with MySQL persistence.
- ML/scheduling: imported Python packages under `src/`, called directly by Django in-process. There is no separate FastAPI service or second application backend.
- Batch computation: Django management commands using the same application code; no task queue or separately deployed worker is part of the MVP.
- API security: Django authentication and groups/permissions, with JWT authentication for REST requests. Planner/approver/viewer/admin permissions are enforced server-side.

Forecast and schedule read APIs should serve persisted outputs. Schedule proposals use the validated scheduling package; the LLM may interpret requests but cannot calculate quantities or bypass confirmation and audit controls. This is the selected project architecture, not a framework requirement imposed by the updated problem statement.

## What was wrong with the dummy files

The Downloads dummy set (`spare_parts_3_datasets`) was already structured for the PDF, but it was **not** fit to train a spare-parts model.

| Issue | Dummy files | This release |
|---|---|---|
| Syntetos–Boylan–Croston classes | **No lumpy series**; 251 Smooth / 50 Erratic / 33 Intermittent / 12 too-short | 119 Smooth, 97 Erratic, 133 Intermittent, **220 Lumpy**, 47 short (new SKUs) |
| History length | 156 weeks; test window shorter than one seasonal cycle | **195** history weeks (~3.7 years) plus 14 future |
| Regional grain | 3 regions | **4** (North, West, South, East) with different scale and festival lift |
| Master data | Implausible costs (fog bulb ~₹5,800, bumper ~₹58, ABS sensor as Copper) | Cost bands by component (fog bulb ~₹80–280, bumper ~₹4.5k–16k, electronics as FR4) |
| `part_type` vs demand | Fast Moving SKUs labelled Intermittent | Volume class is consistent with SBC class (lumpy/intermittent SKUs are never Fast Moving) |
| Duplicate column | `supply_mode` appeared twice in D2 | Single `supply_mode` |
| Festival encoding | Literal `"None"`; one flagged week; weak market index | Blank when no festival; High festivals also flag the prior week; regional festivals; cyclical market index + 12 event weeks |
| Shutdown artefact | Demand forced to ~10% of normal | Make/dependent parts drop more than independent aftermarket demand |
| Future drivers | Promo / campaign / parc blank | Planned Diwali promo and festival campaign filled; unknown market events left blank |
| Pattern coverage | Mostly smooth consumable-like series | Lifecycle, promo, project, batching, bullwhip, panic, substitution, MTO/MTS, dependent demand, dead stock, multi-modal |

Other dummy impurities that are gone: SET UoM on non-kits, Make parts with import suppliers, service level not tied to criticality, MOQ larger than half a year of demand, supplier capacity ~20× too large, open POs double-counted by supplier, answer-key columns mixed into features.

## PDF factors → columns

| Key factor | Where |
|---|---|
| Historical consumption, part-wise demand, weekly/monthly trends | D1 `demand_qty`, `demand_lag52`, `week_of_year`, `month` |
| Seasonality, vehicle/service seasonality | D1 calendar flags, `vehicle_system`, `vehicle_parc`, `avg_vehicle_age_years`, `monsoon_flag`, `summer_peak_flag` |
| Festivals, holidays | D1 `festival_*`, `weeks_to_next_festival`, `weeks_since_last_festival`, `public_holiday_flag` |
| Market and regional patterns | D1 `region`, `market_condition_index`, `market_event_flag` |
| Previous-year same period | D1 `demand_lag52` |
| Promotions, campaigns | D1 `promotion_*`, `campaign_*` |
| Spikes, unusual consumption | D1 `out_spike_flag`, `out_unusual_demand_flag`, `demand_qty_capped` |
| Working days, shutdowns, plant holidays | D1 `plant_working_days`, `plant_shutdown_flag`, `weeks_since_last_shutdown` |
| Part criticality, service level | D3 (also D1/D2) `criticality`, `service_level_target` |
| Lead time, supplier terms, MOQ | D2 `sup1_*` … `sup3_*` |
| Supplier/production capacity | D2 `sup*_monthly_capacity`, `sup*_capacity_committed_pct` |
| Existing inventory, open orders | D2 `on_hand_qty` … `emergency_po_open_flag` |

## Demand-pattern coverage (how to use it)

SBC class is computed on **train_val only** (`demand_class_trainval`, `adi_trainval`, `cv2_trainval`) so there is no test leakage.

| Pattern | How it appears |
|---|---|
| Smooth / Erratic / Intermittent / Lumpy | Assigned from component economics, then retuned until ADI/CV² land in the right bucket |
| Level, trend, seasonality, cycle | Lifecycle curve, system-month seasonality, 78-week market sine |
| Step change, spikes | `step_change` tag; `out_spike_flag` from train-val 98.5th percentile |
| Introduction / growth / maturity / decline / fad | `lifecycle_pattern`; New / Phase-out / Obsolete in catalogue |
| Promo + pull-forward dip | `promo_sensitive` parts; post-promo week ×0.72 |
| Project-based | Rare large orders on `project_based` / MTO parts |
| Bullwhip, batching, panic | Tags plus dealer first-week-of-month pulse, market-event multipliers |
| Substitution | Five `substitution_group` pairs; demand shifts from 2025-04-07 |
| Dependent vs independent | Make parts tagged `dependent_demand` (harder shutdown drop) |
| MTO vs MTS | `fulfilment_mode` |
| Multi-modal / dead stock | Mixture sizes; rare tiny orders on `dead_stock` |

**Outcomes, never features:** `out_consumption_qty`, `out_backorder_qty`, `out_lost_sales_qty`, `out_stockout_flag`, `out_spike_flag`, `out_unusual_demand_flag`.

**Split (`split_role`):** warmup (to 2024-01-01, lags only) → train_val (to 2026-03-23) → final_test (2026-03-30 to 2026-09-21, open once) → future.

`is_missing_week=1` rows (72) have a blank target; exclude them from training and scoring.

## Suggested modelling workflow

1. Classify series with ADI/CV² (`demand_class_trainval`).
2. Smooth / seasonal: exponential smoothing, Holt–Winters, SARIMA, or gradient boosting with calendar features.
3. Erratic: ML with promo, festival, market, and parc drivers plus safety stock.
4. Intermittent / lumpy: Croston / SBA / TSB or ML on monthly aggregates; do not treat zeros as missing.
5. New SKUs: analogous-part forecast from same `component` + `vehicle_type`.
6. Convert the SKU-week forecast into a firm schedule with D2 (net requirement, MOQ, capacity, lead time, criticality).

## Known limits (state in the report)

- Data are synthetic; festival dates are calendar-accurate enough for modelling, not a legal holiday calendar.
- Dataset 2 is a single snapshot; a scheduling replay needs a short simulator warm-up.
- 47 series are too short to classify (new launches). Use analogues, not SBC class.
- Obsolete parts (P0046, P0087) have history only and no D2 row, by design.
