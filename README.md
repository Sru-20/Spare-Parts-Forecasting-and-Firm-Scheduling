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

## Quick-start demo (5 minutes)

```powershell
# 1. Backend — from the repository root
$env:DJANGO_SECRET_KEY = "demo-secret-key-change-in-production"
$env:DB_NAME = Join-Path $PWD "backend_phase9_demo.sqlite3"
py -3.10 -B backend/manage.py runserver 127.0.0.1:8000

# 2. Frontend — new terminal, from the repository root
Set-Location .\frontend
npm ci
npm run dev
```

Open the Vite URL (usually http://localhost:5173). The demo database is pre-populated with users, catalogue, Phase 6 results, KPIs, and sample conversations. Default admin credentials are in the demo DB.

---

## Application architecture

- Frontend: React + TypeScript.
- Backend: one Django application exposing APIs through Django REST Framework. SQLite is the development default; MySQL and PostgreSQL are selectable through environment configuration.
- ML/scheduling: imported Python packages under `src/`, called directly by Django in-process. There is no separate FastAPI service or second application backend.
- Batch computation: Django management commands using the same application code; no task queue or separately deployed worker is part of the MVP.
- API security: SimpleJWT authentication with refresh rotation and token revocation. Viewer, planner, approver, and admin permissions are enforced server-side.

Forecast and schedule read APIs should serve persisted outputs. Schedule proposals use the validated scheduling package; the LLM may interpret requests but cannot calculate quantities or bypass confirmation and audit controls. This is the selected project architecture, not a framework requirement imposed by the updated problem statement.

### Django API development

The backend runs from the repository root and reuses the existing Python packages:

```powershell
py -3.10 -m pip install -r requirements.txt
$env:DJANGO_SECRET_KEY = "<local-development-secret>"
py -3.10 -B backend/manage.py migrate
py -3.10 -B backend/manage.py check
py -3.10 -B backend/manage.py runserver
```

Do not commit the development secret. SQLite is selected by default. To use a server database, configure `DB_ENGINE` (`mysql` or `postgres`), `DB_NAME`, `DB_USER`, and `DB_HOST`; provide `DB_PASSWORD` and `DB_PORT` through the environment as needed. Django endpoints are versioned under `/api/v1/`. The rule-based Ask AI fallback uses no external model key and cannot directly access the database or apply proposals.

The Phase 8B test/evaluation commands are:

```powershell
$env:DJANGO_SECRET_KEY = "<local-development-secret>"
py -3.10 -B backend/manage.py test backend.tests.test_phase8b_api
py -3.10 -B backend/manage.py evaluate_assistant
```

Assistant evaluation fixtures are generated multilingual paraphrases, not human-validated requests. The current report limits claims to paraphrase robustness; proposal writes remain explicit, role-checked, constraint-checked, and audited. See `reports/assistant_evaluation_phase8b_v1.md` and the Phase 8B entry in `STATE.md`. The Phase 9 website is implemented, but its overall gate remains partial; inspect the sequential [01 executive summary](reports/01_phase9_executive_summary_v1.md), [02 evidence report](reports/02_phase9_closeout_v1.md), and [03 validation record](results/03_phase9_ui_validation_v1.json). Phase 7 remains unevaluated and unfinalized.

### Phase 9 web application

The React application lives in `frontend/`. Start Django and Vite in separate PowerShell terminals. Use a separate `DB_NAME` when an existing local database file must be preserved:

```powershell
# Backend terminal, from the repository root
$env:DJANGO_SECRET_KEY = "<local-development-secret>"
$env:DB_NAME = Join-Path $PWD "backend_phase9_demo.sqlite3"
py -3.10 -B backend/manage.py migrate
py -3.10 -B backend/manage.py import_catalogue
py -3.10 -B backend/manage.py register_phase6_snapshot
py -3.10 -B backend/manage.py createsuperuser
py -3.10 -B backend/manage.py shell -c "from backend.accounts.models import User; User.objects.filter(is_superuser=True).update(role='admin')"
py -3.10 -B backend/manage.py runserver 127.0.0.1:8000
```

`register_phase6_snapshot` registers the saved `lightgbm_quantile::random_1` + `risk_adjusted_mrp` Phase 6 validation champion and its provisional future schedule for the UI. It reads saved Phase 6 result files and the imported catalogue only, creates `results/phase9_ui_forecasts_phase6_v1.csv` without overwriting an existing file, and is idempotent. Forecast rows are a distinct `(part_id, week)` view of saved schedule `forecast_qty` values; the command does not recompute forecasts or read Phase 7 final-test targets. The KPI records remain Phase 6 validation/simulation evidence; the schedule remains provisional and Phase 5 remains PARTIAL. The future forecast does not replace the Track A point-forecast benchmark.

```powershell
# Frontend terminal, from the repository root
Set-Location .\frontend
npm ci
npm run dev
```

Open the Vite URL printed in the terminal. The development proxy forwards `/api` to `http://127.0.0.1:8000`. Focused frontend checks are `npm run test` and `npm run build`. Model retraining/promotion, final-test evaluation, and unsupported operational integrations are not exposed as UI actions.

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
