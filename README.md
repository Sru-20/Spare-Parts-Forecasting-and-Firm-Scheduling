<p align="center">
  <img src="frontend/public/sparecast-mark.svg" alt="SpareCast logo" width="96" />
</p>

# SpareCast

*Intelligent Spare Parts Demand Forecasting & Firm Scheduling Platform*

A full-stack planning application that connects spare-parts demand forecasts to a constraint-checked, simulated supply schedule. Planners can review saved results, KPIs, warnings and proposed changes in a web interface.

> **Scope:** The supplied data is synthetic. Results demonstrate a method on simulated data; they are not proof of factory savings or a production forecast guarantee. The application does not connect to an ERP system or issue real purchase orders.

## What the system does

1. Validates demand, catalogue and supplier input data.
2. Builds time-safe features and forecasts demand for a 14-week horizon.
3. Calculates simulated orders using inventory, lead time, supplier capacity, minimum order quantity (MOQ), order multiples and other planning rules.
4. Checks each schedule independently. Infeasible orders remain unresolved; the system does not replace them with arbitrary quantities.
5. Simulates inventory, receipts, service, shortages and emergency supply.
6. Saves versioned results for the web application and evaluation reports.
7. Provides a controlled assistant for questions and proposals. Only an explicit, authorized confirmation can apply a proposal to a simulated schedule.

Forecasting predicts demand; the scheduling engine calculates quantities; the assistant interprets requests and calls approved services.

## Data

The source files are in `data/`. They are synthetic, use seed 42, and must not be modified by analysis or application runs.

| File | Grain | Size | Purpose |
|---|---|---:|---|
| `01_demand_forecasting.csv` | Part x region x week | 120,861 rows x 62 columns | Demand history and 14 prediction-only future weeks |
| `02_scheduling_inputs.csv` | One row per part | 158 rows x 61 columns | Inventory, supplier terms, lead times and capacities |
| `03_spare_parts_catalogue.csv` | One row per part | 160 rows x 29 columns | Part master data |

Demand history runs from 2023-01-02 through 2026-09-21. The inventory/supplier input is a single snapshot dated 2026-09-26. The forecast period is 2026-09-28 through 2026-12-28. Demand is represented across North, West, South and East regions and includes smooth, erratic, intermittent, lumpy and short-history patterns. The supplied updated problem statement is `docs/Spare_Parts_Forecasting_Problem_Statement_Updated.pdf`; the original problem-statement PDF is not present.

The data generator recreates the synthetic source CSVs. Do not run it against the project data unless intentionally regenerating those inputs in a disposable or approved workspace. The integrity checker is:

```powershell
python data/generate_datasets.py
python data/validate_datasets.py data
```

The validator has recorded 32 checks passed and 0 failed. Note that a historical validator invocation read `demand_qty` values from final-test rows for integrity checks, not for prediction scoring. Do not claim that final-test labels have never been accessed.

## Project structure

| Path | Responsibility |
|---|---|
| `data/` | Source CSVs and dataset generation/validation scripts |
| `src/data/` | Schema-checked loading, data checks, joins and derived modelling tables |
| `src/features/` | Time-based features, demand segmentation and exploratory summaries |
| `src/forecasting/` | Forecast models, rolling-origin backtests, metrics and comparisons |
| `src/classification/` | Part-category prediction for human review |
| `src/scheduling/` | MRP planning, supplier-risk logic, schedule generation and independent checks |
| `src/simulation/` | Weekly inventory and service simulation |
| `src/evaluation/` | Guarded final-test protocol and evaluation runner |
| `backend/accounts/` | User identities, tokens, roles and permissions |
| `backend/planning/` | Planning APIs, result access, run/catalogue records and proposal services |
| `backend/assistant/` | Conversation API, intent parsing, controlled tools and assistant evaluation |
| `backend/spares_api/` | Django settings and API routing |
| `frontend/src/api/` | Typed API client, request hooks, shared types and formatting |
| `frontend/src/app/` | Application shell, routes, authentication and shared run context |
| `frontend/src/design-system/` | Shared components, design tokens, themes and CSS |
| `frontend/src/components/` | Reusable visual components, including forecast charts |
| `frontend/src/pages/` | Login and planning pages |
| `configs/` | Model, business, evaluation and assistant settings |
| `artifacts/` | Derived datasets and model artifacts |
| `results/` | Versioned forecast, schedule and KPI outputs |
| `reports/` | Method, evaluation and limitations documentation |
| `tests/` | Python data, forecasting, scheduling, simulation and workflow tests |

The `src/assistant/` directory is a placeholder. The implemented assistant is under `backend/assistant/`.

## Web application pages

- **Login:** Sign in with an assigned account.
- **Overview:** Summarizes the selected run, saved KPIs, forecast signal, schedule status, warnings and audit activity.
- **Forecasts:** Filter saved predictions by part, region and week; show prediction ranges when they are available.
- **Firm Schedule:** Inspect simulated order quantities, supplier details, dates, status and recorded constraint checks.
- **Inventory & Supply:** Review saved inventory projections and warnings.
- **KPI Analytics:** Review saved measures and the evidence behind them.
- **Ask AI:** Ask about saved planning data, view conversations and review proposal activity.
- **Audit Trail:** Inspect recorded actions, proposal details and validation information.
- **Run History:** Select and inspect registered planning runs.
- **Settings:** View account permissions, appearance preferences and active planning context. Admins can create accounts.

The frontend displays saved outputs. It does not invent chart data or calculate forecasts and schedules in the browser. There are no separate Catalogue or Proposals routes.

## Architecture

```text
Browser
  React + TypeScript + Vite
        | JSON over REST
        v
  Django + Django REST Framework + SimpleJWT
        | in-process imports
        v
  Python forecasting, scheduling and simulation packages
        |
        +-- Relational database: SQLite / MySQL / PostgreSQL
        +-- Versioned result files
```

There is one Django backend. The Python planning packages run in-process with it; there is no FastAPI service, separate ML server, Redis, Celery or microservice layer.

### Technology stack

- **Frontend:** React 19, TypeScript 7, Vite 8, React Router 7, Recharts and Lucide React.
- **Backend:** Python, Django 5.2, Django REST Framework 3.18 and SimpleJWT 5.5.
- **Database:** SQLite by default for development; MySQL and PostgreSQL are selectable through environment configuration.
- **Data science:** NumPy, Pandas, scikit-learn, LightGBM, XGBoost, SciPy and related Python libraries.
- **Frontend testing/build:** Vitest, Testing Library, TypeScript type checking and Vite production builds.

The shared CSS design system provides deep-navy surfaces, teal, electric-blue and violet accents, readable typography, responsive layouts, light/dark themes and reduced-motion support.

Exact dependency declarations are in `requirements.txt` and `frontend/package.json`.

### API and authorization

JSON endpoints are versioned under `/api/v1/`. They cover authentication, parts, runs, forecasts, schedules, KPIs, warnings, assistant conversations, proposals and audit records. List APIs are paginated.

Server-side roles are:

- **Viewer:** read planning information.
- **Planner:** read and create proposals.
- **Approver:** confirm eligible proposals.
- **Admin:** manage accounts and administrative functions.

SimpleJWT access tokens expire after 15 minutes; refresh tokens last 12 hours and rotate. Logout/revocation, password validation, request throttling, serializer validation and ORM-based queries are configured. Secrets such as `DJANGO_SECRET_KEY` must be supplied through the environment, not committed to source.

### Assistant safety

The assistant supports English, Hindi, Hinglish and Marathi. Its rule-based fallback needs no external model key. It interprets user intent and calls allowlisted backend services; it has no direct database access and does not calculate or invent order quantities.

Proposals are tied to the schedule version on which they were based and expire after 60 minutes by default. Confirmation checks the user's current role, schedule/proposal versions, validation results and duplicate-request controls. Applying or reverting a proposal is an explicit, audited action. A chat response such as "Looks good" is not confirmation.

## Forecasting, scheduling and evaluation

Forecast and scheduling candidates use time-respecting validation. Features and transformations use only information available at each forecast origin. Missing demand weeks remain missing; genuine zero-demand weeks remain zero; outcome columns are excluded from model inputs.

- **Track A - forecasting:** The selected point-forecast model is `lightgbm_tweedie::random_1`. A separate quantile LightGBM model provides uncertainty information for safety stock.
- **Track B - scheduling:** A deterministic MRP engine applies supplier and inventory rules and an independent checker determines whether orders are firm or unresolved.
- **Track C - assistant:** A controlled, rule-based workflow interprets requests and supports read-only answers and explicitly confirmed proposals.

Recorded validation results include:

- Track A held-validation weekly WAPE: **63.25%**; lead-time WAPE: **34.90%**. Weekly bias was **-15.11%** and lead-time bias was **-5.06%**.
- A separate Track B validation comparison selected `lightgbm_quantile::random_1` + `risk_adjusted_mrp`. Its WAPE was **46.35%**, compared with **57.46%** for the traditional baseline, but its simulated fill rate, stock-out rate and carrying-cost proxy were worse. Forecast accuracy alone therefore does not identify the best business outcome.
- The part-category classifier achieved **24.39%** accuracy on held-out components. Its confidence gate disables automatic category filling; predictions are for review.
- The assistant evaluation used 192 generated request paraphrases. The held-out fixtures reported zero incorrect auto-applications and zero invented quantities. They were not reviewed by fluent speakers, so they do not prove real-world multilingual accuracy.

All these findings are from synthetic data and validation/simulation outputs. They are not factory savings, operational service guarantees or final-test results.

## Local demonstration

The local database, `backend.sqlite3`, is populated with catalogue and Phase 6 saved results. Local demo-only admin, planner and approver accounts are provisioned separately for the current demo session. Their passwords are not stored in source control or intended for reuse outside localhost.

Open two PowerShell terminals at the repository root.

**Terminal 1 - backend:**

```powershell
$env:DJANGO_SECRET_KEY = "<set-a-local-only-secret>"
$env:DB_NAME = Join-Path $PWD "backend.sqlite3"
py -3.10 -B backend/manage.py runserver 127.0.0.1:8000
```

Replace the placeholder with a local secret in the terminal. Do not put it in a source file.

**Terminal 2 - frontend:**

```powershell
Set-Location .\frontend
npm ci
npm run dev
```

Open the Vite URL shown in the terminal, usually `http://localhost:5173`. The Vite development proxy forwards `/api` requests to Django at `http://127.0.0.1:8000`.

### Fresh local database

To initialize a new, empty SQLite database:

```powershell
py -3.10 -m pip install -r requirements.txt
$env:DJANGO_SECRET_KEY = "<set-a-local-only-secret>"
$env:DB_NAME = Join-Path $PWD "backend.sqlite3"
py -3.10 -B backend/manage.py migrate
py -3.10 -B backend/manage.py check
py -3.10 -B backend/manage.py import_catalogue
py -3.10 -B backend/manage.py register_phase6_snapshot
py -3.10 -B backend/manage.py createsuperuser
py -3.10 -B backend/manage.py shell -c "from backend.accounts.models import User; User.objects.filter(is_superuser=True).update(role='admin')"
py -3.10 -B backend/manage.py runserver 127.0.0.1:8000
```

The snapshot-registration command uses saved Phase 6 results and the imported catalogue. It does not retrain a model or recalculate the forecast. Use a separate `DB_NAME` when you need to preserve an existing database.

For MySQL or PostgreSQL, set `DB_ENGINE`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_HOST` and, if needed, `DB_PORT` in the environment. Do not put database credentials in source files.

### Free Render demo deployment

The root `render.yaml` Blueprint deploys the Vite frontend and Django API under one Render account. The backend installs only its runtime dependencies from `requirements-render.txt` and uses SQLite inside the free backend service because free Render web services cannot attach persistent disks. Its database and generated proposal schedule files can be lost when the service spins down or redeploys; Render's free web service can also sleep while idle. This is for a temporary demonstration, not production or durable data.

1. Push the repository to a Git provider and create a Render account.
2. In Render, choose **New → Blueprint**, connect the repository, and select the root `render.yaml`.
3. When prompted, provide `SPARECAST_ADMIN_USERNAME`, `SPARECAST_ADMIN_EMAIL`, and a strong `SPARECAST_ADMIN_PASSWORD`. These initialize the demo administrator; do not commit these values.
4. Deploy the Blueprint. It builds the frontend and starts the backend, applies migrations, and imports the catalogue and saved demo snapshot.
5. Open the `sparecast-web` URL and sign in with the administrator credentials you supplied.

The Blueprint uses `sparecast-api.onrender.com` and `sparecast-web.onrender.com`. If Render assigns different service URLs, update `VITE_API_BASE_URL` on the static site to the API URL plus `/api/v1`, and update `DJANGO_CORS_ALLOWED_ORIGINS` on the API to the static-site origin, then redeploy both services. The local `backend.sqlite3` file is not uploaded; this demo creates a fresh database and seed data at startup.

## Tests and checks

Run focused checks from the repository root:

```powershell
# Backend API and assistant lifecycle tests
$env:DJANGO_SECRET_KEY = "<set-a-local-only-secret>"
py -3.10 -B backend/manage.py test backend.tests.test_phase8b_api

# Assistant paraphrase-fixture evaluation
py -3.10 -B backend/manage.py evaluate_assistant

# Frontend tests and production build
Set-Location .\frontend
npm test
npm run build
```

Before running data-wide test or validation commands, review the final-test safeguards in `docs/SPEC.md` and `STATE.md`. A historical dataset integrity check read final-test target values for integrity checks; it was not prediction scoring.

## Limitations and document control

- Data is synthetic, and the supplier/scheduling dataset is a single snapshot. Costs and some planning conditions are proxies, not verified business inputs.
- Forty-seven short-history series have no eligible donor under the strict part-analogue rule. They are recorded as unforecasted rather than given an invented value.
- The category classifier is not accurate enough for automatic filling.
- Assistant evaluation fixtures are generated paraphrases, not human-validated multilingual tests.
- The final-test forecasting evaluation has not been run. Do not use the final-test period to tune models.
- Real ERP integration, purchase-order placement, offline retraining and automatic model promotion are not exposed as web features.
- The governing specification is `docs/SPEC.md` (v4.2). The root-level `SPEC.md` mentioned in its workflow instructions is absent. `docs/SPEC_AMENDMENT.md` is not accepted as a new governing specification because its referenced v4.3 baseline is not present.
- `STATE.md` is the execution record. It contains dated historical entries; use the relevant current status and exact run/initialization when quoting schedule totals.

Further data definitions and project decisions are in `docs/SPEC.md`, `docs/phase5_assumptions.md`, `reports/` and `results/`.
