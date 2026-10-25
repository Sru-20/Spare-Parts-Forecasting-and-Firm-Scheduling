# Phase 0 — Environment and Data Audit

Checked on 2026-10-17 in the repository workspace. This is an environment snapshot, not a deployment certification. `.venv` and `.git` were not inspected as project sources.

## Runtime/tool availability

| Component | Observed | Status / implication |
|---|---|---|
| OS / architecture | Windows 11, x64 | Available |
| Python | 3.10.9 | Available; compatible with project scripts |
| CPU | Intel Core Ultra 5 225U; 12 cores / 14 logical processors | Available |
| GPU | `nvidia-smi` not found | No NVIDIA GPU/driver available through this environment; CPU path is the working assumption |
| Java | 26.0.2.1 | Available. Framework compatibility must be pinned/verified when the backend is scaffolded. |
| Maven | 3.9.16 | Available |
| Node.js / npm | 22.20.0 / 10.9.3 | Available |
| MySQL client | `mysql` not found on PATH | MySQL client/server access not established |
| Pytest | 9.1.1 | Available after a user-level install; `python -m pytest` works |
| Outbound HTTPS | HEAD to `https://pypi.org/simple/` returned HTTP 200 | Basic package-index egress works; this does not establish LLM-provider/API access |
| LLM credentials | No recognized OpenAI/Azure OpenAI/Anthropic/Gemini API-key environment variable names were present | No authenticated LLM API access configured/demonstrated |

## Python packages relevant to planned Tier 1 work

| Package | Present at inspection |
|---|---|
| NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3, scikit-learn 1.7.2 | Yes |
| XGBoost 3.2.0 | Yes (Tier 2 candidate; not required for Tier 1) |
| LightGBM | No |
| OR-Tools / PuLP | No |
| Optuna | No; bounded random search can satisfy the tuning requirement |
| FastAPI 0.141.1, Pydantic 2.13.4, Uvicorn 0.52.4 | Yes at the Phase 0 inventory; not selected for the application architecture |
| Django / Django REST Framework / SimpleJWT / mysqlclient | Not inventoried in Phase 0; added to the project dependency manifest after the backend decision |
| pytest 9.1.1, PyYAML 6.0.3 | Yes |
| PyTorch, Transformers, Sentence Transformers | No |

No modeling/solver packages were installed during Phase 0. This environment inventory predates the selected Django + DRF architecture; `requirements.txt` now records the planned single-backend Python dependencies, and the Django packages were not installed or runtime-verified as part of this architecture-only change. `pypdf` was installed user-level solely to extract the supplied guide/problem-statement PDFs; it is not a project runtime dependency.

## Dataset dimensions and dates

| File | Rows × columns | Grain | Date facts |
|---|---:|---|---|
| `data/01_demand_forecasting.csv` | 120,861 × 62 | Part × region × week | 2023-01-02 to 2026-12-28 |
| `data/02_scheduling_inputs.csv` | 158 × 61 | One row per scheduled part | All 158 snapshots are as of 2026-09-26 |
| `data/03_spare_parts_catalogue.csv` | 160 × 29 | One row per catalogue part | Launch weeks present: 2025-11-24 to 2026-04-06 |

D1 has 30,157 warm-up rows (2023-01-02–2024-01-01), 66,388 train/validation rows (2024-01-08–2026-03-23), 15,804 final-test rows (2026-03-30–2026-09-21), and 8,512 future rows (2026-09-28–2026-12-28, 14 weeks). There are 72 missing historical targets. Of the 112,277 non-missing historical rows, 54,511 are genuine zero demand (48.55%).

There are 616 unique part-region series: 119 Smooth, 97 Erratic, 133 Intermittent, 220 Lumpy, and 47 short/unclassified. The catalogue contains 160 parts; the scheduling snapshot has 158. P0046 and P0087 are obsolete catalogue/history-only parts without scheduling or future rows.

## Validator result

Command: `python data/validate_datasets.py data`

**32 PASS, 0 FAIL.** Checks cover schema uniqueness, referential integrity, master-data consistency, supplier/SOB and inventory arithmetic, target/missing/future definitions, weekly continuity, future horizon and SBC class coverage. Original files were not changed.
