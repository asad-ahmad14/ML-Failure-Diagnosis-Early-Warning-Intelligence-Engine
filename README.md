# ML Failure Diagnosis & Early-Warning Engine

A FastAPI service that screens sensor readings for unusual behaviour and elevated failure risk, using two separate scikit-learn models, a deterministic warning layer, PostgreSQL-backed history and alerts, and a Streamlit dashboard.

> **This is a screening tool built on a synthetic dataset.** It does not diagnose physical root causes, forecast failures, or give failure probabilities. See [What this project does not claim](#what-this-project-does-not-claim).

## What it does

For one reading of machine sensor values the API returns:

- a **failure-risk score** from a Random Forest classifier (a ranking score, not a probability)
- an **anomaly score** from an Isolation Forest trained without labels
- a **warning level** (normal, watch, elevated, high) from fixed thresholds
- **evidence**: the top three features by SHAP contribution, with their actual values
- stored history, deduplicated alerts, a model-info endpoint and a drift report

## Dataset

AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository (id 601), **CC BY 4.0**. It is **synthetic**: 10,000 rows, 339 failures (3.39%), no timestamps and no machine ids.

Notes from the exploratory analysis (`ml/eda.py`):

- 27 rows have a `Machine failure` label that disagrees with the five failure-mode columns (18 have an active mode but no failure, 9 the reverse). The published target is used unchanged.
- The failure-mode columns define the target, so they are never used as features.
- Random Failure (RNF) is random by design and cannot be learned.

## ML workflow

| Step | Script | What it does |
|---|---|---|
| Ingest | `ml.ingest` | Downloads the data and prints basic validation. |
| Split | `ml.split` | Stratified 80/20 dev/test split, seed 42. |
| Compare | `ml.compare_models` | 5-fold stratified cross-validation on dev only. |
| Train | `ml.train_final` | Fits Random Forest on all dev rows, chooses thresholds from out-of-fold scores. |
| Anomaly | `ml.train_anomaly` | Isolation Forest on the 7 numeric features, flags the top 1% of dev scores. |
| Explain | `ml.explain` | SHAP TreeExplainer, global importance, reference statistics for drift. |
| Test | `ml.evaluate_test` | One-time held-out evaluation with bootstrap intervals. |

Features: product type, air temperature, process temperature, rotational speed, torque, tool wear, plus two engineered features (temperature difference and mechanical power).

### Results (measured by running the scripts)

Dev cross-validation, 5-fold:

| Model | PR-AUC (out-of-fold) | ROC-AUC |
|---|---|---|
| HistGradientBoosting | 0.8749 | 0.9767 |
| Random Forest (chosen) | 0.8719 | 0.9793 |
| Logistic Regression | 0.4810 | 0.9207 |
| No-skill baseline | 0.0338 | 0.4985 |

Random Forest and HistGradientBoosting differ by less than one tenth of the fold-to-fold spread (about 0.03), so this is a tie. Random Forest was chosen because SHAP's TreeExplainer supports it.

Held-out test set (2,000 rows, 68 failures): **PR-AUC 0.858** (95% bootstrap interval 0.772 to 0.930), ROC-AUC 0.974. Always predicting "no failure" scores 96.6% accuracy, which is why accuracy is not reported as a result.

| Level | Rule (set on dev out-of-fold scores) | Cutoff | Test precision | Test recall |
|---|---|---|---|---|
| Watch | recall at least 0.90 | 0.1092 | 0.394 (0.318 to 0.475) | 0.927 (0.859 to 0.985) |
| Elevated | recall at least 0.80 | 0.3662 | 0.862 (0.774 to 0.937) | 0.824 (0.730 to 0.909) |
| High | precision at least 0.95 | 0.5170 | 0.929 (0.854 to 0.985) | 0.765 (0.667 to 0.862) |

Intervals are wide because the test set has only 68 failures.

**Main weakness:** tool-wear failures are largely missed (1 of 10 caught at the Elevated cutoff on the test set, 3 of 36 on dev). UCI describes them as occurring at a randomly chosen wear time, which the features cannot capture.

## API

Interactive docs at `/docs` once the service is running.

| Method and path | Purpose |
|---|---|
| `GET /health` | Service, model and database status |
| `POST /predict` | Score one reading, store it, create an alert if warranted |
| `POST /predict/batch` | Up to 100 readings (413 above that), one transaction |
| `GET /predictions`, `/predictions/{id}` | History with filtering and pagination |
| `GET /alerts`, `/alerts/{id}` | Alert list with status and level filters |
| `PATCH /alerts/{id}/status` | open to acknowledged to resolved (409 otherwise) |
| `GET /model` | Metrics and thresholds, read from saved artifact files |
| `GET /monitoring/report` | Warning counts, score spread, input drift (PSI) |

Warning rules: the level comes from the risk score and the three cutoffs. A statistically unusual reading with a normal risk score is raised to Watch only. Repeated identical readings do not create duplicate alerts while one is open or acknowledged.

## Setup

Requires Python 3.12. Data and model files are not committed, so build them first.

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows PowerShell
pip install -r requirements-dev.txt

python -m ml.ingest
python -m ml.split
python -m ml.compare_models
python -m ml.train_final
python -m ml.train_anomaly
python -m ml.explain
python -m ml.evaluate_test
```

Run locally with SQLite:

```bash
Copy-Item .env.example .env
alembic upgrade head
uvicorn app.main:app --reload
```

Dashboard (needs the API running):

```bash
pip install -r requirements-dashboard.txt
streamlit run dashboard/app.py
```

Optional demo traffic: `python scripts/simulate_traffic.py --n 200`

### Docker Compose (PostgreSQL)

```bash
Copy-Item .env.example .env           # then set your own POSTGRES_PASSWORD
docker compose up --build
```

The `artifacts/` folder is mounted read-only, so run the training commands first. Migrations run automatically at container start. Data persists in the `pgdata` volume; `docker compose down -v` deletes it. Do not change `POSTGRES_PASSWORD` after the first start without running `down -v`, because Postgres stores the password in its volume.

## Tests

```bash
pytest -q
```

44 tests pass (warning engine, API validation, the real model through the API, persistence and deduplication, alert lifecycle, monitoring). Tests use a temporary SQLite database, never `app.db`. Coverage has not been measured.

## What this project does not claim

- It does not predict failures ahead of time. The data has no timestamps, so each reading is scored on its own.
- `risk_score` is not a probability of failure. The model is class-weighted.
- A statistical anomaly is not a failure, and a high risk score is not a confirmed fault.
- SHAP values show how the model uses its inputs, not physical causes.
- Results come from a synthetic dataset and say nothing about real machines.
- It is not load-tested or security-hardened: there is no authentication or rate limiting.

## Known limitations

- Thresholds were set on out-of-fold scores from models trained on 80% of dev, while the final model uses 100%, so the cutoffs are approximate.
- Alert deduplication is checked in application code. Two simultaneous identical requests could create two alerts; a database-level partial unique index would close that gap.
- Only direct dependencies are pinned; their dependencies float.
- Data drift is measured with PSI bands that are rules of thumb, and it says nothing about model accuracy because incoming readings have no failure labels.
- The test suite emits one Starlette deprecation warning about its HTTP client.

## Repository layout

```
app/        FastAPI service (routes, schemas, services, models, db)
ml/         training, evaluation and explanation scripts
migrations/ Alembic migrations
dashboard/  Streamlit app
scripts/    demo traffic generator
tests/      pytest suite
docs/       analysis output and screenshots
```

Screenshots to add under `docs/screenshots/`: the scoring tab, the alerts tab, and the model and monitoring tab.


## Attribution

Dataset: AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository, licensed CC BY 4.0. Original paper: S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications", 2020.