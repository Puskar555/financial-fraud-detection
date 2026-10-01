# Financial Fraud Detection Pipeline

An end-to-end machine learning system for credit-card transaction fraud detection, built from data preparation through model evaluation, threshold optimization, production inference, REST API deployment, database persistence, and a React dashboard.

The project is designed as a complete ML engineering pipeline rather than a notebook-only experiment.

## Overview

The system takes a transaction containing the standard `Time`, `V1`–`V28`, and `Amount` features and returns:

- fraud probability
- fraud/legitimate decision
- decision threshold
- model name and version
- transaction ID
- UTC processing timestamp

Predictions are persisted in SQLite and exposed through a React dashboard.

### Architecture

```text
                    ┌─────────────────────────┐
                    │     Transaction Input    │
                    │ Time, V1–V28, Amount     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │      React Dashboard     │
                    │     + Nginx frontend     │
                    └────────────┬────────────┘
                                 │ /api
                                 ▼
                    ┌─────────────────────────┐
                    │       FastAPI API        │
                    │ /predict /metrics /...   │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Production XGBoost 1.0.0│
                    │ Threshold = 0.13        │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
          ┌─────────────────┐       ┌──────────────────┐
          │ Prediction JSON │       │ SQLite Database  │
          │ + model metadata│       │ Prediction log   │
          └─────────────────┘       └──────────────────┘
```

## ML Pipeline

The project follows a staged workflow:

1. Data understanding
2. Exploratory data analysis
3. Data cleaning and validation
4. Train/validation/test separation
5. Feature preprocessing
6. Baseline model training
7. Cross-validation
8. Hyperparameter tuning
9. Model comparison
10. Threshold optimization
11. Error analysis
12. Production model selection
13. FastAPI deployment
14. Database-backed prediction logging
15. React dashboard
16. Docker deployment

### Models

The project evaluates multiple model families:

- Logistic Regression
- Random Forest
- XGBoost
- TensorFlow/Keras Neural Network

The production candidate currently deployed is:

```text
Model:              XGBoost
Version:            1.0.0
Decision threshold: 0.13
```

## Validation Results

The production model metadata records the following results.

| Metric | Result |
|---|---:|
| Tuned CV PR-AUC | 0.857653 |
| Validation PR-AUC | 0.845791 |
| Validation ROC-AUC | 0.973013 |
| Precision @ 0.13 | 0.981818 |
| Recall @ 0.13 | 0.760563 |
| F1 @ 0.13 | 0.857143 |
| True positives | 54 |
| False positives | 1 |
| False negatives | 17 |
| True negatives | 42,487 |

These are validation results associated with the production candidate. They should not be interpreted as performance on an unseen production population.

## Why PR-AUC?

Fraud detection is an imbalanced classification problem. Accuracy alone can be misleading when legitimate transactions substantially outnumber fraudulent transactions.

The pipeline therefore uses **Average Precision / PR-AUC as the primary model-selection objective**, while also tracking ROC-AUC, precision, recall, and F1.

The decision threshold is treated separately from model training. The production threshold is currently:

```text
0.13
```

A transaction is classified as fraud when its predicted fraud probability meets or exceeds this threshold.

## Production Artifacts

The versioned production model is stored under:

```text
artifacts/
└── production/
    ├── current.json
    └── xgboost/
        └── 1.0.0/
            ├── metadata.json
            └── model.joblib
```

`current.json` acts as the production model pointer and records:

- model name
- model version
- artifact directory
- model file
- metadata file
- decision threshold
- SHA-256 model checksum

Intermediate experiment artifacts are intentionally excluded from the Git release.

## Backend

The backend is implemented with **FastAPI**.

### API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Service/model/database health |
| GET | `/model/info` | Production model metadata |
| GET | `/metrics` | Runtime and database prediction metrics |
| GET | `/transactions` | Recent prediction history |
| POST | `/predict` | Score one transaction |
| POST | `/predict/batch` | Score multiple transactions |

### Example single prediction

```json
{
  "Time": 56861.0,
  "V1": -1.6209258457,
  "V2": 1.4084276234,
  "V3": 0.873390222,
  "V4": 0.1965129705,
  "V5": -0.426807709,
  "V6": -0.2083000417,
  "V7": -0.0895107089,
  "V8": 0.7673280231,
  "V9": -0.4468375176,
  "V10": -0.8596123977,
  "V11": -0.7400106381,
  "V12": 1.0152351995,
  "V13": 0.880864312,
  "V14": 0.2268431876,
  "V15": -0.3215896291,
  "V16": -0.5313055088,
  "V17": 0.6048687293,
  "V18": -1.0343849513,
  "V19": 0.0549112897,
  "V20": -0.3030692359,
  "V21": 0.031090404,
  "V22": -0.1073176961,
  "V23": -0.0947172706,
  "V24": 0.1437349691,
  "V25": -0.1273035947,
  "V26": 0.2115662131,
  "V27": -0.5452158637,
  "V28": -0.0403887145,
  "Amount": 9.65
}
```

A successful response contains fields such as:

```json
{
  "transaction_id": "uuid",
  "prediction": 0,
  "label": "legitimate",
  "fraud_probability": 0.00000046,
  "threshold": 0.13,
  "model_name": "xgboost",
  "model_version": "1.0.0",
  "timestamp_utc": "..."
}
```

## Database

Prediction results are persisted in SQLite.

The database stores information including:

- transaction ID
- timestamp
- model name/version
- prediction
- fraud probability
- threshold
- transaction time
- transaction amount

The dashboard uses the stored prediction history for its transaction view and metrics.

## Frontend

The frontend is a React application served through Nginx.

The dashboard provides:

- real-time transaction analysis
- fraud/legitimate result display
- fraud probability
- decision threshold
- model/version information
- transaction ID
- prediction history
- runtime metrics
- database prediction statistics
- sample transaction loading for UI testing

The frontend communicates with the backend through the `/api` Nginx route.

## Docker Deployment

The application can be run as a two-service Docker Compose deployment:

```text
frontend
   │
   │ /api
   ▼
backend
   │
   ├── XGBoost production model
   │
   └── SQLite
```

### Services

| Service | Technology | Port |
|---|---|---:|
| backend | FastAPI + Uvicorn | 8001 |
| frontend | React + Nginx | 5173 |

### Start the application

From the project root:

```bash
docker compose build
docker compose up -d
```

Check the services:

```bash
docker compose ps
```

Open the dashboard:

```text
http://127.0.0.1:5173
```

Check the API through the frontend proxy:

```bash
curl http://127.0.0.1:5173/api/health
```

Stop the application:

```bash
docker compose down
```

## Local Development

### Python environment

The project uses Python for the ML pipeline and backend.

Install the required development dependencies from the project's requirements/environment configuration.

Run the test suite:

```bash
python -m pytest -q
```

Run Ruff:

```bash
python -m ruff check .
```

The validated release currently passes:

```text
8 passed
All Ruff checks passed
```

### Backend

The FastAPI application can be started with Uvicorn from the project root:

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8001
```

### Frontend

The React application is located under:

```text
frontend/
```

Install dependencies and start the development server using the package scripts defined in `frontend/package.json`.

## Project Structure

```text
financial-fraud-detection/
│
├── backend/
│   └── app/
│       ├── db/
│       │   └── database.py
│       └── main.py
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   └── App.jsx
│   ├── Dockerfile
│   ├── nginx.conf
│   └── package.json
│
├── ml/
│   ├── notebooks/
│   │   ├── 01_data_understanding.ipynb
│   │   └── 02_exploratory_data_analysis.ipynb
│   │
│   └── src/
│       ├── data/
│       ├── evaluation/
│       ├── features/
│       └── models/
│
├── artifacts/
│   └── production/
│       ├── current.json
│       └── xgboost/
│           └── 1.0.0/
│
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
└── README.md
```

## Testing Strategy

The project includes automated tests for core functionality and static quality checks.

Current release validation:

```text
pytest: 8 passed
ruff:   all checks passed
```

The deployed application was also manually validated through:

- health endpoint
- model information endpoint
- metrics endpoint
- transaction history endpoint
- single prediction
- batch prediction
- fraud prediction
- legitimate prediction
- React dashboard
- SQLite persistence
- Docker Compose deployment

## Reproducibility and Model Integrity

The production model is versioned and referenced through `current.json`.

The production pointer contains a SHA-256 checksum for the model artifact:

```text
ec8c3ac94aeb1c3931832809d131317cf97240c03c9a9fb916c87462aa2ff1a6
```

This provides an additional integrity check for the deployed model artifact.

## Data and Privacy

Raw transaction datasets are intentionally not included in the Git repository.

The repository ignores:

- raw datasets
- processed datasets
- local databases
- environment files
- secrets
- generated experiment artifacts
- Python caches
- frontend build output
- local development environments

Only the versioned production model artifacts required by the deployment are included.

## Limitations

This project is a machine-learning engineering and deployment demonstration. Several limitations should be considered before treating it as a production banking or payment-system solution:

- Validation performance does not guarantee future production performance.
- Fraud distributions can change over time.
- The dataset may not represent every real-world transaction environment.
- Probability outputs should not automatically be interpreted as calibrated real-world fraud probabilities.
- Threshold selection depends on the operational cost of false positives and false negatives.
- Production monitoring would need to include data drift, concept drift, model performance, latency, and alert quality.
- Real financial deployments require additional security, privacy, governance, auditability, access control, and regulatory controls.

## Future Improvements

Potential extensions include:

- automated model retraining
- model/data drift monitoring
- calibration analysis
- online feature pipelines
- experiment tracking
- model registry integration
- authentication and authorization
- structured application logging
- Prometheus/Grafana monitoring
- CI/CD deployment
- cloud deployment
- champion/challenger model evaluation
- automated rollback of model versions

## Project Status

The current repository contains a tested end-to-end implementation covering:

```text
Data
  ↓
EDA
  ↓
Cleaning
  ↓
Feature processing
  ↓
Model training
  ↓
Cross-validation
  ↓
Hyperparameter tuning
  ↓
Threshold optimization
  ↓
Error analysis
  ↓
Production model selection
  ↓
FastAPI
  ↓
SQLite
  ↓
React + Nginx
  ↓
Docker Compose
```

The current production candidate is **XGBoost 1.0.0** with a decision threshold of **0.13**.
