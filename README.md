# Financial Fraud Detection System

An end-to-end financial fraud detection platform for a highly imbalanced transaction dataset.

## Goals

- Data validation and cleaning
- Exploratory data analysis
- Feature engineering
- Imbalanced classification
- Logistic Regression
- Random Forest
- XGBoost
- TensorFlow/Keras Neural Network
- Stratified K-Fold cross-validation
- Hyperparameter tuning
- Threshold optimization
- Model evaluation and error analysis
- FastAPI prediction service
- React frontend dashboard
- Docker / Docker Compose
- Automated testing
- GitHub Actions CI/CD
- Automated model retraining
- Model validation and promotion
- Model versioning and rollback
- Monitoring

## Technology Stack

| Area | Technology |
|---|---|
| Language | Python |
| Data | Pandas, NumPy |
| ML | Scikit-learn |
| Gradient Boosting | XGBoost |
| Deep Learning | TensorFlow / Keras |
| API | FastAPI |
| Frontend | React (planned) |
| Database | PostgreSQL (planned) |
| Containers | Docker / Docker Compose |
| Testing | Pytest |
| CI/CD | GitHub Actions |

## Architecture

```text
Raw Data
   |
   v
Data Validation -> Cleaning -> EDA
   |
   v
Feature Engineering
   |
   v
Preprocessing + Imbalance Handling
   |
   v
Stratified K-Fold Cross-Validation
   |
   +--> Logistic Regression
   +--> Random Forest
   +--> XGBoost
   +--> TensorFlow Neural Network
   |
   v
Model Comparison -> Tuning -> Threshold Optimization
   |
   v
Final Model
   |
   v
FastAPI -> Frontend Dashboard
   |
   v
Docker -> CI/CD -> Retraining -> Validation -> Rollback
```

## Dataset

The current project uses a credit-card transaction dataset containing 284,807 transactions and 31 columns.

The target column is `Class`:

- `0` = legitimate
- `1` = fraudulent

The dataset is highly imbalanced, so accuracy will not be the primary model-selection metric.

**The raw CSV is intentionally excluded from Git tracking. Do not commit transaction data to a public repository.**

## Evaluation

Primary metrics:

- Precision
- Recall
- F1-score
- PR-AUC / Average Precision
- ROC-AUC
- Confusion matrix
- False positives / false negatives
- Threshold-performance trade-offs

## Roadmap

### ML
- [ ] Dataset audit
- [ ] Class distribution
- [ ] Duplicate investigation
- [ ] Data cleaning
- [ ] EDA
- [ ] Train/test strategy
- [ ] Preprocessing pipeline
- [ ] Imbalance handling
- [ ] Logistic Regression
- [ ] Random Forest
- [ ] XGBoost
- [ ] TensorFlow Neural Network
- [ ] Stratified K-Fold comparison
- [ ] Hyperparameter tuning
- [ ] Threshold optimization
- [ ] Error analysis
- [ ] Final model

### Backend
- [ ] FastAPI
- [ ] `/health`
- [ ] `/model/info`
- [ ] `/predict`
- [ ] `/predict/batch`
- [ ] Prediction logging

### Frontend
- [ ] React dashboard
- [ ] Transaction analyzer
- [ ] Prediction results
- [ ] Fraud analytics
- [ ] Model information

### Deployment / MLOps
- [ ] Docker
- [ ] Docker Compose
- [ ] Automated tests
- [ ] GitHub Actions CI
- [ ] Continuous deployment
- [ ] Scheduled retraining
- [ ] Model validation gate
- [ ] Model versioning
- [ ] Model promotion
- [ ] Rollback
- [ ] Monitoring

## Repository Structure

```text
financial-fraud-detection/
├── data/
│   ├── raw/
│   └── processed/
├── ml/
│   ├── notebooks/
│   ├── src/
│   │   ├── data/
│   │   ├── features/
│   │   ├── models/
│   │   ├── training/
│   │   └── evaluation/
│   └── artifacts/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── models/
│   │   ├── schemas/
│   │   ├── services/
│   │   └── main.py
│   └── tests/
├── frontend/
├── tests/
├── docker/
├── .github/workflows/
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Local Setup

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
pytest
```

Application and Docker commands will be added as development progresses.

## Current Status

**Repository setup complete.**

Next task:

**Task 2 — Analyze the fraud vs. legitimate transaction distribution.**

Each task will be implemented, tested, and committed before moving to the next stage.
