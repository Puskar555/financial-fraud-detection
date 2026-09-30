"""
Task 17 — FastAPI Fraud Detection Backend

Endpoints
---------
GET  /health
GET  /model/info
GET  /metrics
GET  /transactions
POST /predict
POST /predict/batch

The API loads the versioned production candidate created in Task 16.

IMPORTANT
---------
No training, validation, or test dataset is loaded by this API.
"""

from __future__ import annotations

from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
import hashlib
import json
import time
import uuid

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

PRODUCTION_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "production"
)

CURRENT_MODEL_PATH = (
    PRODUCTION_ROOT
    / "current.json"
)


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

API_TITLE = "Financial Fraud Detection API"

API_VERSION = "1.0.0"

MAX_BATCH_SIZE = 1000

MAX_TRANSACTION_HISTORY = 1000


# ============================================================
# MODEL STATE
# ============================================================

MODEL: Any | None = None

MODEL_METADATA: dict[str, Any] = {}

CURRENT_CONFIG: dict[str, Any] = {}

MODEL_PATH: Path | None = None

MODEL_LOADED_AT: str | None = None


# ============================================================
# RUNTIME METRICS
# ============================================================

RUNTIME_LOCK = Lock()

RUNTIME_METRICS = {
    "total_requests": 0,
    "single_prediction_requests": 0,
    "batch_prediction_requests": 0,
    "transactions_scored": 0,
    "fraud_predictions": 0,
    "legitimate_predictions": 0,
    "prediction_errors": 0,
    "total_inference_ms": 0.0,
}


# ============================================================
# RECENT TRANSACTIONS
# ============================================================

TRANSACTION_LOCK = Lock()

RECENT_TRANSACTIONS: deque = deque(
    maxlen=MAX_TRANSACTION_HISTORY
)


# ============================================================
# REQUEST SCHEMA
# ============================================================

class TransactionFeatures(BaseModel):
    """
    Feature schema for one transaction.

    V1-V28 are anonymized numerical features from the
    credit-card fraud dataset.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    Time: float = Field(
        ...,
        ge=0,
        description="Elapsed time in seconds.",
    )

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    Amount: float = Field(
        ...,
        ge=0,
        description="Transaction amount.",
    )


class BatchPredictionRequest(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )

    transactions: list[
        TransactionFeatures
    ] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
    )


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class PredictionResponse(BaseModel):

    transaction_id: str

    prediction: int

    label: str

    fraud_probability: float

    threshold: float

    model_name: str

    model_version: str

    timestamp_utc: str


class BatchPredictionResponse(BaseModel):

    count: int

    fraud_count: int

    legitimate_count: int

    inference_ms: float

    predictions: list[
        PredictionResponse
    ]


# ============================================================
# UTILITY
# ============================================================

def utc_now() -> str:

    return datetime.now(
        timezone.utc
    ).isoformat()


def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

        for chunk in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):

            digest.update(
                chunk
            )

    return digest.hexdigest()


# ============================================================
# MODEL LOADING
# ============================================================

def load_production_model() -> None:

    global MODEL
    global MODEL_METADATA
    global CURRENT_CONFIG
    global MODEL_PATH
    global MODEL_LOADED_AT

    if not CURRENT_MODEL_PATH.exists():

        raise FileNotFoundError(
            "Production current.json not found: "
            f"{CURRENT_MODEL_PATH}"
        )

    with CURRENT_MODEL_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        current = json.load(
            file
        )

    required_current_fields = {
        "model_name",
        "model_version",
        "artifact_directory",
        "model_file",
        "metadata_file",
        "decision_threshold",
        "model_sha256",
    }

    missing = (
        required_current_fields
        - set(
            current.keys()
        )
    )

    if missing:

        raise ValueError(
            "current.json is missing fields: "
            f"{sorted(missing)}"
        )

    artifact_directory = (
        PRODUCTION_ROOT
        / current[
            "artifact_directory"
        ]
    )

    model_path = (
        artifact_directory
        / current[
            "model_file"
        ]
    )

    metadata_path = (
        artifact_directory
        / current[
            "metadata_file"
        ]
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Production model not found: "
            f"{model_path}"
        )

    if not metadata_path.exists():

        raise FileNotFoundError(
            "Production metadata not found: "
            f"{metadata_path}"
        )

    # --------------------------------------------------------
    # CHECKSUM
    # --------------------------------------------------------

    actual_hash = sha256_file(
        model_path
    )

    expected_hash = current[
        "model_sha256"
    ]

    if actual_hash != expected_hash:

        raise RuntimeError(
            "Production model SHA-256 "
            "checksum verification failed."
        )

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(
            file
        )

    expected_features = metadata.get(
        "expected_features"
    )

    if not expected_features:

        raise ValueError(
            "metadata.json does not contain "
            "expected_features."
        )

    if len(
        expected_features
    ) != 30:

        raise ValueError(
            "Expected exactly 30 model features."
        )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = joblib.load(
        model_path
    )

    if not hasattr(
        model,
        "predict_proba",
    ):

        raise TypeError(
            "Production model does not support "
            "predict_proba()."
        )

    if hasattr(
        model,
        "n_features_in_",
    ):

        if (
            int(
                model.n_features_in_
            )
            != len(
                expected_features
            )
        ):

            raise ValueError(
                "Production model feature-count "
                "mismatch."
            )

    # --------------------------------------------------------
    # CROSS-CHECK CURRENT VS METADATA
    # --------------------------------------------------------

    if (
        current["model_name"]
        != metadata["model_name"]
    ):

        raise ValueError(
            "Model name mismatch between "
            "current.json and metadata.json."
        )

    if (
        current["model_version"]
        != metadata["model_version"]
    ):

        raise ValueError(
            "Model version mismatch between "
            "current.json and metadata.json."
        )

    if not np.isclose(
        float(
            current[
                "decision_threshold"
            ]
        ),
        float(
            metadata[
                "decision_threshold"
            ]
        ),
    ):

        raise ValueError(
            "Threshold mismatch between "
            "current.json and metadata.json."
        )

    MODEL = model

    MODEL_METADATA = metadata

    CURRENT_CONFIG = current

    MODEL_PATH = model_path

    MODEL_LOADED_AT = utc_now()


# ============================================================
# MODEL ACCESS
# ============================================================

def ensure_model_loaded() -> None:

    if MODEL is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Fraud detection model "
                "is not available."
            ),
        )


def get_expected_features() -> list[str]:

    ensure_model_loaded()

    return MODEL_METADATA[
        "expected_features"
    ]


def get_threshold() -> float:

    ensure_model_loaded()

    return float(
        MODEL_METADATA[
            "decision_threshold"
        ]
    )


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_dataframe(
    transactions: list[
        TransactionFeatures
    ],
) -> pd.DataFrame:

    expected_features = (
        get_expected_features()
    )

    rows = [
        transaction.model_dump()
        for transaction in transactions
    ]

    frame = pd.DataFrame(
        rows
    )

    missing = [
        feature
        for feature
        in expected_features
        if feature not in frame.columns
    ]

    if missing:

        raise HTTPException(
            status_code=422,
            detail=(
                "Missing required features: "
                f"{missing}"
            ),
        )

    frame = frame[
        expected_features
    ]

    try:

        frame = frame.astype(
            np.float64
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise HTTPException(
            status_code=422,
            detail=(
                "All model features must "
                "be numeric."
            ),
        ) from exc

    values = frame.to_numpy()

    if not np.isfinite(
        values
    ).all():

        raise HTTPException(
            status_code=422,
            detail=(
                "Features cannot contain "
                "NaN or infinity."
            ),
        )

    return frame


# ============================================================
# INFERENCE
# ============================================================

def score_transactions(
    transactions: list[
        TransactionFeatures
    ],
) -> tuple[
    list[PredictionResponse],
    float,
]:

    ensure_model_loaded()

    frame = prepare_dataframe(
        transactions
    )

    threshold = get_threshold()

    start = time.perf_counter()

    try:

        probabilities = (
            MODEL.predict_proba(
                frame
            )[:, 1]
        )

    except Exception as exc:

        with RUNTIME_LOCK:

            RUNTIME_METRICS[
                "prediction_errors"
            ] += 1

        raise HTTPException(
            status_code=500,
            detail=(
                "Model inference failed."
            ),
        ) from exc

    inference_ms = (
        time.perf_counter()
        - start
    ) * 1000.0

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if len(
        probabilities
    ) != len(
        transactions
    ):

        raise HTTPException(
            status_code=500,
            detail=(
                "Model returned an unexpected "
                "number of predictions."
            ),
        )

    if not np.isfinite(
        probabilities
    ).all():

        raise HTTPException(
            status_code=500,
            detail=(
                "Model returned invalid "
                "probabilities."
            ),
        )

    predictions = (
        probabilities
        >= threshold
    ).astype(int)

    timestamp = utc_now()

    responses = []

    history_records = []

    for (
        probability,
        prediction,
    ) in zip(
        probabilities,
        predictions,
    ):

        transaction_id = str(
            uuid.uuid4()
        )

        response = PredictionResponse(
            transaction_id=
                transaction_id,

            prediction=
                int(
                    prediction
                ),

            label=(
                "fraud"
                if prediction == 1
                else "legitimate"
            ),

            fraud_probability=
                float(
                    probability
                ),

            threshold=
                threshold,

            model_name=
                CURRENT_CONFIG[
                    "model_name"
                ],

            model_version=
                CURRENT_CONFIG[
                    "model_version"
                ],

            timestamp_utc=
                timestamp,
        )

        responses.append(
            response
        )

        history_records.append(
            response.model_dump()
        )

    fraud_count = int(
        predictions.sum()
    )

    legitimate_count = (
        len(
            predictions
        )
        - fraud_count
    )

    with RUNTIME_LOCK:

        RUNTIME_METRICS[
            "transactions_scored"
        ] += len(
            predictions
        )

        RUNTIME_METRICS[
            "fraud_predictions"
        ] += fraud_count

        RUNTIME_METRICS[
            "legitimate_predictions"
        ] += legitimate_count

        RUNTIME_METRICS[
            "total_inference_ms"
        ] += inference_ms

    with TRANSACTION_LOCK:

        RECENT_TRANSACTIONS.extend(
            history_records
        )

    return (
        responses,
        inference_ms,
    )


# ============================================================
# APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    print(
        "\nLoading production fraud model..."
    )

    load_production_model()

    print(
        "Production model loaded successfully."
    )

    print(
        f"Model     : "
        f"{CURRENT_CONFIG['model_name']}"
    )

    print(
        f"Version   : "
        f"{CURRENT_CONFIG['model_version']}"
    )

    print(
        f"Threshold : "
        f"{get_threshold():.3f}"
    )

    print(
        "No dataset was loaded.\n"
    )

    yield


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=(
        "Inference API for the financial "
        "fraud detection pipeline."
    ),
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

# Development configuration.
# We will tighten this for deployment.

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "service":
            API_TITLE,

        "version":
            API_VERSION,

        "docs":
            "/docs",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    ensure_model_loaded()

    return {
        "status":
            "healthy",

        "model_loaded":
            True,

        "model_name":
            CURRENT_CONFIG[
                "model_name"
            ],

        "model_version":
            CURRENT_CONFIG[
                "model_version"
            ],

        "model_loaded_at_utc":
            MODEL_LOADED_AT,
    }


# ============================================================
# MODEL INFO
# ============================================================

@app.get("/model/info")
def model_info():

    ensure_model_loaded()

    return {
        "model_name":
            CURRENT_CONFIG[
                "model_name"
            ],

        "model_version":
            CURRENT_CONFIG[
                "model_version"
            ],

        "stage":
            CURRENT_CONFIG[
                "stage"
            ],

        "decision_threshold":
            get_threshold(),

        "expected_feature_count":
            len(
                get_expected_features()
            ),

        "expected_features":
            get_expected_features(),

        "model_sha256":
            CURRENT_CONFIG[
                "model_sha256"
            ],

        "model_loaded_at_utc":
            MODEL_LOADED_AT,

        "selection_evidence":
            MODEL_METADATA.get(
                "selection_evidence",
                {},
            ),
    }


# ============================================================
# SINGLE PREDICTION
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    transaction: TransactionFeatures,
):

    with RUNTIME_LOCK:

        RUNTIME_METRICS[
            "total_requests"
        ] += 1

        RUNTIME_METRICS[
            "single_prediction_requests"
        ] += 1

    predictions, _ = (
        score_transactions(
            [transaction]
        )
    )

    return predictions[0]


# ============================================================
# BATCH PREDICTION
# ============================================================

@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
)
def predict_batch(
    request: BatchPredictionRequest,
):

    with RUNTIME_LOCK:

        RUNTIME_METRICS[
            "total_requests"
        ] += 1

        RUNTIME_METRICS[
            "batch_prediction_requests"
        ] += 1

    predictions, inference_ms = (
        score_transactions(
            request.transactions
        )
    )

    fraud_count = sum(
        prediction.prediction
        for prediction
        in predictions
    )

    return BatchPredictionResponse(
        count=len(
            predictions
        ),

        fraud_count=
            fraud_count,

        legitimate_count=(
            len(
                predictions
            )
            - fraud_count
        ),

        inference_ms=
            inference_ms,

        predictions=
            predictions,
    )


# ============================================================
# RUNTIME METRICS
# ============================================================

@app.get("/metrics")
def metrics():

    ensure_model_loaded()

    with RUNTIME_LOCK:

        snapshot = dict(
            RUNTIME_METRICS
        )

    transactions_scored = snapshot[
        "transactions_scored"
    ]

    if transactions_scored:

        average_inference_ms = (
            snapshot[
                "total_inference_ms"
            ]
            / transactions_scored
        )

        fraud_prediction_rate = (
            snapshot[
                "fraud_predictions"
            ]
            / transactions_scored
        )

    else:

        average_inference_ms = 0.0

        fraud_prediction_rate = 0.0

    return {
        **snapshot,

        "average_inference_ms_per_transaction":
            average_inference_ms,

        "fraud_prediction_rate":
            fraud_prediction_rate,

        "history_size":
            len(
                RECENT_TRANSACTIONS
            ),
    }


# ============================================================
# RECENT TRANSACTIONS
# ============================================================

@app.get("/transactions")
def transactions(
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
):

    ensure_model_loaded()

    with TRANSACTION_LOCK:

        records = list(
            RECENT_TRANSACTIONS
        )

    records = records[
        -limit:
    ]

    records.reverse()

    return {
        "count":
            len(
                records
            ),

        "transactions":
            records,
    }