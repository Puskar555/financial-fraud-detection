"""
Task 18 — SQLite Persistence Layer

Stores fraud-detection inference records.

SQLite is used for local development.
The schema is designed so the persistence layer can later
be migrated to PostgreSQL.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import sqlite3


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

DATABASE_DIR = (
    PROJECT_ROOT
    / "data"
    / "database"
)

DATABASE_PATH = (
    DATABASE_DIR
    / "fraud_detection.db"
)


# ============================================================
# CONNECTION
# ============================================================

def get_connection() -> sqlite3.Connection:

    DATABASE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=30,
    )

    connection.row_factory = (
        sqlite3.Row
    )

    connection.execute(
        "PRAGMA foreign_keys = ON;"
    )

    connection.execute(
        "PRAGMA journal_mode = WAL;"
    )

    return connection


# ============================================================
# INITIALIZATION
# ============================================================

def initialize_database() -> None:

    with get_connection() as connection:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                transaction_id TEXT NOT NULL UNIQUE,

                timestamp_utc TEXT NOT NULL,

                model_name TEXT NOT NULL,

                model_version TEXT NOT NULL,

                prediction INTEGER NOT NULL
                    CHECK (prediction IN (0, 1)),

                label TEXT NOT NULL
                    CHECK (
                        label IN (
                            'legitimate',
                            'fraud'
                        )
                    ),

                fraud_probability REAL NOT NULL
                    CHECK (
                        fraud_probability >= 0.0
                        AND fraud_probability <= 1.0
                    ),

                threshold REAL NOT NULL
                    CHECK (
                        threshold >= 0.0
                        AND threshold <= 1.0
                    ),

                transaction_time REAL NOT NULL
                    CHECK (
                        transaction_time >= 0.0
                    ),

                amount REAL NOT NULL
                    CHECK (
                        amount >= 0.0
                    ),

                features_json TEXT NOT NULL
            );
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_predictions_timestamp
            ON predictions(timestamp_utc);
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_predictions_prediction
            ON predictions(prediction);
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_predictions_model_version
            ON predictions(model_version);
            """
        )

        connection.commit()


# ============================================================
# INSERT
# ============================================================

def insert_prediction(
    *,
    transaction_id: str,
    timestamp_utc: str,
    model_name: str,
    model_version: str,
    prediction: int,
    label: str,
    fraud_probability: float,
    threshold: float,
    features: dict[str, Any],
) -> None:

    if "Time" not in features:

        raise ValueError(
            "Transaction features are missing Time."
        )

    if "Amount" not in features:

        raise ValueError(
            "Transaction features are missing Amount."
        )

    features_json = json.dumps(
        features,
        separators=(",", ":"),
        allow_nan=False,
    )

    with get_connection() as connection:

        connection.execute(
            """
            INSERT INTO predictions (
                transaction_id,
                timestamp_utc,
                model_name,
                model_version,
                prediction,
                label,
                fraud_probability,
                threshold,
                transaction_time,
                amount,
                features_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                transaction_id,
                timestamp_utc,
                model_name,
                model_version,
                int(prediction),
                label,
                float(fraud_probability),
                float(threshold),
                float(features["Time"]),
                float(features["Amount"]),
                features_json,
            ),
        )

        connection.commit()


# ============================================================
# BATCH INSERT
# ============================================================

def insert_predictions(
    records: list[dict[str, Any]],
) -> None:

    if not records:

        return

    rows = []

    for record in records:

        features = record[
            "features"
        ]

        if "Time" not in features:

            raise ValueError(
                "Transaction features are missing Time."
            )

        if "Amount" not in features:

            raise ValueError(
                "Transaction features are missing Amount."
            )

        features_json = json.dumps(
            features,
            separators=(",", ":"),
            allow_nan=False,
        )

        rows.append(
            (
                record["transaction_id"],
                record["timestamp_utc"],
                record["model_name"],
                record["model_version"],
                int(record["prediction"]),
                record["label"],
                float(
                    record[
                        "fraud_probability"
                    ]
                ),
                float(
                    record["threshold"]
                ),
                float(
                    features["Time"]
                ),
                float(
                    features["Amount"]
                ),
                features_json,
            )
        )

    with get_connection() as connection:

        connection.executemany(
            """
            INSERT INTO predictions (
                transaction_id,
                timestamp_utc,
                model_name,
                model_version,
                prediction,
                label,
                fraud_probability,
                threshold,
                transaction_time,
                amount,
                features_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            rows,
        )

        connection.commit()


# ============================================================
# READ RECENT
# ============================================================

def get_recent_predictions(
    limit: int = 20,
) -> list[dict[str, Any]]:

    with get_connection() as connection:

        rows = connection.execute(
            """
            SELECT
                transaction_id,
                timestamp_utc,
                model_name,
                model_version,
                prediction,
                label,
                fraud_probability,
                threshold,
                transaction_time,
                amount
            FROM predictions
            ORDER BY id DESC
            LIMIT ?;
            """,
            (
                int(limit),
            ),
        ).fetchall()

    return [
        dict(row)
        for row in rows
    ]


# ============================================================
# COUNT
# ============================================================

def count_predictions() -> int:

    with get_connection() as connection:

        row = connection.execute(
            """
            SELECT COUNT(*) AS total
            FROM predictions;
            """
        ).fetchone()

    return int(
        row["total"]
    )


# ============================================================
# DATABASE STATS
# ============================================================

def get_prediction_stats() -> dict[str, Any]:

    with get_connection() as connection:

        row = connection.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(
                    CASE
                        WHEN prediction = 1
                        THEN 1
                        ELSE 0
                    END
                ) AS fraud_predictions,
                SUM(
                    CASE
                        WHEN prediction = 0
                        THEN 1
                        ELSE 0
                    END
                ) AS legitimate_predictions,
                AVG(fraud_probability)
                    AS average_fraud_probability,
                AVG(amount)
                    AS average_amount
            FROM predictions;
            """
        ).fetchone()

    return {
        "total":
            int(
                row["total"] or 0
            ),

        "fraud_predictions":
            int(
                row[
                    "fraud_predictions"
                ] or 0
            ),

        "legitimate_predictions":
            int(
                row[
                    "legitimate_predictions"
                ] or 0
            ),

        "average_fraud_probability":
            float(
                row[
                    "average_fraud_probability"
                ] or 0.0
            ),

        "average_amount":
            float(
                row[
                    "average_amount"
                ] or 0.0
            ),
    }


# ============================================================
# HEALTH CHECK
# ============================================================

def database_health_check() -> bool:

    try:

        with get_connection() as connection:

            row = connection.execute(
                "SELECT 1 AS ok;"
            ).fetchone()

        return (
            row["ok"] == 1
        )

    except sqlite3.Error:

        return False


# ============================================================
# DIRECT EXECUTION CHECK
# ============================================================

if __name__ == "__main__":

    initialize_database()

    print(
        "Database initialized successfully."
    )

    print(
        f"Database path: {DATABASE_PATH}"
    )

    print(
        f"Existing predictions: "
        f"{count_predictions()}"
    )

    print(
        f"Health check: "
        f"{database_health_check()}"
    )