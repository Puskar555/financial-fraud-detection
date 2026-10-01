"""
Task 13 — Hyperparameter Tuning
Financial Fraud Detection Project

Tunes:
1. Random Forest
2. XGBoost
3. TensorFlow/Keras Neural Network

Methodology
-----------
- Uses train.csv ONLY.
- validation.csv is untouched.
- test.csv is untouched.
- Stratified 3-fold CV.
- PR-AUC / Average Precision is the primary objective.
- Preprocessing for the neural network is fitted inside each fold.
- No SMOTE/resampling.
- No class weighting.
- No threshold optimization.
"""
import gc
import itertools
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from ml.src.features.preprocessing import (
    TARGET_COLUMN,
    build_preprocessor,
    validate_features,
)

# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 3

NN_EPOCHS = 27
NN_MAX_EPOCHS = NN_EPOCHS
NN_PATIENCE = 5
NN_BATCH_SIZE = 512

# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_random_seeds(seed=RANDOM_STATE):

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ============================================================
# DATA
# ============================================================

def load_training_data(path: Path):

    if not path.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {path}"
        )

    df = pd.read_csv(path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Missing target column: {TARGET_COLUMN}"
        )

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].copy()

    validate_features(X)

    if set(y.unique()) != {0, 1}:
        raise ValueError(
            "Target must contain classes 0 and 1."
        )

    return X, y


# ============================================================
# PARAMETER GRIDS
# ============================================================

RF_GRID = {
    "n_estimators": [
        200,
        400,
    ],

    "max_depth": [
        None,
        12,
        20,
    ],

    "min_samples_split": [
        2,
        5,
    ],

    "max_features": [
        "sqrt",
        0.7,
    ],
}


XGB_GRID = {
    "n_estimators": [
        200,
        400,
    ],

    "learning_rate": [
        0.03,
        0.10,
    ],

    "max_depth": [
        3,
        6,
    ],

    "subsample": [
        0.8,
        1.0,
    ],

    "colsample_bytree": [
        0.8,
        1.0,
    ],
}


NN_CONFIGS = [
    {
        "hidden_layers": [64, 32],
        "dropout": 0.20,
        "learning_rate": 0.001,
    },

    {
        "hidden_layers": [64, 32, 16],
        "dropout": 0.30,
        "learning_rate": 0.001,
    },

    {
        "hidden_layers": [128, 64, 32],
        "dropout": 0.30,
        "learning_rate": 0.001,
    },

    {
        "hidden_layers": [64, 32, 16],
        "dropout": 0.20,
        "learning_rate": 0.0005,
    },
]


# ============================================================
# PARAMETER COMBINATIONS
# ============================================================

def parameter_combinations(grid):

    keys = list(grid.keys())

    values = [
        grid[key]
        for key in keys
    ]

    for combination in itertools.product(
        *values
    ):

        yield dict(
            zip(
                keys,
                combination,
            )
        )


# ============================================================
# RANDOM FOREST TUNING
# ============================================================

def tune_random_forest(
    X,
    y,
    cv,
):

    print("\n" + "=" * 72)
    print("RANDOM FOREST HYPERPARAMETER TUNING")
    print("=" * 72)

    results = []

    combinations = list(
        parameter_combinations(
            RF_GRID
        )
    )

    print(
        f"\nConfigurations: "
        f"{len(combinations)}"
    )

    for config_number, params in enumerate(
        combinations,
        start=1,
    ):

        print(
            f"\nRF configuration "
            f"{config_number}/"
            f"{len(combinations)}"
        )

        print(params)

        fold_scores = []
        fold_times = []

        for fold, (
            train_idx,
            val_idx,
        ) in enumerate(
            cv.split(X, y),
            start=1,
        ):

            X_train = X.iloc[
                train_idx
            ]

            y_train = y.iloc[
                train_idx
            ]

            X_val = X.iloc[
                val_idx
            ]

            y_val = y.iloc[
                val_idx
            ]

            model = RandomForestClassifier(
                **params,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )

            start = time.perf_counter()

            model.fit(
                X_train,
                y_train,
            )

            probabilities = (
                model.predict_proba(
                    X_val
                )[:, 1]
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            score = (
                average_precision_score(
                    y_val,
                    probabilities,
                )
            )

            fold_scores.append(
                score
            )

            fold_times.append(
                elapsed
            )

            print(
                f"  Fold {fold}: "
                f"PR-AUC={score:.6f} "
                f"time={elapsed:.2f}s"
            )

            del model
            gc.collect()

        result = {
            "model":
                "random_forest",

            **params,

            "pr_auc_mean":
                float(
                    np.mean(
                        fold_scores
                    )
                ),

            "pr_auc_std":
                float(
                    np.std(
                        fold_scores,
                        ddof=1,
                    )
                ),

            "training_seconds_mean":
                float(
                    np.mean(
                        fold_times
                    )
                ),
        }

        results.append(
            result
        )

        print(
            f"  Mean PR-AUC: "
            f"{result['pr_auc_mean']:.6f}"
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# XGBOOST TUNING
# ============================================================

def tune_xgboost(
    X,
    y,
    cv,
):

    print("\n" + "=" * 72)
    print("XGBOOST HYPERPARAMETER TUNING")
    print("=" * 72)

    results = []

    combinations = list(
        parameter_combinations(
            XGB_GRID
        )
    )

    print(
        f"\nConfigurations: "
        f"{len(combinations)}"
    )

    for config_number, params in enumerate(
        combinations,
        start=1,
    ):

        print(
            f"\nXGB configuration "
            f"{config_number}/"
            f"{len(combinations)}"
        )

        print(params)

        fold_scores = []
        fold_times = []

        for fold, (
            train_idx,
            val_idx,
        ) in enumerate(
            cv.split(X, y),
            start=1,
        ):

            X_train = X.iloc[
                train_idx
            ]

            y_train = y.iloc[
                train_idx
            ]

            X_val = X.iloc[
                val_idx
            ]

            y_val = y.iloc[
                val_idx
            ]

            model = XGBClassifier(
                **params,

                objective=
                    "binary:logistic",

                eval_metric=
                    "logloss",

                random_state=
                    RANDOM_STATE,

                n_jobs=-1,

                tree_method=
                    "hist",
            )

            start = time.perf_counter()

            model.fit(
                X_train,
                y_train,
            )

            probabilities = (
                model.predict_proba(
                    X_val
                )[:, 1]
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            score = (
                average_precision_score(
                    y_val,
                    probabilities,
                )
            )

            fold_scores.append(
                score
            )

            fold_times.append(
                elapsed
            )

            print(
                f"  Fold {fold}: "
                f"PR-AUC={score:.6f} "
                f"time={elapsed:.2f}s"
            )

            del model
            gc.collect()

        result = {
            "model":
                "xgboost",

            **params,

            "pr_auc_mean":
                float(
                    np.mean(
                        fold_scores
                    )
                ),

            "pr_auc_std":
                float(
                    np.std(
                        fold_scores,
                        ddof=1,
                    )
                ),

            "training_seconds_mean":
                float(
                    np.mean(
                        fold_times
                    )
                ),
        }

        results.append(
            result
        )

        print(
            f"  Mean PR-AUC: "
            f"{result['pr_auc_mean']:.6f}"
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# BUILD NN
# ============================================================

def build_neural_network(
    input_dim,
    hidden_layers,
    dropout,
    learning_rate,
):

    model = tf.keras.Sequential()

    model.add(
        tf.keras.layers.Input(
            shape=(input_dim,)
        )
    )

    for units in hidden_layers:

        model.add(
            tf.keras.layers.Dense(
                units,
                activation="relu",
            )
        )

        model.add(
            tf.keras.layers.Dropout(
                dropout
            )
        )

    model.add(
        tf.keras.layers.Dense(
            1,
            activation="sigmoid",
        )
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=
                learning_rate
        ),

        loss=
            "binary_crossentropy",

        metrics=[
            tf.keras.metrics.AUC(
                curve="PR",
                name="pr_auc",
            )
        ],
    )

    return model


# ============================================================
# NEURAL NETWORK TUNING
# ============================================================

def tune_neural_network(
    X,
    y,
    cv,
):

    print("\n" + "=" * 72)
    print("NEURAL NETWORK HYPERPARAMETER TUNING")
    print("=" * 72)

    results = []

    for config_number, config in enumerate(
        NN_CONFIGS,
        start=1,
    ):

        print(
            f"\nNN configuration "
            f"{config_number}/"
            f"{len(NN_CONFIGS)}"
        )

        print(config)

        fold_scores = []
        fold_times = []
        fold_epochs = []

        for fold, (
            train_idx,
            val_idx,
        ) in enumerate(
            cv.split(X, y),
            start=1,
        ):

            # --------------------------------------------
            # SPLIT FOLD
            # --------------------------------------------

            X_train = X.iloc[
                train_idx
            ].copy()

            y_train = y.iloc[
                train_idx
            ].copy()

            X_val = X.iloc[
                val_idx
            ].copy()

            y_val = y.iloc[
                val_idx
            ].copy()

            # --------------------------------------------
            # FIT PREPROCESSING INSIDE FOLD
            # --------------------------------------------

            preprocessor = (
                build_preprocessor()
            )

            X_train_processed = (
                preprocessor.fit_transform(
                    X_train
                )
            )

            X_val_processed = (
                preprocessor.transform(
                    X_val
                )
            )

            X_train_processed = (
                np.asarray(
                    X_train_processed,
                    dtype=np.float32,
                )
            )

            X_val_processed = (
                np.asarray(
                    X_val_processed,
                    dtype=np.float32,
                )
            )

            # --------------------------------------------
            # RESET TENSORFLOW
            # --------------------------------------------

            tf.keras.backend.clear_session()

            seed = (
                RANDOM_STATE
                + config_number * 100
                + fold
            )

            set_random_seeds(
                seed
            )

            # --------------------------------------------
            # BUILD MODEL
            # --------------------------------------------

            model = build_neural_network(
                input_dim=
                    X_train_processed.shape[1],

                hidden_layers=
                    config["hidden_layers"],

                dropout=
                    config["dropout"],

                learning_rate=
                    config["learning_rate"],
            )

            early_stopping = (
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_pr_auc",
                    mode="max",
                    patience=NN_PATIENCE,
                    restore_best_weights=True,
                    verbose=0,
                )
            )

            # --------------------------------------------
            # TRAIN
            # --------------------------------------------

            start = time.perf_counter()

            history = model.fit(
                X_train_processed,

                np.asarray(
                    y_train,
                    dtype=np.float32,
                ),

                validation_data=(
                    X_val_processed,

                    np.asarray(
                        y_val,
                        dtype=np.float32,
                    ),
                ),

                epochs=
                    NN_MAX_EPOCHS,

                batch_size=
                    NN_BATCH_SIZE,

                callbacks=[
                    early_stopping
                ],

                verbose=0,

                shuffle=True,
            )

            probabilities = (
                model.predict(
                    X_val_processed,
                    batch_size=
                        NN_BATCH_SIZE,
                    verbose=0,
                )
                .reshape(-1)
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            score = (
                average_precision_score(
                    y_val,
                    probabilities,
                )
            )

            epochs_trained = len(
                history.history[
                    "loss"
                ]
            )

            fold_scores.append(
                score
            )

            fold_times.append(
                elapsed
            )

            fold_epochs.append(
                epochs_trained
            )

            print(
                f"  Fold {fold}: "
                f"PR-AUC={score:.6f} "
                f"epochs={epochs_trained} "
                f"time={elapsed:.2f}s"
            )

            del model
            del preprocessor

            tf.keras.backend.clear_session()

            gc.collect()

        result = {
            "model":
                "neural_network",

            "hidden_layers":
                str(
                    config[
                        "hidden_layers"
                    ]
                ),

            "dropout":
                config["dropout"],

            "learning_rate":
                config[
                    "learning_rate"
                ],

            "pr_auc_mean":
                float(
                    np.mean(
                        fold_scores
                    )
                ),

            "pr_auc_std":
                float(
                    np.std(
                        fold_scores,
                        ddof=1,
                    )
                ),

            "training_seconds_mean":
                float(
                    np.mean(
                        fold_times
                    )
                ),

            "epochs_mean":
                float(
                    np.mean(
                        fold_epochs
                    )
                ),
        }

        results.append(
            result
        )

        print(
            f"  Mean PR-AUC: "
            f"{result['pr_auc_mean']:.6f}"
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# BEST CONFIGURATION
# ============================================================

def get_best_result(df):

    best_index = (
        df["pr_auc_mean"]
        .idxmax()
    )

    return df.loc[
        best_index
    ]


# ============================================================
# MAIN
# ============================================================

def main():

    set_random_seeds()

    project_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    train_path = (
        project_root
        / "data"
        / "processed"
        / "splits"
        / "train.csv"
    )

    artifact_dir = (
        project_root
        / "artifacts"
        / "hyperparameter_tuning"
    )

    artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)

    print(
        "TASK 13 — HYPERPARAMETER TUNING"
    )

    print("=" * 72)

    print(
        "\nOnly train.csv will be used."
    )

    print(
        "validation.csv remains untouched."
    )

    print(
        "test.csv remains untouched."
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    print(
        "\nLoading training data..."
    )

    X, y = load_training_data(
        train_path
    )

    print(
        f"\nTraining shape : {X.shape}"
    )

    print(
        f"Legitimate     : "
        f"{int((y == 0).sum()):,}"
    )

    print(
        f"Fraud          : "
        f"{int((y == 1).sum()):,}"
    )

    print(
        f"Fraud rate     : "
        f"{y.mean() * 100:.4f}%"
    )

    # ========================================================
    # CV
    # ========================================================

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    rf_results = tune_random_forest(
        X,
        y,
        cv,
    )

    # ========================================================
    # XGBOOST
    # ========================================================

    xgb_results = tune_xgboost(
        X,
        y,
        cv,
    )

    # ========================================================
    # NEURAL NETWORK
    # ========================================================

    nn_results = tune_neural_network(
        X,
        y,
        cv,
    )

    # ========================================================
    # SAVE ALL RESULTS
    # ========================================================

    rf_path = (
        artifact_dir
        / "random_forest_tuning.csv"
    )

    xgb_path = (
        artifact_dir
        / "xgboost_tuning.csv"
    )

    nn_path = (
        artifact_dir
        / "neural_network_tuning.csv"
    )

    rf_results.to_csv(
        rf_path,
        index=False,
    )

    xgb_results.to_csv(
        xgb_path,
        index=False,
    )

    nn_results.to_csv(
        nn_path,
        index=False,
    )

    # ========================================================
    # BEST CONFIGURATIONS
    # ========================================================

    best_rf = get_best_result(
        rf_results
    )

    best_xgb = get_best_result(
        xgb_results
    )

    best_nn = get_best_result(
        nn_results
    )

    print("\n" + "=" * 90)

    print(
        "BEST HYPERPARAMETER CONFIGURATIONS"
    )

    print("=" * 90)

    print(
        "\nRandom Forest"
    )

    print(
        best_rf.to_string()
    )

    print(
        "\nXGBoost"
    )

    print(
        best_xgb.to_string()
    )

    print(
        "\nNeural Network"
    )

    print(
        best_nn.to_string()
    )

    # ========================================================
    # COMPARISON TABLE
    # ========================================================

    comparison = pd.DataFrame([
        {
            "model":
                "random_forest",

            "best_cv_pr_auc":
                best_rf[
                    "pr_auc_mean"
                ],

            "cv_std":
                best_rf[
                    "pr_auc_std"
                ],
        },

        {
            "model":
                "xgboost",

            "best_cv_pr_auc":
                best_xgb[
                    "pr_auc_mean"
                ],

            "cv_std":
                best_xgb[
                    "pr_auc_std"
                ],
        },

        {
            "model":
                "neural_network",

            "best_cv_pr_auc":
                best_nn[
                    "pr_auc_mean"
                ],

            "cv_std":
                best_nn[
                    "pr_auc_std"
                ],
        },
    ])

    comparison = (
        comparison
        .sort_values(
            "best_cv_pr_auc",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    print("\n" + "=" * 90)

    print(
        "TUNED MODEL CV COMPARISON"
    )

    print("=" * 90)

    print(
        comparison.to_string(
            index=False,
            float_format=
                lambda x: f"{x:.6f}",
        )
    )

    # ========================================================
    # SAVE BEST CONFIGS
    # ========================================================

    best_configs = {
        "random_forest": {
            "n_estimators":
                int(
                    best_rf[
                        "n_estimators"
                    ]
                ),

            "max_depth":
                (
                    None
                    if pd.isna(
                        best_rf[
                            "max_depth"
                        ]
                    )
                    else int(
                        best_rf[
                            "max_depth"
                        ]
                    )
                ),

            "min_samples_split":
                int(
                    best_rf[
                        "min_samples_split"
                    ]
                ),

            "max_features":
                (
                    float(
                        best_rf[
                            "max_features"
                        ]
                    )
                    if isinstance(
                        best_rf[
                            "max_features"
                        ],
                        (float, np.floating),
                    )
                    else best_rf[
                        "max_features"
                    ]
                ),
        },

        "xgboost": {
            "n_estimators":
                int(
                    best_xgb[
                        "n_estimators"
                    ]
                ),

            "learning_rate":
                float(
                    best_xgb[
                        "learning_rate"
                    ]
                ),

            "max_depth":
                int(
                    best_xgb[
                        "max_depth"
                    ]
                ),

            "subsample":
                float(
                    best_xgb[
                        "subsample"
                    ]
                ),

            "colsample_bytree":
                float(
                    best_xgb[
                        "colsample_bytree"
                    ]
                ),
        },

        "neural_network": {
            "hidden_layers":
                best_nn[
                    "hidden_layers"
                ],

            "dropout":
                float(
                    best_nn[
                        "dropout"
                    ]
                ),

            "learning_rate":
                float(
                    best_nn[
                        "learning_rate"
                    ]
                ),

            "epochs_mean":
                float(
                    best_nn[
                        "epochs_mean"
                    ]
                ),
        },
    }

    best_configs_path = (
        artifact_dir
        / "best_hyperparameters.json"
    )

    with open(
        best_configs_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            best_configs,
            file,
            indent=4,
        )

    comparison_path = (
        artifact_dir
        / "tuned_model_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    print(
        "\nSaved artifacts:"
    )

    print(
        f"RF tuning      : {rf_path}"
    )

    print(
        f"XGB tuning     : {xgb_path}"
    )

    print(
        f"NN tuning      : {nn_path}"
    )

    print(
        f"Best configs   : "
        f"{best_configs_path}"
    )

    print(
        f"Comparison     : "
        f"{comparison_path}"
    )

    print("\n" + "=" * 72)

    print(
        "TASK 13 — HYPERPARAMETER "
        "TUNING COMPLETED"
    )

    print("=" * 72)

    print(
        """
IMPORTANT

- Tuning used ONLY train.csv.
- validation.csv was NOT used.
- test.csv was NOT used.
- PR-AUC was the optimization objective.
- No SMOTE/resampling was used.
- No class weighting was used.
- No decision-threshold optimization was performed.
- Best parameters are CV-selected parameters, not final test results.
"""
    )


if __name__ == "__main__":
    main()