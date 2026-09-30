"""
Task 13 Correction — Leakage-Safe Neural Network Hyperparameter Tuning

This script corrects the neural-network model-selection methodology.

Method
------
For every hyperparameter configuration and outer CV fold:

1. Outer training and validation folds are created.
2. Outer training is split again into:
      - inner training
      - inner early-stopping validation
3. Preprocessing for epoch selection is fitted only on inner training.
4. Early stopping uses ONLY inner validation.
5. The best epoch is selected.
6. A fresh preprocessor is fitted on the FULL outer-training fold.
7. A fresh neural network is trained on the FULL outer-training fold
   for the selected number of epochs.
8. Outer validation is evaluated exactly once.

validation.csv and test.csv are never loaded.
"""

from pathlib import Path
import ast
import gc
import json
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import average_precision_score
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
)

from ml.src.features.preprocessing import build_preprocessor


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

N_SPLITS = 3

INNER_VALIDATION_SIZE = 0.15

NN_MAX_EPOCHS = 50

NN_PATIENCE = 7

NN_BATCH_SIZE = 512


# These are the configurations used for the original Task 13
# neural-network tuning.
#
# If your original NN_CONFIGS differ, replace ONLY this list
# with the original values from tune_models.py.

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
# REPRODUCIBILITY
# ============================================================

def set_random_seeds(seed=RANDOM_STATE):

    random.seed(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)


# ============================================================
# DATA
# ============================================================

def load_training_data(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Training dataset not found: {path}"
        )

    df = pd.read_csv(path)

    if "Class" not in df.columns:

        raise ValueError(
            "Target column 'Class' not found."
        )

    if df.empty:

        raise ValueError(
            "Training dataset is empty."
        )

    X = df.drop(
        columns=["Class"]
    )

    y = df[
        "Class"
    ].astype(int)

    if set(
        y.unique()
    ) != {0, 1}:

        raise ValueError(
            "Class must contain both 0 and 1."
        )

    return X, y


# ============================================================
# BUILD NEURAL NETWORK
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
                int(units),
                activation="relu",
            )
        )

        model.add(
            tf.keras.layers.Dropout(
                float(dropout)
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
            learning_rate=float(
                learning_rate
            )
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(
                curve="PR",
                name="pr_auc",
            )
        ],
    )

    return model


# ============================================================
# SELECT EPOCH USING INNER VALIDATION
# ============================================================

def select_best_epoch(
    X_outer_train,
    y_outer_train,
    config,
    seed,
):

    (
        X_inner_train,
        X_inner_val,
        y_inner_train,
        y_inner_val,
    ) = train_test_split(
        X_outer_train,
        y_outer_train,
        test_size=INNER_VALIDATION_SIZE,
        stratify=y_outer_train,
        random_state=seed,
    )

    # --------------------------------------------------------
    # Preprocessing is fitted ONLY on inner training.
    # --------------------------------------------------------

    preprocessor = build_preprocessor()

    X_inner_train_processed = (
        preprocessor.fit_transform(
            X_inner_train
        )
    )

    X_inner_val_processed = (
        preprocessor.transform(
            X_inner_val
        )
    )

    X_inner_train_processed = np.asarray(
        X_inner_train_processed,
        dtype=np.float32,
    )

    X_inner_val_processed = np.asarray(
        X_inner_val_processed,
        dtype=np.float32,
    )

    y_inner_train_array = np.asarray(
        y_inner_train,
        dtype=np.float32,
    )

    y_inner_val_array = np.asarray(
        y_inner_val,
        dtype=np.float32,
    )

    tf.keras.backend.clear_session()

    set_random_seeds(seed)

    model = build_neural_network(
        input_dim=
            X_inner_train_processed.shape[1],

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

    history = model.fit(
        X_inner_train_processed,
        y_inner_train_array,

        validation_data=(
            X_inner_val_processed,
            y_inner_val_array,
        ),

        epochs=NN_MAX_EPOCHS,

        batch_size=NN_BATCH_SIZE,

        callbacks=[
            early_stopping
        ],

        verbose=0,

        shuffle=True,
    )

    # --------------------------------------------------------
    # Determine epoch with highest INNER validation PR-AUC.
    #
    # argmax is zero-based, so add 1.
    # --------------------------------------------------------

    validation_pr_auc = np.asarray(
        history.history[
            "val_pr_auc"
        ],
        dtype=float,
    )

    if validation_pr_auc.size == 0:

        raise RuntimeError(
            "No inner validation PR-AUC "
            "values were produced."
        )

    best_epoch = (
        int(
            np.argmax(
                validation_pr_auc
            )
        )
        + 1
    )

    best_inner_pr_auc = float(
        validation_pr_auc[
            best_epoch - 1
        ]
    )

    epochs_observed = len(
        history.history["loss"]
    )

    del model
    del preprocessor

    tf.keras.backend.clear_session()

    gc.collect()

    return (
        best_epoch,
        best_inner_pr_auc,
        epochs_observed,
    )


# ============================================================
# TRAIN FULL OUTER TRAINING FOLD
# ============================================================

def train_outer_model(
    X_outer_train,
    y_outer_train,
    X_outer_val,
    config,
    selected_epoch,
    seed,
):

    # --------------------------------------------------------
    # Fresh preprocessing.
    #
    # This time it is fitted on ALL outer-training data.
    # --------------------------------------------------------

    preprocessor = build_preprocessor()

    X_outer_train_processed = (
        preprocessor.fit_transform(
            X_outer_train
        )
    )

    X_outer_val_processed = (
        preprocessor.transform(
            X_outer_val
        )
    )

    X_outer_train_processed = np.asarray(
        X_outer_train_processed,
        dtype=np.float32,
    )

    X_outer_val_processed = np.asarray(
        X_outer_val_processed,
        dtype=np.float32,
    )

    y_outer_train_array = np.asarray(
        y_outer_train,
        dtype=np.float32,
    )

    tf.keras.backend.clear_session()

    set_random_seeds(seed)

    model = build_neural_network(
        input_dim=
            X_outer_train_processed.shape[1],

        hidden_layers=
            config["hidden_layers"],

        dropout=
            config["dropout"],

        learning_rate=
            config["learning_rate"],
    )

    start = time.perf_counter()

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # No validation_data.
    # No EarlyStopping.
    # Outer validation labels are never supplied to fit().
    # --------------------------------------------------------

    model.fit(
        X_outer_train_processed,
        y_outer_train_array,

        epochs=int(
            selected_epoch
        ),

        batch_size=NN_BATCH_SIZE,

        verbose=0,

        shuffle=True,
    )

    probabilities = (
        model.predict(
            X_outer_val_processed,
            batch_size=NN_BATCH_SIZE,
            verbose=0,
        )
        .reshape(-1)
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    if not np.isfinite(
        probabilities
    ).all():

        raise RuntimeError(
            "Neural network produced "
            "non-finite probabilities."
        )

    if (
        probabilities.min() < 0
        or probabilities.max() > 1
    ):

        raise RuntimeError(
            "Neural network probabilities "
            "must be between 0 and 1."
        )

    return (
        model,
        preprocessor,
        probabilities,
        elapsed,
    )


# ============================================================
# CORRECTED NN TUNING
# ============================================================

def tune_neural_network(
    X,
    y,
):

    print(
        "\n"
        + "=" * 72
    )

    print(
        "CORRECTED NEURAL NETWORK HYPERPARAMETER TUNING"
    )

    print(
        "=" * 72
    )

    outer_cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = []

    fold_details = []

    for (
        config_number,
        config,
    ) in enumerate(
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

        selected_epochs = []

        for fold, (
            outer_train_idx,
            outer_val_idx,
        ) in enumerate(
            outer_cv.split(
                X,
                y,
            ),
            start=1,
        ):

            X_outer_train = (
                X.iloc[
                    outer_train_idx
                ]
                .copy()
            )

            y_outer_train = (
                y.iloc[
                    outer_train_idx
                ]
                .copy()
            )

            X_outer_val = (
                X.iloc[
                    outer_val_idx
                ]
                .copy()
            )

            y_outer_val = (
                y.iloc[
                    outer_val_idx
                ]
                .copy()
            )

            selection_seed = (
                RANDOM_STATE
                + config_number * 1000
                + fold * 10
            )

            final_seed = (
                selection_seed
                + 1
            )

            # ================================================
            # INNER MODEL SELECTION
            # ================================================

            (
                best_epoch,
                best_inner_pr_auc,
                epochs_observed,
            ) = select_best_epoch(
                X_outer_train,
                y_outer_train,
                config,
                selection_seed,
            )

            # ================================================
            # FRESH OUTER MODEL
            # ================================================

            (
                model,
                preprocessor,
                probabilities,
                elapsed,
            ) = train_outer_model(
                X_outer_train,
                y_outer_train,
                X_outer_val,
                config,
                best_epoch,
                final_seed,
            )

            # ================================================
            # OUTER VALIDATION EVALUATED ONCE
            # ================================================

            outer_pr_auc = (
                average_precision_score(
                    y_outer_val,
                    probabilities,
                )
            )

            fold_scores.append(
                outer_pr_auc
            )

            fold_times.append(
                elapsed
            )

            selected_epochs.append(
                best_epoch
            )

            fold_details.append({
                "config_number":
                    config_number,

                "fold":
                    fold,

                "hidden_layers":
                    str(
                        config[
                            "hidden_layers"
                        ]
                    ),

                "dropout":
                    config[
                        "dropout"
                    ],

                "learning_rate":
                    config[
                        "learning_rate"
                    ],

                "selected_epoch":
                    best_epoch,

                "inner_best_pr_auc":
                    best_inner_pr_auc,

                "inner_epochs_observed":
                    epochs_observed,

                "outer_pr_auc":
                    float(
                        outer_pr_auc
                    ),

                "outer_training_seconds":
                    float(
                        elapsed
                    ),
            })

            print(
                f"  Fold {fold}: "
                f"selected_epoch="
                f"{best_epoch} "
                f"inner_PR-AUC="
                f"{best_inner_pr_auc:.6f} "
                f"outer_PR-AUC="
                f"{outer_pr_auc:.6f} "
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
                config[
                    "dropout"
                ],

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
                        selected_epochs
                    )
                ),

            "epochs_median":
                float(
                    np.median(
                        selected_epochs
                    )
                ),
        }

        results.append(
            result
        )

        print(
            f"\n  Mean OUTER PR-AUC: "
            f"{result['pr_auc_mean']:.6f}"
        )

        print(
            f"  Std OUTER PR-AUC : "
            f"{result['pr_auc_std']:.6f}"
        )

        print(
            f"  Mean selected epoch: "
            f"{result['epochs_mean']:.2f}"
        )

    return (
        pd.DataFrame(
            results
        ),
        pd.DataFrame(
            fold_details
        ),
    )


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
        / "nn_corrected"
    )

    artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 72
    )

    print(
        "TASK 13 CORRECTION — NESTED NN TUNING"
    )

    print(
        "=" * 72
    )

    print(
        "\nOnly train.csv will be loaded."
    )

    print(
        "validation.csv will NOT be loaded."
    )

    print(
        "test.csv will NOT be loaded."
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

    (
        results,
        fold_details,
    ) = tune_neural_network(
        X,
        y,
    )

    best_index = (
        results[
            "pr_auc_mean"
        ]
        .idxmax()
    )

    best = results.loc[
        best_index
    ]

    print(
        "\n"
        + "=" * 90
    )

    print(
        "CORRECTED NN TUNING RESULTS"
    )

    print(
        "=" * 90
    )

    print(
        results.to_string(
            index=False,
            float_format=
                lambda x: f"{x:.6f}",
        )
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "SELECTED NN CONFIGURATION"
    )

    print(
        "=" * 90
    )

    print(
        best.to_string()
    )

    results_path = (
        artifact_dir
        / "neural_network_tuning_corrected.csv"
    )

    folds_path = (
        artifact_dir
        / "neural_network_fold_details.csv"
    )

    config_path = (
        artifact_dir
        / "best_neural_network_corrected.json"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    fold_details.to_csv(
        folds_path,
        index=False,
    )

    hidden_layers = (
        ast.literal_eval(
            best[
                "hidden_layers"
            ]
        )
    )

    best_config = {
        "hidden_layers":
            hidden_layers,

        "dropout":
            float(
                best[
                    "dropout"
                ]
            ),

        "learning_rate":
            float(
                best[
                    "learning_rate"
                ]
            ),

        "epochs_mean":
            float(
                best[
                    "epochs_mean"
                ]
            ),

        "epochs_median":
            float(
                best[
                    "epochs_median"
                ]
            ),

        "cv_pr_auc_mean":
            float(
                best[
                    "pr_auc_mean"
                ]
            ),

        "cv_pr_auc_std":
            float(
                best[
                    "pr_auc_std"
                ]
            ),

        "methodology":
            (
                "Outer stratified CV with "
                "inner stratified early-stopping "
                "split. Outer validation folds "
                "are evaluated exactly once."
            ),
    }

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            best_config,
            file,
            indent=4,
        )

    print(
        "\nSaved:"
    )

    print(
        results_path
    )

    print(
        folds_path
    )

    print(
        config_path
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "TASK 13 NN CORRECTION COMPLETED"
    )

    print(
        "=" * 72
    )

    print(
        """
IMPORTANT

- Only train.csv was used.
- validation.csv was NOT loaded.
- test.csv was NOT loaded.
- Inner validation selected the training epoch.
- Outer validation was NOT used for EarlyStopping.
- A fresh model was trained on the full outer-training fold.
- Outer validation was evaluated exactly once.
- PR-AUC remained the optimization objective.
- No SMOTE/resampling was used.
- No class weighting was used.
- No decision-threshold optimization was performed.
"""
    )


if __name__ == "__main__":
    main()