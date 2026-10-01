import ast
import random
import time

import numpy as np
import tensorflow as tf

from ml.src.features.preprocessing import (
    build_preprocessor,
)

RANDOM_STATE = 42
def train_neural_network(
    X_train,
    y_train,
    X_val,
    config,
):
    """
    Train the tuned TensorFlow/Keras neural network.

    Important:
    - Preprocessor is fitted ONLY on training data.
    - Validation data is transformed using the fitted preprocessor.
    - Validation labels are NOT used during model training.
    - No early stopping is performed on validation.csv.
    - Model trains for 27 epochs, based on Task 13 CV results.
    - No SMOTE.
    - No class weighting.
    - Returns validation probabilities for Task 14 threshold analysis.
    """

    print("\n" + "=" * 72)
    print("TRAINING TUNED NEURAL NETWORK")
    print("=" * 72)

    print(f"\nParameters from Task 13: {config}")

    # --------------------------------------------------------
    # 1. Read tuned hyperparameters
    # --------------------------------------------------------

    hidden_layers = config["hidden_layers"]

    # Task 13 JSON may store hidden_layers as:
    # "[64, 32]"
    #
    # Convert it safely back to a Python list.

    if isinstance(hidden_layers, str):
        hidden_layers = ast.literal_eval(hidden_layers)

    if not isinstance(hidden_layers, (list, tuple)):
        raise TypeError(
            "hidden_layers must be a list or tuple."
        )

    hidden_layers = [
        int(units)
        for units in hidden_layers
    ]

    dropout = float(
        config["dropout"]
    )

    learning_rate = float(
        config["learning_rate"]
    )

    # Task 13 mean epochs = 26.6667
    # Use rounded fixed epoch count.

    epochs = 27

    batch_size = 512

    print("\nNeural Network configuration:")
    print(f"Hidden layers : {hidden_layers}")
    print(f"Dropout       : {dropout}")
    print(f"Learning rate : {learning_rate}")
    print(f"Epochs        : {epochs}")
    print(f"Batch size    : {batch_size}")

    # --------------------------------------------------------
    # 2. Build preprocessing pipeline
    # --------------------------------------------------------

    print(
        "\nFitting preprocessing pipeline "
        "on training data only..."
    )

    preprocessor = build_preprocessor()

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

    # --------------------------------------------------------
    # 3. Convert to float32
    # --------------------------------------------------------

    X_train_processed = np.asarray(
        X_train_processed,
        dtype=np.float32,
    )

    X_val_processed = np.asarray(
        X_val_processed,
        dtype=np.float32,
    )

    y_train_array = np.asarray(
        y_train,
        dtype=np.float32,
    )

    print(
        f"\nProcessed training shape   : "
        f"{X_train_processed.shape}"
    )

    print(
        f"Processed validation shape : "
        f"{X_val_processed.shape}"
    )

    # --------------------------------------------------------
    # 4. Reset TensorFlow state
    # --------------------------------------------------------

    tf.keras.backend.clear_session()

    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    # --------------------------------------------------------
    # 5. Build neural network
    # --------------------------------------------------------

    model = tf.keras.Sequential(
        name="tuned_fraud_neural_network"
    )

    model.add(
        tf.keras.layers.Input(
            shape=(
                X_train_processed.shape[1],
            )
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

    # --------------------------------------------------------
    # 6. Compile model
    # --------------------------------------------------------

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),

        loss="binary_crossentropy",

        metrics=[
            tf.keras.metrics.AUC(
                curve="PR",
                name="pr_auc",
            ),

            tf.keras.metrics.AUC(
                curve="ROC",
                name="roc_auc",
            ),
        ],
    )

    print("\nModel architecture:")

    model.summary()

    # --------------------------------------------------------
    # 7. Train model
    # --------------------------------------------------------

    print(
        "\nTraining neural network..."
    )

    print(
        "Validation labels are NOT being used "
        "for training or early stopping."
    )

    start = time.perf_counter()

    history = model.fit(
    X_train_processed,
    y_train_array,
    epochs=27,
    batch_size=512,
    verbose=2,
    shuffle=True,
    )
    training_seconds = (
        time.perf_counter()
        - start
    )

    print(
        f"\nTraining completed in "
        f"{training_seconds:.2f} seconds."
    )

    # --------------------------------------------------------
    # 8. Generate validation probabilities
    # --------------------------------------------------------

    print(
        "\nGenerating validation probabilities..."
    )

    probabilities = (
        model.predict(
            X_val_processed,
            batch_size=batch_size,
            verbose=0,
        )
        .reshape(-1)
    )

    # --------------------------------------------------------
    # 9. Validation checks
    # --------------------------------------------------------

    if len(probabilities) != len(X_val):
        raise RuntimeError(
            "Validation prediction length mismatch."
        )

    if not np.isfinite(
        probabilities
    ).all():

        raise RuntimeError(
            "Neural network generated "
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

    # --------------------------------------------------------
    # 10. Training diagnostics
    # --------------------------------------------------------

    epochs_trained = len(
        history.history["loss"]
    )

    final_training_loss = float(
        history.history["loss"][-1]
    )

    final_training_pr_auc = float(
        history.history["pr_auc"][-1]
    )

    final_training_roc_auc = float(
        history.history["roc_auc"][-1]
    )

    print("\nTraining diagnostics:")

    print(
        f"Epochs trained        : "
        f"{epochs_trained}"
    )

    print(
        f"Final training loss   : "
        f"{final_training_loss:.6f}"
    )

    print(
        f"Final training PR-AUC : "
        f"{final_training_pr_auc:.6f}"
    )

    print(
        f"Final training ROC-AUC: "
        f"{final_training_roc_auc:.6f}"
    )

    print(
        f"\nValidation probability range: "
        f"{probabilities.min():.8f} "
        f"to "
        f"{probabilities.max():.8f}"
    )

    # --------------------------------------------------------
    # 11. Return everything Task 14 needs
    # --------------------------------------------------------

    return (
        model,
        preprocessor,
        probabilities,
        training_seconds,
        epochs_trained,
    )