import os
import numpy as np
import joblib

from sklearn.ensemble import IsolationForest


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"

FEATURES_DIR = os.path.join(BASE_DIR, "features")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


CONTAMINATION_VALUES = [
    0.01,
    0.03,
    0.05,
    0.10,
    0.15
]


def train_dataset(dataset_name):

    print("\n" + "=" * 70)
    print(f"ENHANCED ISOLATION FOREST TRAINING - {dataset_name}")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load TRAIN data only
    # ---------------------------------------------------------

    X_train = np.load(
        os.path.join(
            FEATURES_DIR,
            f"{dataset_name}_enhanced_X_train_processed.npy"
        )
    )

    print("Training data shape:", X_train.shape)

    # ---------------------------------------------------------
    # Safety checks
    # ---------------------------------------------------------

    if not np.isfinite(X_train).all():
        raise ValueError(
            "Training data contains NaN or infinite values."
        )

    print("Training data contains only finite values: YES")

    # ---------------------------------------------------------
    # Train candidate models
    # ---------------------------------------------------------

    models = {}

    for contamination in CONTAMINATION_VALUES:

        print(
            f"\nTraining Isolation Forest "
            f"(contamination={contamination})..."
        )

        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train)

        models[contamination] = model

        # Raw anomaly score
        scores = -model.score_samples(X_train)

        predictions = model.predict(X_train)

        anomaly_count = np.sum(
            predictions == -1
        )

        anomaly_percentage = (
            anomaly_count / len(predictions)
        ) * 100

        print(
            "Training anomaly count:",
            anomaly_count
        )

        print(
            f"Training anomaly percentage: "
            f"{anomaly_percentage:.2f}%"
        )

        print(
            f"Score range: "
            f"{scores.min():.6f} - "
            f"{scores.max():.6f}"
        )

    # ---------------------------------------------------------
    # Save candidate models
    # ---------------------------------------------------------

    model_file = os.path.join(
        MODELS_DIR,
        f"{dataset_name}_enhanced_if_candidates.pkl"
    )

    joblib.dump(
        models,
        model_file
    )

    print(
        "\nCandidate models saved:",
        model_file
    )

    print(
        "\nNumber of candidate models:",
        len(models)
    )


if __name__ == "__main__":

    train_dataset("D1")
    train_dataset("D2")

    print("\n" + "=" * 70)
    print("STEP 18 COMPLETE")
    print("=" * 70)