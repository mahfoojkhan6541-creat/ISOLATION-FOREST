import os
import numpy as np
import joblib

from sklearn.ensemble import IsolationForest


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"

FEATURES_DIR = os.path.join(BASE_DIR, "features")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)


# Candidate contamination values.
# These are candidates, NOT the final-test abnormal percentage.
CONTAMINATION_VALUES = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20]


def train_dataset(dataset_name):

    print("\n" + "=" * 60)
    print(f"ISOLATION FOREST TRAINING - {dataset_name}")
    print("=" * 60)

    train_path = os.path.join(
        FEATURES_DIR,
        f"{dataset_name}_X_train.npy"
    )

    validation_path = os.path.join(
        FEATURES_DIR,
        f"{dataset_name}_X_validation.npy"
    )

    X_train = np.load(train_path)
    X_validation = np.load(validation_path)

    print("Training shape:", X_train.shape)
    print("Validation shape:", X_validation.shape)

    models = {}

    for contamination in CONTAMINATION_VALUES:

        print("\n" + "-" * 60)
        print(f"Training contamination = {contamination}")

        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train)

        # Isolation Forest:
        # -1 = anomaly
        # +1 = normal
        predictions = model.predict(X_validation)

        anomaly_count = np.sum(predictions == -1)
        normal_count = np.sum(predictions == 1)

        print("Validation NORMAL predictions :", normal_count)
        print("Validation ABNORMAL predictions:", anomaly_count)

        models[contamination] = model

    output_path = os.path.join(
        MODELS_DIR,
        f"{dataset_name}_if_candidates.pkl"
    )

    joblib.dump(
        models,
        output_path
    )

    print("\nSaved candidate models:")
    print(output_path)


if __name__ == "__main__":

    train_dataset("D1")
    train_dataset("D2")

    print("\n" + "=" * 60)
    print("STEP 10 COMPLETE")
    print("=" * 60)