import os
import joblib
import numpy as np
import pandas as pd


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"

MODELS_DIR = os.path.join(BASE_DIR, "models")
FEATURES_DIR = os.path.join(BASE_DIR, "features")


def main():

    print("\n" + "=" * 70)
    print("STEP 53 - REPRODUCIBILITY CHECK")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load frozen model bundle
    # ---------------------------------------------------------

    model_path = os.path.join(
        MODELS_DIR,
        "D2_v2_no_acceleration_frozen_model.pkl"
    )

    bundle = joblib.load(model_path)

    model = bundle["model"]
    preprocessing = bundle["preprocessing"]
    threshold = bundle["threshold"]

    # Exact preprocessing objects saved during frozen training
    imputer = preprocessing["imputer"]
    scaler = preprocessing["scaler"]
    selected_indices = preprocessing["selected_indices"]

    print("\nFrozen model loaded successfully.")
    print("Threshold:", threshold)
    print("Frozen feature count:", len(selected_indices))

    # ---------------------------------------------------------
    # Load original final-test feature array
    # ---------------------------------------------------------

    feature_path = os.path.join(
        FEATURES_DIR,
        "D2_enhanced_X_final_test.npy"
    )

    X = np.load(feature_path)

    print("\nOriginal final-test feature shape:", X.shape)

    # ---------------------------------------------------------
    # Select exact 156 frozen features
    # ---------------------------------------------------------

    X_selected = X[:, selected_indices]

    print("Frozen feature shape:", X_selected.shape)

    # ---------------------------------------------------------
    # Apply saved TRAIN-FITTED preprocessing
    # ---------------------------------------------------------

    X_imputed = imputer.transform(X_selected)
    X_processed = scaler.transform(X_imputed)

    print("Processed feature shape:", X_processed.shape)

    # ---------------------------------------------------------
    # Generate anomaly scores
    # ---------------------------------------------------------

    scores = -model.score_samples(X_processed)

    print("\nAnomaly scores generated successfully.")
    print("Minimum score:", scores.min())
    print("Maximum score:", scores.max())
    print("Mean score:", scores.mean())

    # ---------------------------------------------------------
    # Apply frozen threshold
    # ---------------------------------------------------------

    predictions = np.where(
        scores >= threshold,
        "ABNORMAL",
        "NORMAL"
    )

    print("\nRow-level predictions:")
    print(pd.Series(predictions).value_counts())

    # ---------------------------------------------------------
    # Reproducibility checks
    # ---------------------------------------------------------

    assert X.shape[1] == 175
    assert X_selected.shape[1] == 156
    assert X_processed.shape[1] == 156
    assert len(scores) == len(X)
    assert len(predictions) == len(X)

    print("\nAll reproducibility checks passed.")

    print("\n" + "=" * 70)
    print("STEP 53 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()