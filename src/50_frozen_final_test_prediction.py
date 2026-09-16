import os
import joblib
import numpy as np
import pandas as pd


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"

FEATURES_DIR = os.path.join(BASE_DIR, "features")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def main():

    print("\n" + "=" * 70)
    print("STEP 50 - FROZEN FINAL TEST PREDICTION")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load frozen model
    # ---------------------------------------------------------

    model_path = os.path.join(
        MODELS_DIR,
        "D2_v2_no_acceleration_frozen_model.pkl"
    )

    frozen = joblib.load(model_path)

    model = frozen["model"]
    preprocessing = frozen["preprocessing"]
    threshold = frozen["threshold"]

    selected_indices = preprocessing["selected_indices"]

    print("\nFrozen model loaded.")
    print("Features:", len(selected_indices))
    print("Threshold:", threshold)

    # ---------------------------------------------------------
    # Load FINAL TEST feature array
    # ---------------------------------------------------------

    final_test_path = os.path.join(
        FEATURES_DIR,
        "D2_enhanced_X_final_test.npy"
    )

    print("\nLoading:")
    print(final_test_path)

    X_final_full = np.load(final_test_path)

    print("Original FINAL TEST shape:")
    print(X_final_full.shape)

    # ---------------------------------------------------------
    # Select frozen 156 features
    # ---------------------------------------------------------

    X_final = X_final_full[:, selected_indices]

    print("Frozen FINAL TEST shape:")
    print(X_final.shape)

    # ---------------------------------------------------------
    # Apply TRAIN-FITTED preprocessing
    # ---------------------------------------------------------

    imputer = preprocessing["imputer"]
    scaler = preprocessing["scaler"]

    X_final_imputed = imputer.transform(
        X_final
    )

    X_final_processed = scaler.transform(
        X_final_imputed
    )

    # ---------------------------------------------------------
    # Calculate anomaly scores
    # ---------------------------------------------------------

    anomaly_scores = -model.score_samples(
        X_final_processed
    )

    print("\nAnomaly scores calculated.")
    print(
        "Score range:",
        anomaly_scores.min(),
        "to",
        anomaly_scores.max()
    )

    # ---------------------------------------------------------
    # Load FINAL TEST metadata
    # ---------------------------------------------------------

    final_test_csv = os.path.join(
        FEATURES_DIR,
        "D2_final_test_enhanced.csv"
    )

    df = pd.read_csv(final_test_csv)

    print("\nMetadata shape:")
    print(df.shape)

    # ---------------------------------------------------------
    # Verify alignment
    # ---------------------------------------------------------

    assert len(df) == len(anomaly_scores), (
        "Final-test CSV and feature array row counts do not match."
    )

    # ---------------------------------------------------------
    # Create row-level result
    # ---------------------------------------------------------

    result = df[
        [
            "MaterialID",
            "StepID",
            "duration_ms",
            "is_test",
            "target"
        ]
    ].copy()

    result["anomaly_score"] = anomaly_scores

    # ---------------------------------------------------------
    # Material-level aggregation
    # ---------------------------------------------------------

    material_scores = (
        result
        .groupby("MaterialID", as_index=False)
        ["anomaly_score"]
        .mean()
        .rename(
            columns={
                "anomaly_score": "mean_score"
            }
        )
    )

    # ---------------------------------------------------------
    # Apply FROZEN validation threshold
    # ---------------------------------------------------------

    material_scores["prediction"] = np.where(
        material_scores["mean_score"] >= threshold,
        "ABNORMAL",
        "NORMAL"
    )

    # ---------------------------------------------------------
    # Actual target is ONLY attached for evaluation
    # ---------------------------------------------------------

    material_targets = (
        result
        .groupby("MaterialID", as_index=False)
        ["target"]
        .first()
    )

    material_targets["actual"] = np.where(
        material_targets["target"] == 1,
        "ABNORMAL",
        "NORMAL"
    )

    final_result = material_scores.merge(
        material_targets[
            ["MaterialID", "actual"]
        ],
        on="MaterialID",
        how="left"
    )

    # ---------------------------------------------------------
    # Save final-test predictions
    # ---------------------------------------------------------

    output_path = os.path.join(
        RESULTS_DIR,
        "D2_v2_no_acceleration_final_test_predictions.csv"
    )

    final_result.to_csv(
        output_path,
        index=False
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("\nMaterial-level FINAL TEST predictions:")
    print(
        final_result["prediction"].value_counts()
    )

    print(
        "\nTotal FINAL TEST MaterialIDs:",
        len(final_result)
    )

    print(
        "\nPrediction file saved:"
    )
    print(output_path)

    print("\nFinal-test target used for prediction: NO")
    print("Final-test target used for threshold selection: NO")
    print("Threshold source: VALIDATION")

    print("\n" + "=" * 70)
    print("STEP 50 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()