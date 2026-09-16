import os
import pandas as pd
import numpy as np
import joblib


BASE_DIR = r"D:\WhiteBox\SIH2026_IF"

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)


def main():

    score_file = os.path.join(
        RESULTS_DIR,
        "D2_final_test_enhanced_anomaly_scores.csv"
    )

    frozen_model_file = os.path.join(
        MODELS_DIR,
        "D2_final_frozen_model.pkl"
    )

    if not os.path.exists(score_file):
        raise FileNotFoundError(
            f"Score file not found:\n{score_file}"
        )

    if not os.path.exists(frozen_model_file):
        raise FileNotFoundError(
            f"Frozen model not found:\n{frozen_model_file}"
        )

    # ---------------------------------------------------------
    # Load frozen configuration
    # ---------------------------------------------------------

    frozen = joblib.load(
        frozen_model_file
    )

    threshold = float(
        frozen["threshold"]
    )

    # ---------------------------------------------------------
    # Load final-test anomaly scores
    # ---------------------------------------------------------

    df = pd.read_csv(
        score_file
    )

    # The score file contains score_0.01, score_0.03, etc.
    # All contamination models produce the same score_samples
    # values. The frozen configuration uses the 0.01 model.

    score_column = "score_0.01"

    if score_column not in df.columns:
        raise ValueError(
            f"{score_column} not found in score file."
        )

    anomaly_scores = df[
        score_column
    ].to_numpy()

    # ---------------------------------------------------------
    # Apply frozen threshold
    # ---------------------------------------------------------

    prediction = np.where(
        anomaly_scores >= threshold,
        1,
        0
    )

    prediction_label = np.where(
        prediction == 1,
        "ABNORMAL",
        "NORMAL"
    )

    # ---------------------------------------------------------
    # Add predictions
    # ---------------------------------------------------------

    df["anomaly_score"] = anomaly_scores

    df["prediction"] = prediction

    df["prediction_label"] = prediction_label

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    output_file = os.path.join(
        RESULTS_DIR,
        "D2_final_test_predictions.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    # ---------------------------------------------------------
    # Display
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL TEST PREDICTION")
    print("=" * 70)

    print(
        "\nFinal-test samples:",
        len(df)
    )

    print(
        "Anomaly score column:",
        score_column
    )

    print(
        f"Frozen threshold: {threshold:.6f}"
    )

    print("\nDecision rule:")

    print(
        f"Score >= {threshold:.6f} -> ABNORMAL"
    )

    print(
        f"Score <  {threshold:.6f} -> NORMAL"
    )

    print("\nPrediction counts:")

    print(
        pd.Series(
            prediction_label
        ).value_counts()
    )

    print("\nAnomaly score range:")

    print(
        f"Minimum: {anomaly_scores.min():.6f}"
    )

    print(
        f"Maximum: {anomaly_scores.max():.6f}"
    )

    print(
        f"Mean:    {anomaly_scores.mean():.6f}"
    )

    print(
        "\nPredictions saved:"
    )

    print(
        output_file
    )

    print("\n" + "=" * 70)
    print("STEP 24 COMPLETE")
    print("=" * 70)

    print(
        "\nFinal-test TARGET was not used "
        "to select or change the threshold."
    )


if __name__ == "__main__":
    main()