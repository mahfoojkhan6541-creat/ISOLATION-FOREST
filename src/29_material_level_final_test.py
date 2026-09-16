import os
import pandas as pd
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

    # ---------------------------------------------------------
    # Input files
    # ---------------------------------------------------------

    score_file = os.path.join(
        RESULTS_DIR,
        "D2_final_test_enhanced_anomaly_scores.csv"
    )

    config_file = os.path.join(
        MODELS_DIR,
        "D2_material_level_frozen_config.pkl"
    )

    if not os.path.exists(score_file):
        raise FileNotFoundError(
            f"Final test score file not found:\n{score_file}"
        )

    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"Frozen configuration not found:\n{config_file}"
        )

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    df = pd.read_csv(score_file)

    config = joblib.load(
        config_file
    )

    # ---------------------------------------------------------
    # Check required columns
    # ---------------------------------------------------------

    required_columns = [
        "MaterialID",
        "target",
        "score_0.01"
    ]

    for column in required_columns:

        if column not in df.columns:
            raise ValueError(
                f"Required column not found: {column}"
            )

    # ---------------------------------------------------------
    # Verify target is constant within MaterialID
    # ---------------------------------------------------------

    target_counts = (
        df.groupby("MaterialID")["target"]
        .nunique()
    )

    if (target_counts > 1).any():

        raise ValueError(
            "Some MaterialID groups contain multiple target values."
        )

    # ---------------------------------------------------------
    # MaterialID-level aggregation
    # ---------------------------------------------------------

    material_df = (
        df.groupby("MaterialID")
        .agg(
            mean_score=("score_0.01", "mean"),
            target=("target", "first"),
            row_count=("score_0.01", "size")
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # Apply FROZEN threshold
    # ---------------------------------------------------------

    threshold = float(
        config["threshold"]
    )

    material_df["prediction"] = (
        material_df["mean_score"] >= threshold
    ).astype(int)

    # ---------------------------------------------------------
    # Add readable labels
    # ---------------------------------------------------------

    material_df["actual_label"] = (
        material_df["target"]
        .map({
            0: "NORMAL",
            1: "ABNORMAL"
        })
    )

    material_df["predicted_label"] = (
        material_df["prediction"]
        .map({
            0: "NORMAL",
            1: "ABNORMAL"
        })
    )

    # ---------------------------------------------------------
    # Save predictions
    # ---------------------------------------------------------

    output_file = os.path.join(
        RESULTS_DIR,
        "D2_material_level_final_test_predictions.csv"
    )

    material_df.to_csv(
        output_file,
        index=False
    )

    # ---------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("MATERIALID-LEVEL FINAL TEST PREDICTION")
    print("=" * 70)

    print(
        f"\nFinal Test MaterialID groups: "
        f"{len(material_df)}"
    )

    print(
        f"Frozen threshold: "
        f"{threshold:.6f}"
    )

    print("\nActual:")
    print(
        material_df["actual_label"]
        .value_counts()
    )

    print("\nPredicted:")
    print(
        material_df["predicted_label"]
        .value_counts()
    )

    print("\nPrediction file saved:")
    print(output_file)

    print("\n" + "=" * 70)
    print("STEP 29 COMPLETE")
    print("=" * 70)

    print(
        "\nIMPORTANT:"
    )

    print(
        "The threshold was NOT changed using Final Test."
    )


if __name__ == "__main__":
    main()